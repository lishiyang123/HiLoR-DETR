"""RT-DETR postprocessor with an explicit, bypassable QCCR-UMR insertion."""

import torch
import torch.nn.functional as F
import torchvision

from src.core import register

from .hilor_artifacts import load_bound_inference_artifact
from .hilor_qccr import QueryCompetitionConditionalRecalibration
from .hilor_umr import UMR
from .rtdetr_postprocessor import RTDETRPostProcessor


__all__ = ["HiLoRRTDETRPostProcessor"]


@register
class HiLoRRTDETRPostProcessor(RTDETRPostProcessor):
    """Score-only adapter before native sigmoid + global QxC Top-K.

    When both methods are disabled, ``forward`` immediately delegates to the
    untouched official implementation. This avoids sending an identity QCCR
    through softplus and is required for bit-exact native replay.
    """

    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        qccr_enabled=False,
        qccr_beta0=0.0,
        qccr_beta_positive=1.0,
        qccr_beta_negative=1.0,
        qccr_duplicate_penalty=0.0,
        umr_enabled=False,
        umr_checkpoint=None,
        umr_topk=300,
        umr_tau=0.0,
        umr_beta=2.0,
        num_queries=300,
        artifact_manifest=None,
        expected_detector_checkpoint_sha256=None,
        allow_unbound_artifacts=False,
    ):
        super().__init__(
            num_classes=num_classes,
            use_focal_loss=use_focal_loss,
            num_top_queries=num_top_queries,
            remap_mscoco_category=remap_mscoco_category,
        )
        self.qccr_enabled = bool(qccr_enabled)
        self.umr_enabled = bool(umr_enabled)
        if self.umr_enabled and not self.qccr_enabled:
            raise ValueError("UMR requires the fitted QCCR mapping")

        bound = None
        if self.qccr_enabled and not bool(allow_unbound_artifacts):
            if not artifact_manifest:
                raise ValueError("active QCCR–UMR requires a bound inference manifest")
            bound = load_bound_inference_artifact(
                artifact_manifest, require_umr=self.umr_enabled
            )
            if expected_detector_checkpoint_sha256 != bound["detector_checkpoint_sha256"]:
                raise RuntimeError("configured detector checkpoint does not match the binding")
            qccr_beta0 = bound["qccr_parameters"]["beta0"]
            qccr_beta_positive = bound["qccr_parameters"]["beta_positive"]
            qccr_beta_negative = bound["qccr_parameters"]["beta_negative"]
            qccr_duplicate_penalty = bound["qccr_parameters"]["duplicate_penalty"]
            self.artifact_manifest_sha256 = bound["manifest_sha256"]
            self.detector_checkpoint_sha256 = bound["detector_checkpoint_sha256"]
            if self.umr_enabled:
                umr_checkpoint = bound["umr"]["checkpoint"]
                umr_topk = bound["umr"]["topk"]
                umr_tau = bound["umr"]["tau"]
                umr_beta = bound["umr"]["beta"]
        else:
            self.artifact_manifest_sha256 = None
            self.detector_checkpoint_sha256 = expected_detector_checkpoint_sha256

        self.qccr = None
        self.umr = None
        if self.umr_enabled:
            self.umr = UMR(
                checkpoint=umr_checkpoint,
                num_classes=num_classes,
                num_queries=num_queries,
                topk=umr_topk,
                tau=umr_tau,
                beta=umr_beta,
                beta0=qccr_beta0,
                beta_positive=qccr_beta_positive,
                beta_negative=qccr_beta_negative,
                duplicate_penalty=qccr_duplicate_penalty,
                expected_checkpoint_sha256=(
                    bound["umr"]["checkpoint_sha256"] if bound is not None else None
                ),
            )
        elif self.qccr_enabled:
            self.qccr = QueryCompetitionConditionalRecalibration(
                beta0=qccr_beta0,
                beta_positive=qccr_beta_positive,
                beta_negative=qccr_beta_negative,
                duplicate_penalty=qccr_duplicate_penalty,
            )

    def forward(self, outputs, orig_target_sizes):
        if not self.qccr_enabled and not self.umr_enabled:
            return super().forward(outputs, orig_target_sizes)
        if not self.use_focal_loss:
            raise RuntimeError("QCCR-UMR requires RT-DETR's focal QxC candidate pool")

        raw_logits = outputs["pred_logits"]
        normalized_boxes = outputs["pred_boxes"]
        if raw_logits.ndim != 3 or normalized_boxes.shape != (*raw_logits.shape[:2], 4):
            raise RuntimeError("HiLoR postprocessor expects logits [B,Q,C] and boxes [B,Q,4]")
        if raw_logits.shape[-1] != self.num_classes:
            raise RuntimeError("postprocessor class count does not match detector logits")
        if self.num_top_queries > raw_logits.shape[1] * raw_logits.shape[2]:
            raise RuntimeError("num_top_queries exceeds the QxC candidate pool")
        if orig_target_sizes.shape != (raw_logits.shape[0], 2):
            raise RuntimeError("orig_target_sizes must be [B,2] in width-height order")
        absolute_boxes = torchvision.ops.box_convert(
            normalized_boxes, in_fmt="cxcywh", out_fmt="xyxy"
        )
        absolute_boxes = absolute_boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.umr_enabled:
            if "hilor_decoder_features" not in outputs:
                raise RuntimeError(
                    "UMR requires hilor_decoder_features; enable the decoder capture interface"
                )
            logits = self.umr(
                raw_logits,
                normalized_boxes,
                absolute_boxes,
                outputs["hilor_decoder_features"],
            )
        else:
            logits = self.qccr(raw_logits, normalized_boxes)

        scores = F.sigmoid(logits)
        scores, flat_index = torch.topk(
            scores.flatten(1), self.num_top_queries, axis=-1
        )
        labels = flat_index % self.num_classes
        query_index = flat_index // self.num_classes
        boxes = absolute_boxes.gather(
            dim=1,
            index=query_index.unsqueeze(-1).repeat(1, 1, absolute_boxes.shape[-1]),
        )

        if self.deploy_mode:
            return labels, boxes, scores
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category

            labels = torch.tensor(
                [mscoco_label2category[int(value.item())] for value in labels.flatten()]
            ).to(boxes.device).reshape(labels.shape)
        return [
            dict(labels=image_labels, boxes=image_boxes, scores=image_scores)
            for image_labels, image_boxes, image_scores in zip(labels, boxes, scores)
        ]
