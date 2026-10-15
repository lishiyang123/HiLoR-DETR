"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register
from .qccr import QueryCompetitionConditionalRecalibration
from .umr import UMR


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

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
        umr_topk=300,
        umr_tau=0.0,
        umr_beta=2.0,
        umr_checkpoint=None,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.qccr_enabled = bool(qccr_enabled)

        if self.qccr_enabled and not self.use_focal_loss:
            raise ValueError(
                "QCCR currently requires use_focal_loss=True"
            )

        if self.qccr_enabled:
            self.qccr = (
                QueryCompetitionConditionalRecalibration(
                    beta0=float(qccr_beta0),
                    beta_positive=float(qccr_beta_positive),
                    beta_negative=float(qccr_beta_negative),
                    duplicate_penalty=float(
                        qccr_duplicate_penalty
                    ),
                )
            )
        else:
            self.qccr = None

        self.umr_enabled = bool(umr_enabled)
        if self.umr_enabled:
            if not self.qccr_enabled:
                raise ValueError("UMR requires qccr_enabled=True")
            self.umr = UMR(
                num_classes=self.num_classes,
                num_queries=self.num_top_queries,
                topk=int(umr_topk),
                tau=float(umr_tau),
                beta=float(umr_beta),
                beta0=float(qccr_beta0),
                beta_positive=float(qccr_beta_positive),
                beta_negative=float(qccr_beta_negative),
                duplicate_penalty=float(qccr_duplicate_penalty),
                checkpoint=umr_checkpoint,
            )
        else:
            self.umr = None

        self.deploy_mode = False
        # CUDA Graphs remove Python/kernel-launch overhead from the fixed-shape
        # UMR inference path without changing any UMR or QCCR arithmetic. The
        # cache is deliberately runtime-only: it is neither a buffer nor part
        # of the formal detector checkpoint.
        self._umr_cuda_graph_cache = {}

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, '
            f'num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, '
            f'qccr_enabled={self.qccr_enabled}, '
            f'umr_enabled={self.umr_enabled}'
        )

    def state_dict(self, *args, **kwargs):
        # The PostProcessor is inference-only. Its UMR head weights are loaded
        # from a dedicated frozen weight file (umr_checkpoint) and are
        # intentionally excluded from the model checkpoint so that strict
        # loading of detector checkpoints (whose postprocessor state
        # is empty) keeps missing=0 / unexpected=0.
        return OrderedDict()

    def load_state_dict(self, state_dict, strict=True):
        # Detector checkpoints carry an empty postprocessor state. The UMR head
        # is restored from its own frozen weight file, so nothing to load here.
        if strict and state_dict:
            raise RuntimeError(
                "PostProcessor expects an empty checkpoint state; "
                f"received keys: {sorted(state_dict)}"
            )
        return None

    def _run_umr(self, logits, boxes, bbox_pred, decoder_features):
        """Run UMR eagerly or replay an exact fixed-shape CUDA graph."""
        inputs = (logits, boxes, bbox_pred, decoder_features)
        graph_eligible = (
            not self.training
            and logits.is_cuda
            and all(tensor.is_cuda for tensor in inputs)
            and not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.cuda.is_current_stream_capturing()
        )
        if not graph_eligible:
            return self.umr(*inputs)

        signature = tuple(
            (
                tuple(tensor.shape),
                tensor.dtype,
                tuple(tensor.stride()),
                tensor.device.index,
            )
            for tensor in inputs
        )
        cached = self._umr_cuda_graph_cache.get(signature)
        if cached is None:
            static_inputs = tuple(tensor.detach().clone() for tensor in inputs)

            # Warm the exact eager path on a side stream so all lazy CUDA
            # allocations are complete before capture.
            current_stream = torch.cuda.current_stream(logits.device)
            warmup_stream = torch.cuda.Stream(device=logits.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    static_output = self.umr(*static_inputs)
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(logits.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self.umr(*static_inputs)
            cached = (graph, static_inputs, static_output)
            self._umr_cuda_graph_cache[signature] = cached

        graph, static_inputs, static_output = cached
        for static_input, current_input in zip(static_inputs, inputs):
            static_input.copy_(current_input)
        graph.replay()
        return static_output

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            # QCCR operates on all Q x C raw candidates before
            # sigmoid and the global Top-K selection. Boxes are
            # used only to compute competition risk and are never
            # modified by QCCR.
            if self.umr is not None:
                # UMR fusion: baseline QCCR -> features -> correct raw
                # logits -> recompute QCCR. Boxes are never modified.
                logits = self._run_umr(
                    logits, boxes, bbox_pred, outputs['umr_decoder_features']
                )
            elif self.qccr is not None:
                logits = self.qccr(logits, boxes)

            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
