"""RT-DETR-specific host adapter for GCCS classification-role handover.

The official matcher remains authoritative. A second detached, geometry-only
Hungarian assignment is computed exclusively on native-unmatched queries and
is used only to add a bounded correction to the final-layer VFL term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.core import register
from src.misc.dist import get_world_size, is_dist_available_and_initialized

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .hilor_ufd import RTDETRUFD
from .rtdetr_criterion import SetCriterion


__all__ = ["HiLoRSetCriterion"]


@register
class HiLoRSetCriterion(RTDETRUFD, SetCriterion):
    """Official RT-DETR criterion plus final-VFL GCCS role correction."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]
    requires_step_context = True

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=80,
        lcct_enabled=True,
        lcct_warmup_epoch=20,
        lcct_delta=0.02,
        lcct_continuous_handover=True,
        lcct_transition_width=0.02,
        lcct_ramp_epochs=5,
        ufd_enabled=True,
        ufd_cdf_path=None,
        ufd_total_epochs=102,
        ufd_k=9,
        ufd_base_size=640.0,
        ufd_alpha_h=0.5,
        ufd_alpha_u_max=0.0,
        ufd_lambda_max=0.75,
        ufd_lambda_ramp_fraction=0.05,
        ufd_reg_max=32,
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            eos_coef=eos_coef,
            num_classes=num_classes,
        )
        self.lcct_enabled = bool(lcct_enabled)
        self.lcct_warmup_epoch = int(lcct_warmup_epoch)
        self.lcct_delta = float(lcct_delta)
        self.lcct_continuous_handover = bool(lcct_continuous_handover)
        self.lcct_transition_width = float(lcct_transition_width)
        self.lcct_ramp_epochs = int(lcct_ramp_epochs)
        if self.lcct_warmup_epoch < 0 or self.lcct_delta < 0.0:
            raise ValueError("LCCT warmup and delta must be non-negative")
        if self.lcct_continuous_handover and (
            self.lcct_transition_width <= 0.0 or self.lcct_ramp_epochs <= 0
        ):
            raise ValueError("continuous LCCT requires positive width and ramp epochs")
        self._hilor_handover_pairs = []
        self._hilor_last_diag = {}
        self._init_ufd(
            ufd_enabled=ufd_enabled,
            ufd_cdf_path=ufd_cdf_path,
            ufd_total_epochs=ufd_total_epochs,
            ufd_k=ufd_k,
            ufd_base_size=ufd_base_size,
            ufd_alpha_h=ufd_alpha_h,
            ufd_alpha_u_max=ufd_alpha_u_max,
            ufd_lambda_max=ufd_lambda_max,
            ufd_lambda_ramp_fraction=ufd_lambda_ramp_fraction,
            ufd_reg_max=ufd_reg_max,
        )

    def _ramp_progress(self, epoch: int) -> float:
        if not self.lcct_continuous_handover:
            return 1.0
        return min(
            max(
                (float(epoch) - float(self.lcct_warmup_epoch))
                / float(self.lcct_ramp_epochs),
                0.0,
            ),
            1.0,
        )

    @torch.no_grad()
    def _build_handover_pairs(self, outputs, targets, native_indices, epoch):
        self._hilor_handover_pairs = []
        native_positive_count = sum(int(source.numel()) for source, _ in native_indices)
        self._hilor_last_diag = {
            "native_positive_count": native_positive_count,
            "active_count": 0,
            "effective_count": 0,
            "ramp_progress": 0.0,
            "mean_absolute_iou_gain": 0.0,
            "dense_images_skipped": 0,
        }
        if not self.lcct_enabled or not self.training:
            return
        if epoch is None:
            raise RuntimeError("GCCS requires epoch context during training")
        epoch = int(epoch)
        if epoch < self.lcct_warmup_epoch:
            return
        ramp_progress = self._ramp_progress(epoch)
        self._hilor_last_diag["ramp_progress"] = ramp_progress
        if self.lcct_continuous_handover and ramp_progress <= 0.0:
            return

        pred_boxes = outputs["pred_boxes"]
        batch_size, num_queries, _ = pred_boxes.shape
        if len(targets) != batch_size or len(native_indices) != batch_size:
            raise RuntimeError("GCCS batch dimensions do not agree")

        active_gains = []
        effective_count = 0
        active_count = 0
        for batch_index, target in enumerate(targets):
            num_gt = int(target["labels"].numel())
            source_native, target_native = native_indices[batch_index]
            if num_gt == 0:
                if source_native.numel() or target_native.numel():
                    raise RuntimeError("native matcher returned positives for empty GT")
                continue
            if num_queries < 2 * num_gt:
                self._hilor_last_diag["dense_images_skipped"] += 1
                continue
            if source_native.numel() != num_gt or target_native.numel() != num_gt:
                raise RuntimeError("GCCS requires one native query per GT")

            device = pred_boxes.device
            source_native = source_native.to(device=device, dtype=torch.long)
            target_native = target_native.to(device=device, dtype=torch.long)
            expected_targets = torch.arange(num_gt, device=device)
            if not torch.equal(torch.sort(target_native).values, expected_targets):
                raise RuntimeError("native target indices must cover every GT once")
            if torch.unique(source_native).numel() != num_gt:
                raise RuntimeError("native source queries are not one-to-one")

            native_by_target = torch.full((num_gt,), -1, dtype=torch.long, device=device)
            native_by_target[target_native] = source_native
            residual_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
            residual_mask[source_native] = False
            residual_queries = torch.nonzero(residual_mask, as_tuple=False).flatten()
            if residual_queries.numel() < num_gt:
                raise RuntimeError("insufficient native-unmatched queries for residual matching")

            detached_boxes = pred_boxes[batch_index].detach().float()
            gt_boxes = target["boxes"].detach().to(device=device, dtype=torch.float32)
            residual_boxes = detached_boxes[residual_queries]
            residual_cost = 5.0 * torch.cdist(residual_boxes, gt_boxes, p=1)
            residual_cost = residual_cost + 2.0 * (
                1.0
                - generalized_box_iou(
                    box_cxcywh_to_xyxy(residual_boxes),
                    box_cxcywh_to_xyxy(gt_boxes),
                )
            )
            if not torch.isfinite(residual_cost).all():
                raise RuntimeError("residual Hungarian cost contains NaN or Inf")
            row, column = linear_sum_assignment(residual_cost.cpu().numpy())
            row = torch.as_tensor(row, dtype=torch.long, device=device)
            column = torch.as_tensor(column, dtype=torch.long, device=device)
            if row.numel() != num_gt or not torch.equal(
                torch.sort(column).values, expected_targets
            ):
                raise RuntimeError("residual Hungarian did not cover every GT")

            residual_by_target = torch.full((num_gt,), -1, dtype=torch.long, device=device)
            residual_by_target[column] = residual_queries[row]
            if torch.isin(residual_by_target, native_by_target).any():
                raise RuntimeError("residual assignment reused a native query")

            native_iou = torch.diagonal(
                box_iou(
                    box_cxcywh_to_xyxy(detached_boxes[native_by_target]),
                    box_cxcywh_to_xyxy(gt_boxes),
                )[0]
            )
            residual_iou = torch.diagonal(
                box_iou(
                    box_cxcywh_to_xyxy(detached_boxes[residual_by_target]),
                    box_cxcywh_to_xyxy(gt_boxes),
                )[0]
            )
            gain = (residual_iou - native_iou).detach()
            active = gain > self.lcct_delta
            active_count += int(active.sum().item())
            if not active.any():
                continue
            if self.lcct_continuous_handover:
                weight = ((gain - self.lcct_delta) / self.lcct_transition_width).clamp(0.0, 1.0)
                weight = (weight * ramp_progress).detach()
            else:
                weight = active.to(dtype=gain.dtype)
            effective = active & (weight > 0.0)
            if not effective.any():
                continue
            selected_targets = torch.nonzero(effective, as_tuple=False).flatten()
            effective_count += int(selected_targets.numel())
            active_gains.append(gain[selected_targets].cpu())
            self._hilor_handover_pairs.append(
                {
                    "batch_index": batch_index,
                    "target_indices": selected_targets,
                    "native_queries": native_by_target[selected_targets],
                    "residual_queries": residual_by_target[selected_targets],
                    "native_iou": native_iou[selected_targets],
                    "residual_iou": residual_iou[selected_targets],
                    "weight": weight[selected_targets],
                }
            )

        self._hilor_last_diag["active_count"] = active_count
        self._hilor_last_diag["effective_count"] = effective_count
        if active_gains:
            self._hilor_last_diag["mean_absolute_iou_gain"] = float(
                torch.cat(active_gains).mean().item()
            )

    def _vfl_role_correction(self, outputs, targets, num_boxes):
        logits = outputs["pred_logits"]
        correction = logits.sum() * 0.0
        for pair in self._hilor_handover_pairs:
            batch_index = pair["batch_index"]
            target_indices = pair["target_indices"].to(targets[batch_index]["labels"].device)
            labels = targets[batch_index]["labels"][target_indices].to(logits.device)
            native_queries = pair["native_queries"].to(logits.device)
            residual_queries = pair["residual_queries"].to(logits.device)
            native_logits = logits[batch_index, native_queries, labels]
            residual_logits = logits[batch_index, residual_queries, labels]
            native_iou = pair["native_iou"].to(logits).clamp(0.0, 1.0)
            residual_iou = pair["residual_iou"].to(logits).clamp(0.0, 1.0)
            weight = pair["weight"].to(logits).detach()

            positive_native = native_iou * F.binary_cross_entropy_with_logits(
                native_logits, native_iou, reduction="none"
            )
            positive_residual = residual_iou * F.binary_cross_entropy_with_logits(
                residual_logits, residual_iou, reduction="none"
            )
            negative_native = self.alpha * torch.sigmoid(native_logits).detach().pow(
                self.gamma
            ) * F.binary_cross_entropy_with_logits(
                native_logits, torch.zeros_like(native_logits), reduction="none"
            )
            negative_residual = self.alpha * torch.sigmoid(residual_logits).detach().pow(
                self.gamma
            ) * F.binary_cross_entropy_with_logits(
                residual_logits, torch.zeros_like(residual_logits), reduction="none"
            )
            correction = correction + (
                weight
                * (
                    negative_native
                    - positive_native
                    + positive_residual
                    - negative_residual
                )
            ).sum()
        return correction / float(num_boxes)

    def forward(self, outputs, targets, **context):
        outputs_without_aux = {key: value for key, value in outputs.items() if "aux" not in key}
        native_indices = self.matcher(outputs_without_aux, targets)
        self._build_handover_pairs(
            outputs_without_aux, targets, native_indices, context.get("epoch")
        )

        num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            loss_dict = self.get_loss(
                loss_name, outputs, targets, native_indices, num_boxes
            )
            if loss_name == "vfl" and "loss_vfl" in loss_dict:
                loss_dict["loss_vfl"] = (
                    loss_dict["loss_vfl"]
                    + self._vfl_role_correction(outputs, targets, num_boxes)
                )
            loss_dict = {
                key: value * self.weight_dict[key]
                for key, value in loss_dict.items()
                if key in self.weight_dict
            }
            losses.update(loss_dict)

        if "aux_outputs" in outputs:
            for index, auxiliary_outputs in enumerate(outputs["aux_outputs"]):
                auxiliary_indices = self.matcher(auxiliary_outputs, targets)
                for loss_name in self.losses:
                    if loss_name == "masks":
                        continue
                    kwargs = {"log": False} if loss_name == "labels" else {}
                    loss_dict = self.get_loss(
                        loss_name,
                        auxiliary_outputs,
                        targets,
                        auxiliary_indices,
                        num_boxes,
                        **kwargs,
                    )
                    loss_dict = {
                        key: value * self.weight_dict[key]
                        for key, value in loss_dict.items()
                        if key in self.weight_dict
                    }
                    losses.update({f"{key}_aux_{index}": value for key, value in loss_dict.items()})

        if "dn_aux_outputs" in outputs:
            if "dn_meta" not in outputs:
                raise RuntimeError("dn_aux_outputs requires dn_meta")
            dn_indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            for index, auxiliary_outputs in enumerate(outputs["dn_aux_outputs"]):
                for loss_name in self.losses:
                    if loss_name == "masks":
                        continue
                    kwargs = {"log": False} if loss_name == "labels" else {}
                    loss_dict = self.get_loss(
                        loss_name,
                        auxiliary_outputs,
                        targets,
                        dn_indices,
                        dn_num_boxes,
                        **kwargs,
                    )
                    loss_dict = {
                        key: value * self.weight_dict[key]
                        for key, value in loss_dict.items()
                        if key in self.weight_dict
                    }
                    losses.update({f"{key}_dn_{index}": value for key, value in loss_dict.items()})
        if self.ufd_enabled:
            required = ("epoch", "step", "epoch_step")
            missing = [key for key in required if key not in context]
            if missing:
                raise RuntimeError(f"UFD requires training context: {missing}")
            losses.update(
                self._ufd_loss(
                    outputs,
                    targets,
                    epoch=context["epoch"],
                    step=context["step"],
                    epoch_step=context["epoch_step"],
                )
            )
        return losses
