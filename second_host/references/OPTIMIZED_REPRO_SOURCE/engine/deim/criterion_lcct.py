"""Method-specific criterion extension extracted verbatim from deim_criterion."""

import copy
import json
import math

import torch
import torch.distributed
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
    generalized_box_iou_aligned,
)
from .dfine_utils import bbox2distance, distance2bbox, weighting_function


class LCCTCriterionMixin:
    """Behavior-preserving methods mixed into DEIMCriterion."""

    @torch.no_grad()
    def _build_lcct_indices(
        self,
        outputs,
        targets,
        indices,
        epoch,
    ):
        """Build LCCT-v1 classification indices for FINAL MAL only.

        LCCT = Localization-Consistent Classification Credit Transfer.

        The native Hungarian assignment in ``indices`` is never modified.
        Residual candidates are drawn only from native-Hungarian-unmatched
        queries and are assigned one-to-one to GTs using a detached,
        localization-only Hungarian cost.

        When a residual query improves GT IoU by more than ``lcct_delta``,
        only the FINAL GT-class MAL positive identity is transferred from
        q_H to q_R. Regression/GO/auxiliary/encoder/DN/UFD supervision and
        inference remain unchanged.
        """
        if "pred_boxes" not in outputs:
            raise RuntimeError(
                "LCCT requires final outputs['pred_boxes']"
            )

        pred_boxes = outputs["pred_boxes"]

        if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
            raise RuntimeError(
                "LCCT expected pred_boxes with shape [B, N, 4], "
                f"got {tuple(pred_boxes.shape)}"
            )

        batch_size, num_queries, _ = pred_boxes.shape

        if len(targets) != batch_size:
            raise RuntimeError(
                "LCCT batch-size mismatch between outputs and targets"
            )

        if len(indices) != batch_size:
            raise RuntimeError(
                "LCCT batch-size mismatch between outputs and "
                "native Hungarian indices"
            )

        # Always return an independent indices object. The native
        # matcher output itself must never be overwritten.
        indices_cls = [
            (src.clone(), tgt.clone())
            for src, tgt in indices
        ]

        total_gt = sum(
            int(t["labels"].numel())
            for t in targets
        )

        # LCCT reassigns classification identities but must preserve
        # the number of positives produced by the native matcher.
        # For ordinary images this equals total_gt; when GT exceeds
        # the fixed query budget, the native matcher count is smaller.
        native_positive_count = sum(
            int(src.numel())
            for src, _ in indices_cls
        )

        # Iteration-local diagnostics. These are NOT losses.
        self._lcct_last_diag = {
            "lcct_active_count": 0,
            "lcct_active_ratio": 0.0,
            "lcct_mean_gain_active": 0.0,
            "lcct_max_gain": 0.0,
            "lcct_positive_count": native_positive_count,
            "lcct_duplicate_positive_count": 0,
            "lcct_eligible_gt_count": 0,
            "lcct_skipped_dense_image_count": 0,
            "lcct_skipped_dense_gt_count": 0,
            "lcct_gain_safe_count": 0,
            "lcct_gain_transition_count": 0,
            "lcct_gain_full_count": 0,
            "lcct_handover_ramp": 0.0,
            "lcct_effective_count": 0,
            "lcct_transition_count": 0,
            "lcct_full_transfer_count": 0,
            "lcct_handover_weight_mean": 0.0,
            "lcct_handover_weight_max": 0.0,
        }
        self._lcct_last_active_pairs = []
        self._lcct_last_handover_pairs = []

        # LCCT is training-only. Disabled/eval execution is exact
        # classification-identity pass-through.
        if not self.lcct_enabled or not self.training:
            return indices_cls

        if epoch is None:
            raise RuntimeError(
                "LCCT is enabled during training but epoch was not "
                "provided to DEIMCriterion.forward()"
            )

        epoch_value = int(epoch)

        # 0-based epochs:
        # epoch 0...warmup_epoch-1 are exactly native UFD-v1.
        if epoch_value < self.lcct_warmup_epoch:
            return indices_cls

        if self.lcct_continuous_handover:
            ramp_progress = min(
                max(
                    float(
                        epoch_value
                        - self.lcct_warmup_epoch
                    ) / float(self.lcct_ramp_epochs),
                    0.0,
                ),
                1.0,
            )
        else:
            ramp_progress = 1.0

        # With continuous handover, the first ramp epoch has exactly zero
        # transfer weight.  Native MAL, all other losses, assignments and
        # gradients are therefore unchanged; skip the detached residual
        # Hungarian and its diagnostics-only device synchronizations.
        if self.lcct_continuous_handover and ramp_progress <= 0.0:
            return indices_cls

        device = pred_boxes.device

        active_count = 0
        duplicate_positive_count = 0

        all_gains = []
        active_gains = []
        active_pairs = []
        handover_pairs = []
        handover_weights = []
        transition_count = 0
        full_transfer_count = 0
        eligible_gt_count = 0
        skipped_dense_image_count = 0
        skipped_dense_gt_count = 0
        gain_safe_count = 0
        gain_transition_count = 0
        gain_full_count = 0

        for b in range(batch_size):
            gt_boxes_raw = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]

            num_gt = int(gt_labels.numel())

            if int(gt_boxes_raw.shape[0]) != num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: GT boxes/labels count mismatch"
                )

            src_native, tgt_native = indices[b]

            if num_gt == 0:
                if (
                    src_native.numel() != 0
                    or tgt_native.numel() != 0
                ):
                    raise RuntimeError(
                        f"LCCT image {b}: native matcher returned "
                        "non-empty indices for empty GT"
                    )
                continue

            if num_queries < (2 * num_gt):
                # LCCT needs one native query q_H and one distinct
                # residual query q_R for every GT. Therefore its exact
                # admissibility condition is Q >= 2G. Dense augmented
                # images outside this domain retain their cloned native
                # classification assignment; other batch images remain
                # eligible for LCCT.
                skipped_dense_image_count += 1
                skipped_dense_gt_count += num_gt
                continue

            eligible_gt_count += num_gt

            if (
                src_native.numel() != num_gt
                or tgt_native.numel() != num_gt
            ):
                raise RuntimeError(
                    f"LCCT image {b}: native final Hungarian match "
                    "does not contain exactly one query per GT"
                )

            # Use prediction device for all LCCT geometry.
            src_native_dev = src_native.to(
                device=device,
                dtype=torch.long,
            )
            tgt_native_dev = tgt_native.to(
                device=device,
                dtype=torch.long,
            )

            if (
                (src_native_dev < 0).any()
                or (src_native_dev >= num_queries).any()
            ):
                raise RuntimeError(
                    f"LCCT image {b}: native source query out of range"
                )

            if (
                (tgt_native_dev < 0).any()
                or (tgt_native_dev >= num_gt).any()
            ):
                raise RuntimeError(
                    f"LCCT image {b}: native target index out of range"
                )

            if torch.unique(src_native_dev).numel() != num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: native Hungarian source "
                    "queries are not one-to-one"
                )

            expected_tgt = torch.arange(
                num_gt,
                dtype=torch.long,
                device=device,
            )

            if not torch.equal(
                torch.sort(tgt_native_dev).values,
                expected_tgt,
            ):
                raise RuntimeError(
                    f"LCCT image {b}: native Hungarian target "
                    "indices do not cover every GT exactly once"
                )

            # Map native Hungarian query q_H by GT index.
            q_h_by_tgt = torch.full(
                (num_gt,),
                -1,
                dtype=torch.long,
                device=device,
            )
            q_h_by_tgt[tgt_native_dev] = src_native_dev

            if (q_h_by_tgt < 0).any():
                raise RuntimeError(
                    f"LCCT image {b}: failed to build q_H mapping"
                )

            # --------------------------------------------------
            # Residual candidate pool U = Q \ H.
            # --------------------------------------------------
            residual_mask = torch.ones(
                num_queries,
                dtype=torch.bool,
                device=device,
            )
            residual_mask[src_native_dev] = False

            residual_queries = torch.nonzero(
                residual_mask,
                as_tuple=False,
            ).flatten()

            if residual_queries.numel() < num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: only "
                    f"{residual_queries.numel()} residual queries "
                    f"for {num_gt} GTs"
                )

            pred_boxes_b = pred_boxes[b].detach().float()

            gt_boxes = gt_boxes_raw.detach().to(
                device=device,
                dtype=torch.float32,
            )

            residual_boxes = pred_boxes_b[
                residual_queries
            ]

            # --------------------------------------------------
            # Detached localization-only residual assignment:
            #
            # C_R = 5 * L1(cxcywh)
            #     + 2 * (1 - GIoU(xyxy))
            #
            # No classification cost or semantic gate.
            # --------------------------------------------------
            cost_l1 = torch.cdist(
                residual_boxes,
                gt_boxes,
                p=1,
            )

            cost_giou = 1.0 - generalized_box_iou(
                box_cxcywh_to_xyxy(residual_boxes),
                box_cxcywh_to_xyxy(gt_boxes),
            )

            cost_residual = (
                5.0 * cost_l1
                + 2.0 * cost_giou
            )

            if not torch.isfinite(cost_residual).all():
                raise RuntimeError(
                    f"LCCT image {b}: residual Hungarian cost "
                    "contains NaN or Inf"
                )

            row_ind, col_ind = linear_sum_assignment(
                cost_residual.detach().cpu().numpy()
            )

            row_ind = torch.as_tensor(
                row_ind,
                dtype=torch.long,
                device=device,
            )
            col_ind = torch.as_tensor(
                col_ind,
                dtype=torch.long,
                device=device,
            )

            if row_ind.numel() != num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: residual Hungarian produced "
                    f"{row_ind.numel()} assignments for {num_gt} GTs"
                )

            if not torch.equal(
                torch.sort(col_ind).values,
                expected_tgt,
            ):
                raise RuntimeError(
                    f"LCCT image {b}: residual Hungarian did not "
                    "cover every GT exactly once"
                )

            q_r_by_tgt = torch.full(
                (num_gt,),
                -1,
                dtype=torch.long,
                device=device,
            )

            q_r_by_tgt[col_ind] = residual_queries[
                row_ind
            ]

            if (q_r_by_tgt < 0).any():
                raise RuntimeError(
                    f"LCCT image {b}: failed to build q_R mapping"
                )

            if torch.unique(q_r_by_tgt).numel() != num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: residual queries are not "
                    "one-to-one"
                )

            if torch.isin(
                q_r_by_tgt,
                q_h_by_tgt,
            ).any():
                raise RuntimeError(
                    f"LCCT image {b}: residual q_R overlaps native "
                    "Hungarian query set"
                )

            # --------------------------------------------------
            # Localization gain:
            # Delta_j = IoU(q_R, GT) - IoU(q_H, GT)
            #
            # q_H/q_R arrays are both ordered by target index,
            # therefore the diagonal gives the paired IoUs.
            # --------------------------------------------------
            q_h_boxes = pred_boxes_b[
                q_h_by_tgt
            ]
            q_r_boxes = pred_boxes_b[
                q_r_by_tgt
            ]

            iou_h_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(q_h_boxes),
                box_cxcywh_to_xyxy(gt_boxes),
            )
            iou_r_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(q_r_boxes),
                box_cxcywh_to_xyxy(gt_boxes),
            )

            q_h_iou = torch.diagonal(
                iou_h_matrix,
                offset=0,
            )
            q_r_iou = torch.diagonal(
                iou_r_matrix,
                offset=0,
            )

            gain = (q_r_iou - q_h_iou).detach()

            if not torch.isfinite(gain).all():
                raise RuntimeError(
                    f"LCCT image {b}: localization gain contains "
                    "NaN or Inf"
                )

            active_by_tgt = gain > self.lcct_delta

            if self.lcct_continuous_handover:
                handover_weight = (
                    (gain - self.lcct_delta)
                    / self.lcct_transition_width
                ).clamp(min=0.0, max=1.0)
                handover_weight = (
                    handover_weight * ramp_progress
                ).detach()
            else:
                handover_weight = active_by_tgt.to(
                    dtype=gain.dtype
                )

            # Consolidate diagnostics into one device-to-host transfer per
            # image. Query ids are below 2**24, so float32 represents them
            # exactly before the CPU-side integer cast.
            diagnostic_cpu = torch.stack(
                (
                    q_h_by_tgt.to(dtype=gain.dtype),
                    q_r_by_tgt.to(dtype=gain.dtype),
                    q_h_iou,
                    q_r_iou,
                    gain,
                    handover_weight,
                ),
                dim=1,
            ).detach().cpu()
            q_h_by_tgt_cpu = diagnostic_cpu[:, 0].to(torch.long)
            q_r_by_tgt_cpu = diagnostic_cpu[:, 1].to(torch.long)
            q_h_iou_cpu = diagnostic_cpu[:, 2]
            q_r_iou_cpu = diagnostic_cpu[:, 3]
            gain_cpu = diagnostic_cpu[:, 4]
            handover_weight_cpu = diagnostic_cpu[:, 5]
            active_by_tgt_cpu = gain_cpu > self.lcct_delta

            gain_safe_count += int(
                (gain_cpu <= self.lcct_delta).sum().item()
            )
            gain_transition_count += int((
                (gain_cpu > self.lcct_delta)
                & (
                    gain_cpu
                    < self.lcct_delta
                    + self.lcct_transition_width
                )
            ).sum().item())
            gain_full_count += int((
                gain_cpu
                >= self.lcct_delta
                + self.lcct_transition_width
            ).sum().item())

            all_gains.append(gain_cpu)

            if bool(active_by_tgt_cpu.any()):
                active_gain_b_cpu = gain_cpu[
                    active_by_tgt_cpu
                ]
                active_gains.append(active_gain_b_cpu)
                active_count += int(
                    active_by_tgt_cpu.sum().item()
                )

            # --------------------------------------------------
            # Exclusive final classification identity transfer.
            #
            # Preserve native target-index tensor and its order.
            # Only source query identity changes for active GTs.
            # --------------------------------------------------
            src_cls, tgt_cls = indices_cls[b]

            if not torch.equal(
                tgt_cls,
                tgt_native,
            ):
                raise RuntimeError(
                    f"LCCT image {b}: cloned target indices changed"
                )

            active_native = active_by_tgt[
                tgt_native_dev
            ]

            if (
                active_native.any()
                and not self.lcct_continuous_handover
            ):
                replacement_queries = q_r_by_tgt[
                    tgt_native_dev[
                        active_native
                    ]
                ]

                active_native_on_src_device = (
                    active_native.to(
                        device=src_cls.device
                    )
                )

                src_cls[
                    active_native_on_src_device
                ] = replacement_queries.to(
                    device=src_cls.device,
                    dtype=src_cls.dtype,
                )

            if src_cls.numel() != num_gt:
                raise RuntimeError(
                    f"LCCT image {b}: final classification "
                    "positive count changed"
                )

            duplicate_count_b = (
                int(src_cls.numel())
                - int(torch.unique(src_cls).numel())
            )

            duplicate_positive_count += (
                duplicate_count_b
            )

            if duplicate_count_b != 0:
                raise RuntimeError(
                    f"LCCT image {b}: duplicate final MAL "
                    "positive query identity detected"
                )

            # Smoke/debug provenance for active transfers only.
            active_targets = torch.nonzero(
                active_by_tgt,
                as_tuple=False,
            ).flatten()
            active_targets_cpu = torch.nonzero(
                active_by_tgt_cpu,
                as_tuple=False,
            ).flatten()

            if (
                self.lcct_continuous_handover
                and active_targets.numel() > 0
            ):
                active_weight = handover_weight[
                    active_targets
                ]
                handover_pairs.append({
                    "image_index": int(b),
                    "target_indices": (
                        active_targets.detach()
                    ),
                    "q_h": q_h_by_tgt[
                        active_targets
                    ].detach(),
                    "q_r": q_r_by_tgt[
                        active_targets
                    ].detach(),
                    "iou_h": q_h_iou[
                        active_targets
                    ].detach(),
                    "iou_r": q_r_iou[
                        active_targets
                    ].detach(),
                    "weight": active_weight.detach(),
                })
                active_weight_cpu = handover_weight_cpu[
                    active_targets_cpu
                ]
                handover_weights.append(active_weight_cpu)
                transition_count += int((
                    (active_weight_cpu > 0.0)
                    & (active_weight_cpu < 1.0)
                ).sum().item())
                full_transfer_count += int((
                    active_weight_cpu >= 1.0
                ).sum().item())

            for target_idx in active_targets_cpu.tolist():
                active_pairs.append({
                    "image_index": int(b),
                    "target_index": int(target_idx),
                    "q_h": int(
                        q_h_by_tgt_cpu[target_idx].item()
                    ),
                    "q_r": int(
                        q_r_by_tgt_cpu[target_idx].item()
                    ),
                    "q_h_iou": float(
                        q_h_iou_cpu[target_idx].item()
                    ),
                    "q_r_iou": float(
                        q_r_iou_cpu[target_idx].item()
                    ),
                    "gain": float(
                        gain_cpu[target_idx].item()
                    ),
                })

        positive_count = sum(
            int(src.numel())
            for src, _ in indices_cls
        )

        if positive_count != native_positive_count:
            raise RuntimeError(
                "LCCT final MAL positive identity count changed "
                "relative to the native matcher: "
                f"{positive_count} vs native count "
                f"{native_positive_count}"
            )

        if duplicate_positive_count != 0:
            raise RuntimeError(
                "LCCT duplicate final MAL positive identity detected"
            )

        if all_gains:
            all_gain_tensor = torch.cat(
                all_gains,
                dim=0,
            )
            max_gain = float(
                all_gain_tensor.max().item()
            )
        else:
            max_gain = 0.0

        if active_gains:
            active_gain_tensor = torch.cat(
                active_gains,
                dim=0,
            )
            mean_gain_active = float(
                active_gain_tensor.mean().item()
            )
        else:
            mean_gain_active = 0.0

        active_ratio = (
            float(active_count) / float(total_gt)
            if total_gt > 0
            else 0.0
        )

        self._lcct_last_diag = {
            "lcct_active_count": int(active_count),
            "lcct_active_ratio": float(active_ratio),
            "lcct_mean_gain_active": float(
                mean_gain_active
            ),
            "lcct_max_gain": float(max_gain),
            "lcct_positive_count": int(
                positive_count
            ),
            "lcct_duplicate_positive_count": int(
                duplicate_positive_count
            ),
            "lcct_eligible_gt_count": int(eligible_gt_count),
            "lcct_skipped_dense_image_count": int(
                skipped_dense_image_count
            ),
            "lcct_skipped_dense_gt_count": int(
                skipped_dense_gt_count
            ),
            "lcct_gain_safe_count": int(gain_safe_count),
            "lcct_gain_transition_count": int(
                gain_transition_count
            ),
            "lcct_gain_full_count": int(gain_full_count),
        }

        self._lcct_last_active_pairs = active_pairs
        self._lcct_last_handover_pairs = handover_pairs

        if handover_weights:
            weight_tensor = torch.cat(
                handover_weights,
                dim=0,
            )
            effective_count = int((
                weight_tensor > 0.0
            ).sum().item())
            mean_weight = float(
                weight_tensor.mean().item()
            )
            max_weight = float(
                weight_tensor.max().item()
            )
        else:
            effective_count = 0
            mean_weight = 0.0
            max_weight = 0.0

        self._lcct_last_diag.update({
            "lcct_handover_ramp": float(ramp_progress),
            "lcct_effective_count": int(effective_count),
            "lcct_transition_count": int(transition_count),
            "lcct_full_transfer_count": int(full_transfer_count),
            "lcct_handover_weight_mean": float(mean_weight),
            "lcct_handover_weight_max": float(max_weight),
        })

        return indices_cls

    def _lcct_mal_handover_correction(
        self,
        outputs,
        targets,
        num_boxes,
    ):
        """Exchange complete FINAL-MAL roles for active q_H/q_R pairs.

        For each pair this adds

            w * [(N_H - P_H) + (P_R - N_R)]

        to native MAL.  Therefore w=0 is native MAL and w=1 is the
        released hard transfer for the same candidate pair.  The gate,
        IoUs and assignments are detached; gradients flow only to logits.
        """
        logits = outputs["pred_logits"]
        correction = logits.sum() * 0.0

        if not self._lcct_last_handover_pairs:
            return correction

        negative_alpha = (
            1.0
            if self.mal_alpha is None
            else float(self.mal_alpha)
        )

        for pair in self._lcct_last_handover_pairs:
            image_index = pair["image_index"]
            target_indices = pair["target_indices"]
            labels = targets[image_index]["labels"][
                target_indices.to(
                    device=targets[image_index]["labels"].device
                )
            ].to(device=logits.device, dtype=torch.long)

            q_h = pair["q_h"].to(
                device=logits.device,
                dtype=torch.long,
            )
            q_r = pair["q_r"].to(
                device=logits.device,
                dtype=torch.long,
            )
            z_h = logits[image_index, q_h, labels]
            z_r = logits[image_index, q_r, labels]

            iou_h = pair["iou_h"].to(
                device=logits.device,
                dtype=logits.dtype,
            ).clamp(min=0.0, max=1.0)
            iou_r = pair["iou_r"].to(
                device=logits.device,
                dtype=logits.dtype,
            ).clamp(min=0.0, max=1.0)
            weight = pair["weight"].to(
                device=logits.device,
                dtype=logits.dtype,
            ).detach()

            positive_h = F.binary_cross_entropy_with_logits(
                z_h,
                iou_h.pow(self.gamma),
                reduction="none",
            )
            positive_r = F.binary_cross_entropy_with_logits(
                z_r,
                iou_r.pow(self.gamma),
                reduction="none",
            )

            negative_h_weight = (
                negative_alpha
                * torch.sigmoid(z_h).detach().pow(self.gamma)
            )
            negative_r_weight = (
                negative_alpha
                * torch.sigmoid(z_r).detach().pow(self.gamma)
            )
            negative_h = negative_h_weight * (
                F.binary_cross_entropy_with_logits(
                    z_h,
                    torch.zeros_like(z_h),
                    reduction="none",
                )
            )
            negative_r = negative_r_weight * (
                F.binary_cross_entropy_with_logits(
                    z_r,
                    torch.zeros_like(z_r),
                    reduction="none",
                )
            )

            correction = correction + (
                weight
                * (
                    negative_h
                    - positive_h
                    + positive_r
                    - negative_r
                )
            ).sum()

        return correction / float(num_boxes)
