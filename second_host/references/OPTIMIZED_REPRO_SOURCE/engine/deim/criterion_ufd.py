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
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized


class UFDCriterionMixin:
    """Behavior-preserving methods mixed into DEIMCriterion."""

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def _ufd_empirical_cdf(self, values, sorted_samples):
        if sorted_samples.numel() == 0:
            raise RuntimeError(
                "UFD CDF requested but buffer is empty"
            )

        idx = torch.searchsorted(
            sorted_samples,
            values,
            right=True,
        )

        return (
            idx.to(values.dtype)
            / float(sorted_samples.numel())
        )

    def _ufd_max_ioa(self, boxes_xyxy):
        n = int(boxes_xyxy.shape[0])

        if n <= 1:
            return torch.zeros(
                n,
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )

        lt = torch.maximum(
            boxes_xyxy[:, None, :2],
            boxes_xyxy[None, :, :2],
        )

        rb = torch.minimum(
            boxes_xyxy[:, None, 2:],
            boxes_xyxy[None, :, 2:],
        )

        wh = (rb - lt).clamp(min=0)

        inter = (
            wh[..., 0]
            * wh[..., 1]
        )

        area = (
            (
                boxes_xyxy[:, 2]
                - boxes_xyxy[:, 0]
            ).clamp_min(0)
            *
            (
                boxes_xyxy[:, 3]
                - boxes_xyxy[:, 1]
            ).clamp_min(0)
        )

        ioa = (
            inter
            / area[:, None].clamp_min(1e-12)
        )

        eye = torch.eye(
            n,
            device=boxes_xyxy.device,
            dtype=torch.bool,
        )

        ioa = ioa.masked_fill(
            eye,
            0.0,
        )

        return ioa.max(dim=1).values

    @torch.no_grad()
    def _ufd_difficulty(self, target_boxes):
        """
        Frozen difficulty:

            h = [1 - F_A(a) + F_R(r) + F_O(o)] / 3

        Current augmented normalized boxes are mapped into
        the fixed official-train CDFs.
        """
        if target_boxes.numel() == 0:
            return target_boxes.new_zeros(
                (0,)
            )

        boxes = target_boxes.float()

        w = boxes[:, 2].clamp_min(
            1e-12
        )

        h = boxes[:, 3].clamp_min(
            1e-12
        )

        area_640 = (
            w
            * h
            * (
                self.ufd_base_size
                ** 2
            )
        )

        aspect = torch.abs(
            torch.log(
                w / h
            )
        )

        boxes_xyxy = (
            box_cxcywh_to_xyxy(
                boxes
            )
        )

        max_ioa = self._ufd_max_ioa(
            boxes_xyxy
        )

        f_area = self._ufd_empirical_cdf(
            area_640,
            self._ufd_area_cdf,
        )

        f_aspect = self._ufd_empirical_cdf(
            aspect,
            self._ufd_aspect_cdf,
        )

        f_ioa = self._ufd_empirical_cdf(
            max_ioa,
            self._ufd_ioa_cdf,
        )

        difficulty = (
            (
                1.0 - f_area
            )
            + f_aspect
            + f_ioa
        ) / 3.0

        return difficulty.clamp(
            min=0.0,
            max=1.0,
        ).detach()

    @torch.no_grad()
    def _ufd_side_uncertainty(
        self,
        outputs,
        targets,
        indices,
    ):
        """
        Reuse FINAL L3 O2O Hungarian indices.

        No second matcher and no GO-union identity.
        """
        all_uncertainty = []

        m = self.reg_max + 1
        norm = math.log(float(m))

        corners_all = outputs[
            "pred_corners"
        ]

        for b, target in enumerate(
            targets
        ):
            n_gt = len(
                target["boxes"]
            )

            u = torch.zeros(
                (
                    n_gt,
                    4,
                ),
                dtype=torch.float32,
                device=corners_all.device,
            )

            src_idx, tgt_idx = (
                indices[b]
            )

            if len(src_idx) > 0:
                src_idx = src_idx.to(
                    corners_all.device
                )

                tgt_idx = tgt_idx.to(
                    corners_all.device
                )

                corners = (
                    corners_all[
                        b,
                        src_idx,
                    ]
                    .float()
                    .reshape(
                        -1,
                        4,
                        m,
                    )
                )

                prob = F.softmax(
                    corners,
                    dim=-1,
                )

                entropy = -(
                    prob
                    * torch.log(
                        prob.clamp_min(
                            1e-12
                        )
                    )
                ).sum(
                    dim=-1
                ) / norm

                u[
                    tgt_idx
                ] = entropy.detach()

            all_uncertainty.append(
                u
            )

        return all_uncertainty

    def _ufd_schedule(
        self,
        epoch,
        step,
        epoch_step,
    ):
        epoch = float(epoch)
        step = float(step)
        epoch_step = max(
            float(epoch_step),
            1.0,
        )

        progress = (
            epoch
            + step / epoch_step
        ) / float(
            self.ufd_total_epochs
        )

        progress = max(
            0.0,
            min(
                progress,
                1.0,
            ),
        )

        if (
            self.ufd_lambda_ramp_fraction
            > 0
        ):
            lambda_scale = min(
                progress
                / self.ufd_lambda_ramp_fraction,
                1.0,
            )
        else:
            lambda_scale = 1.0

        lambda_ufd = (
            self.ufd_lambda_max
            * lambda_scale
        )

        delay = (
            self.ufd_uncertainty_delay_fraction
        )

        ramp = (
            self.ufd_uncertainty_ramp_fraction
        )

        if progress <= delay:
            alpha_u = 0.0

        elif ramp <= 0:
            alpha_u = (
                self.ufd_alpha_u_max
            )

        elif progress < (
            delay + ramp
        ):
            alpha_u = (
                self.ufd_alpha_u_max
                * (
                    progress - delay
                )
                / ramp
            )

        else:
            alpha_u = (
                self.ufd_alpha_u_max
            )

        return (
            lambda_ufd,
            alpha_u,
            progress,
        )

    @staticmethod
    def _ufd_raw_distance(
        ref_points,
        gt_xyxy,
        reg_scale,
    ):
        """
        Exact continuous quantity immediately before
        translate_gt() in current bbox2distance().
        """
        rs = torch.abs(
            reg_scale
        ).to(
            dtype=ref_points.dtype,
            device=ref_points.device,
        )

        left = (
            (
                ref_points[:, 0]
                - gt_xyxy[:, 0]
            )
            / (
                ref_points[:, 2]
                / rs
                + 1e-16
            )
            - 0.5 * rs
        )

        top = (
            (
                ref_points[:, 1]
                - gt_xyxy[:, 1]
            )
            / (
                ref_points[:, 3]
                / rs
                + 1e-16
            )
            - 0.5 * rs
        )

        right = (
            (
                gt_xyxy[:, 2]
                - ref_points[:, 0]
            )
            / (
                ref_points[:, 2]
                / rs
                + 1e-16
            )
            - 0.5 * rs
        )

        bottom = (
            (
                gt_xyxy[:, 3]
                - ref_points[:, 1]
            )
            / (
                ref_points[:, 3]
                / rs
                + 1e-16
            )
            - 0.5 * rs
        )

        return torch.stack(
            (
                left,
                top,
                right,
                bottom,
            ),
            dim=-1,
        )

    @staticmethod
    def _ufd_centerness(
        centers,
        gt_xyxy,
    ):
        x1, y1, x2, y2 = (
            gt_xyxy
        )

        dl = (
            centers[:, 0]
            - x1
        ).clamp_min(0)

        dt = (
            centers[:, 1]
            - y1
        ).clamp_min(0)

        dr = (
            x2
            - centers[:, 0]
        ).clamp_min(0)

        db = (
            y2
            - centers[:, 1]
        ).clamp_min(0)

        lr = (
            torch.minimum(
                dl,
                dr,
            )
            / torch.maximum(
                dl,
                dr,
            ).clamp_min(
                1e-12
            )
        )

        tb = (
            torch.minimum(
                dt,
                db,
            )
            / torch.maximum(
                dt,
                db,
            ).clamp_min(
                1e-12
            )
        )

        return torch.sqrt(
            (
                lr
                * tb
            ).clamp_min(0)
        )

    @torch.no_grad()
    def _ufd_candidate_queues(
        self,
        refs,
        valid_mask,
        spatial_shapes,
        gt_xyxy,
        difficulty,
        reg_scale,
        weight_min,
        weight_max,
    ):
        """
        Build exact F8 -> F16 -> F32 candidate queues for all GTs in one image.

        Geometry is evaluated for every GT at once on each native feature level.
        The original first-reachable-level rule, FP32 centerness values, stable
        descending order and CPU conflict-resolution inputs are preserved.
        """
        shapes = (
            spatial_shapes
            .detach()
            .cpu()
            .tolist()
        )

        if len(shapes) != 3:
            raise RuntimeError(
                "UFD-v1 expects exactly "
                "F8/F16/F32"
            )

        starts = []
        start = 0

        for h, w in shapes:
            n = int(h) * int(w)
            starts.append((start, start + n))
            start += n

        if start != refs.shape[0]:
            raise RuntimeError(
                "UFD spatial-shape/token "
                "count mismatch"
            )

        n_gt = len(gt_xyxy)
        queues = [[] for _ in range(n_gt)]
        unresolved = set(range(n_gt))
        difficulty_cpu = difficulty.detach().cpu()
        rs = torch.abs(reg_scale).to(
            dtype=refs.dtype,
            device=refs.device,
        )

        for begin, end in starts:
            if not unresolved:
                break

            ref_lvl = refs[begin:end]
            valid_lvl = valid_mask[begin:end]
            centers = ref_lvl[:, :2]

            center_x = centers[:, 0].unsqueeze(0)
            center_y = centers[:, 1].unsqueeze(0)
            gt_x1 = gt_xyxy[:, 0].unsqueeze(1)
            gt_y1 = gt_xyxy[:, 1].unsqueeze(1)
            gt_x2 = gt_xyxy[:, 2].unsqueeze(1)
            gt_y2 = gt_xyxy[:, 3].unsqueeze(1)

            inside = (
                valid_lvl.unsqueeze(0)
                & (center_x >= gt_x1)
                & (center_x <= gt_x2)
                & (center_y >= gt_y1)
                & (center_y <= gt_y2)
            )

            ref_x = ref_lvl[:, 0].unsqueeze(0)
            ref_y = ref_lvl[:, 1].unsqueeze(0)
            ref_w = ref_lvl[:, 2].unsqueeze(0)
            ref_h = ref_lvl[:, 3].unsqueeze(0)
            scale_w = ref_w / rs + 1e-16
            scale_h = ref_h / rs + 1e-16

            left = (ref_x - gt_x1) / scale_w - 0.5 * rs
            top = (ref_y - gt_y1) / scale_h - 0.5 * rs
            right = (gt_x2 - ref_x) / scale_w - 0.5 * rs
            bottom = (gt_y2 - ref_y) / scale_h - 0.5 * rs
            reachable = (
                (left >= weight_min)
                & (left <= weight_max)
                & (top >= weight_min)
                & (top <= weight_max)
                & (right >= weight_min)
                & (right <= weight_max)
                & (bottom >= weight_min)
                & (bottom <= weight_max)
            )
            candidate = inside & reachable

            dl = (center_x - gt_x1).clamp_min(0)
            dt = (center_y - gt_y1).clamp_min(0)
            dr = (gt_x2 - center_x).clamp_min(0)
            db = (gt_y2 - center_y).clamp_min(0)
            lr = (
                torch.minimum(dl, dr)
                / torch.maximum(dl, dr).clamp_min(1e-12)
            )
            tb = (
                torch.minimum(dt, db)
                / torch.maximum(dt, db).clamp_min(1e-12)
            )
            centerness = torch.sqrt((lr * tb).clamp_min(0))

            # One mask transfer and one value transfer per level replace
            # per-GT/per-level synchronization. Sorting remains stable on CPU,
            # with original token order as the exact tie-break.
            candidate_cpu = candidate.detach().cpu()
            centerness_cpu = centerness.detach().cpu()
            resolved_here = []

            for j in sorted(unresolved):
                idx = torch.nonzero(
                    candidate_cpu[j],
                    as_tuple=False,
                ).flatten()

                if idx.numel() == 0:
                    continue

                c_values_t = centerness_cpu[j, idx]
                order = torch.argsort(
                    c_values_t,
                    descending=True,
                    stable=True,
                )
                idx = idx[order]
                c_values_t = c_values_t[order]
                h_value = float(difficulty_cpu[j].item())

                queue = []
                for token_offset, c_tensor in zip(idx.tolist(), c_values_t):
                    c_value = float(c_tensor.item())
                    queue.append(
                        (
                            int(token_offset + begin),
                            c_value,
                            float(
                                c_value
                                * (
                                    1.0
                                    + self.ufd_alpha_h
                                    * h_value
                                )
                            ),
                        )
                    )

                queues[j] = queue
                resolved_here.append(j)

            unresolved.difference_update(resolved_here)

        return queues

    @torch.no_grad()
    def _ufd_resolve_conflicts(
        self,
        queues,
    ):
        """
        Deterministic round-based greedy allocation.

        Each GT receives at most one token per round.
        Conflict winner:
            centerness * (1 + alpha_h * difficulty)

        Losers continue down their ranked reachable queue
        until quota or exhaustion.

        No outside-token fallback.
        """
        n_gt = len(queues)

        assigned = [
            []
            for _ in range(n_gt)
        ]

        cursor = [
            0
            for _ in range(n_gt)
        ]

        claimed = set()

        for _round in range(
            self.ufd_k
        ):
            pending = {
                j
                for j in range(
                    n_gt
                )
                if (
                    len(
                        assigned[j]
                    )
                    < self.ufd_k
                    and len(
                        assigned[j]
                    )
                    < len(
                        queues[j]
                    )
                )
            }

            while pending:
                proposals = {}

                exhausted = []

                for j in sorted(
                    pending
                ):
                    q = queues[j]

                    while (
                        cursor[j]
                        < len(q)
                        and q[
                            cursor[j]
                        ][0] in claimed
                    ):
                        cursor[j] += 1

                    if cursor[j] >= len(q):
                        exhausted.append(
                            j
                        )
                        continue

                    token, c, priority = (
                        q[
                            cursor[j]
                        ]
                    )

                    proposals.setdefault(
                        token,
                        [],
                    ).append(
                        (
                            j,
                            c,
                            priority,
                        )
                    )

                for j in exhausted:
                    pending.discard(
                        j
                    )

                if not proposals:
                    break

                for token in sorted(
                    proposals
                ):
                    candidates = (
                        proposals[token]
                    )

                    # Higher priority wins.
                    # Smaller GT index is deterministic
                    # tie-break.
                    winner = max(
                        candidates,
                        key=lambda x: (
                            x[2],
                            -x[0],
                        ),
                    )

                    winner_j = winner[0]
                    winner_c = winner[1]

                    if token in claimed:
                        raise RuntimeError(
                            "UFD token assigned twice"
                        )

                    assigned[
                        winner_j
                    ].append(
                        (
                            token,
                            winner_c,
                        )
                    )

                    claimed.add(
                        token
                    )

                    cursor[
                        winner_j
                    ] += 1

                    pending.discard(
                        winner_j
                    )

                    # Losing GTs backfill using
                    # their next ranked candidate.
                    for loser_j, _, _ in candidates:
                        if loser_j == winner_j:
                            continue

                        cursor[
                            loser_j
                        ] += 1

        return assigned

    def _ufd_average_denominator(
        self,
        denominator,
    ):
        d = (
            denominator
            .detach()
            .clone()
        )

        if (
            is_dist_available_and_initialized()
        ):
            torch.distributed.all_reduce(
                d
            )

            d = (
                d
                / get_world_size()
            )

        return d.clamp_min(
            1e-12
        )

    def _ufd_loss(
        self,
        outputs,
        targets,
        indices,
        **kwargs,
    ):
        """
        UFD-v1 total:

          lambda_UFD(t) *
          [
              0.15 * FGL
            + 5.00 * L1
            + 2.00 * GIoU
          ]

        These coefficients are applied HERE exactly once.
        They must NOT be added to global weight_dict.
        """
        epoch = kwargs.get(
            "epoch",
            0,
        )

        step = kwargs.get(
            "step",
            0,
        )

        epoch_step = kwargs.get(
            "epoch_step",
            1,
        )

        (
            lambda_ufd,
            alpha_u,
            _,
        ) = self._ufd_schedule(
            epoch,
            step,
            epoch_step,
        )

        zero = (
            outputs[
                "ufd_logits"
            ].sum()
            * 0.0
        )

        # At the exact first point of the ramp all ranks
        # have the same zero lambda, so safely skip work.
        if lambda_ufd <= 0:
            return {
                "loss_ufd_fgl":
                    zero,

                "loss_ufd_bbox":
                    zero,

                "loss_ufd_giou":
                    zero,
            }

        ufd_logits = outputs[
            "ufd_logits"
        ]

        refs = outputs[
            "ufd_ref_points"
        ].float()

        valid_mask = outputs[
            "ufd_valid_mask"
        ].bool()

        spatial_shapes = outputs[
            "ufd_spatial_shapes"
        ]

        reg_scale = outputs[
            "reg_scale"
        ].float()

        up = outputs[
            "up"
        ].float()

        if (
            ufd_logits.shape[1]
            != refs.shape[0]
        ):
            raise RuntimeError(
                "UFD logits/reference "
                "token mismatch"
            )

        project = weighting_function(
            self.reg_max,
            up,
            reg_scale,
        ).float()

        weight_min = float(
            project.min()
            .detach()
            .cpu()
            .item()
        )

        weight_max = float(
            project.max()
            .detach()
            .cpu()
            .item()
        )

        # Before the uncertainty schedule activates, alpha_u is exactly zero.
        # Entropy cannot affect any UFD numerator or denominator in that phase,
        # so avoid the softmax/log work and the associated index transfers.
        if alpha_u > 0.0:
            uncertainty = (
                self._ufd_side_uncertainty(
                    outputs,
                    targets,
                    indices,
                )
            )
        else:
            uncertainty = [
                ufd_logits.new_zeros(
                    (len(target["boxes"]), 4),
                    dtype=torch.float32,
                )
                for target in targets
            ]

        batch_ids = []
        token_ids = []
        gt_boxes_selected = []
        spatial_mass = []
        difficulty_selected = []
        uncertainty_selected = []

        for b, target in enumerate(
            targets
        ):
            target_boxes = (
                target["boxes"]
                .float()
            )

            if target_boxes.numel() == 0:
                continue

            gt_xyxy = (
                box_cxcywh_to_xyxy(
                    target_boxes
                )
            )

            difficulty = (
                self._ufd_difficulty(
                    target_boxes
                )
            )

            queues = (
                self._ufd_candidate_queues(
                    refs,
                    valid_mask,
                    spatial_shapes,
                    gt_xyxy,
                    difficulty,
                    reg_scale,
                    weight_min,
                    weight_max,
                )
            )

            assignments = (
                self._ufd_resolve_conflicts(
                    queues
                )
            )

            for j, selected in enumerate(
                assignments
            ):
                if not selected:
                    # Reachability/conflict exhaustion:
                    # no outside token and no partial-side
                    # supervision in UFD-v1.
                    continue

                c = torch.tensor(
                    [
                        x[1]
                        for x in selected
                    ],
                    dtype=torch.float32,
                    device=ufd_logits.device,
                )

                c_sum = c.sum()

                if float(
                    c_sum.detach()
                    .cpu()
                    .item()
                ) > 1e-12:
                    c_tilde = (
                        c
                        / c_sum
                    )

                else:
                    c_tilde = (
                        torch.ones_like(c)
                        / len(c)
                    )

                if not torch.isfinite(
                    c_tilde
                ).all():
                    raise RuntimeError(
                        "Non-finite UFD "
                        "spatial mass"
                    )

                for k, (
                    token,
                    _,
                ) in enumerate(
                    selected
                ):
                    batch_ids.append(
                        b
                    )

                    token_ids.append(
                        int(token)
                    )

                    gt_boxes_selected.append(
                        target_boxes[j]
                    )

                    spatial_mass.append(
                        c_tilde[k]
                    )

                    difficulty_selected.append(
                        difficulty[j]
                    )

                    uncertainty_selected.append(
                        uncertainty[b][j]
                    )

        if batch_ids:
            batch_ids_t = torch.tensor(
                batch_ids,
                dtype=torch.long,
                device=ufd_logits.device,
            )

            token_ids_t = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=ufd_logits.device,
            )

            pred_logits = (
                ufd_logits[
                    batch_ids_t,
                    token_ids_t,
                ]
                .float()
                .reshape(
                    -1,
                    4,
                    self.reg_max + 1,
                )
            )

            ref_selected = (
                refs[
                    token_ids_t
                ]
                .float()
            )

            gt_boxes = torch.stack(
                gt_boxes_selected,
                dim=0,
            ).float()

            gt_xyxy = (
                box_cxcywh_to_xyxy(
                    gt_boxes
                )
            )

            mass = torch.stack(
                spatial_mass,
                dim=0,
            ).float()

            h = torch.stack(
                difficulty_selected,
                dim=0,
            ).float()

            u = torch.stack(
                uncertainty_selected,
                dim=0,
            ).float().detach()

            side_gain = (
                1.0
                + self.ufd_alpha_h
                * h[:, None]
                + float(alpha_u)
                * u
            )

            side_weight = (
                mass[:, None]
                * side_gain
            )

            u_bar = u.mean(
                dim=1
            )

            box_gain = (
                1.0
                + self.ufd_alpha_h
                * h
                + float(alpha_u)
                * u_bar
            )

            box_weight = (
                mass
                * box_gain
            )

            (
                target_corners,
                weight_right,
                weight_left,
            ) = bbox2distance(
                ref_selected,
                gt_xyxy,
                self.reg_max,
                reg_scale,
                up,
            )

            pred_flat = (
                pred_logits
                .reshape(
                    -1,
                    self.reg_max + 1,
                )
            )

            target_flat = (
                target_corners
                .reshape(-1)
            )

            left_bin = (
                target_flat.long()
            )

            right_bin = (
                left_bin + 1
            )

            fgl_per_side = (
                F.cross_entropy(
                    pred_flat,
                    left_bin,
                    reduction="none",
                )
                * weight_left.reshape(-1)
                +
                F.cross_entropy(
                    pred_flat,
                    right_bin,
                    reduction="none",
                )
                * weight_right.reshape(-1)
            )

            side_weight_flat = (
                side_weight
                .reshape(-1)
            )

            fgl_num = (
                fgl_per_side
                * side_weight_flat
            ).sum()

            side_den_local = (
                side_weight_flat.sum()
            )

            # Native Integral-equivalent decode.
            prob = F.softmax(
                pred_flat,
                dim=-1,
            )

            distance = (
                prob
                * project.reshape(
                    1,
                    -1,
                )
            ).sum(
                dim=-1
            ).reshape(
                -1,
                4,
            )

            pred_boxes = distance2bbox(
                ref_selected,
                distance,
                reg_scale,
            )

            l1_per_point = (
                torch.abs(
                    pred_boxes
                    - gt_boxes
                )
                .sum(
                    dim=-1
                )
            )

            bbox_num = (
                l1_per_point
                * box_weight
            ).sum()

            # Preserve the checkpoint-generating implementation
            # for ordinary sparse batches. Use the mathematically
            # equivalent O(N) implementation only for dense batches
            # where the pairwise N x N matrix is unsafe.
            if pred_boxes.shape[0] <= 4096:
                giou_per_point = (
                    1.0
                    - torch.diag(
                        generalized_box_iou(
                            box_cxcywh_to_xyxy(
                                pred_boxes
                            ),
                            gt_xyxy,
                        )
                    )
                )
            else:
                giou_per_point = (
                    1.0
                    - generalized_box_iou_aligned(
                        box_cxcywh_to_xyxy(
                            pred_boxes
                        ),
                        gt_xyxy,
                    )
                )

            giou_num = (
                giou_per_point
                * box_weight
            ).sum()

            box_den_local = (
                box_weight.sum()
            )

        else:
            # Keep graph connection on ranks with no local
            # selected UFD positions.
            fgl_num = zero
            bbox_num = zero
            giou_num = zero

            side_den_local = (
                zero.detach()
            )

            box_den_local = (
                zero.detach()
            )

        # DDP-correct weighted normalization:
        # use average denominator per rank, analogous
        # to DEIM's num_boxes normalization.
        side_den = (
            self._ufd_average_denominator(
                side_den_local
            )
        )

        box_den = (
            self._ufd_average_denominator(
                box_den_local
            )
        )

        loss_fgl = (
            fgl_num
            / side_den
        )

        loss_bbox = (
            bbox_num
            / box_den
        )

        loss_giou = (
            giou_num
            / box_den
        )

        # IMPORTANT:
        # 0.15 / 5 / 2 and lambda_UFD
        # are applied exactly ONCE here.
        return {
            "loss_ufd_fgl":
                loss_fgl
                * 0.15
                * float(lambda_ufd),

            "loss_ufd_bbox":
                loss_bbox
                * 5.0
                * float(lambda_ufd),

            "loss_ufd_giou":
                loss_giou
                * 2.0
                * float(lambda_ufd),
        }
