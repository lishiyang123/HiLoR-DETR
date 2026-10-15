"""Host-independent auxiliary geometric supervision for RT-DETR memory.

RT-DETR directly regresses four box coordinates and exposes no native
boundary-distribution entropy. The formal second-host configuration therefore
locks ``alpha_u_max=0``. This module retains difficulty-aware reachable-token
assignment, deterministic conflict resolution, and FGL/L1/GIoU supervision.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from src.misc.dist import get_world_size, is_dist_available_and_initialized

from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)


def weighting_function(reg_max, up, reg_scale):
    upper1 = torch.abs(up[0]) * torch.abs(reg_scale)
    upper2 = upper1 * 2
    step = (upper1 + 1) ** (2 / (reg_max - 2))
    left = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [(step ** i) - 1 for i in range(1, reg_max // 2)]
    return torch.cat(
        [-upper2, *left, torch.zeros_like(up[:1]), *right, upper2], dim=0
    )


def _translate_distance(values, reg_max, reg_scale, up):
    values = values.reshape(-1)
    bins = weighting_function(reg_max, up, reg_scale)
    differences = bins.unsqueeze(0) - values.unsqueeze(1)
    left_index = (differences <= 0).sum(dim=1) - 1
    index = left_index.float()
    right_weight = torch.zeros_like(index)
    left_weight = torch.zeros_like(index)
    valid = (index >= 0) & (index < reg_max)
    valid_index = index[valid].long()
    left_value = bins[valid_index]
    right_value = bins[valid_index + 1]
    left_difference = torch.abs(values[valid] - left_value)
    right_difference = torch.abs(right_value - values[valid])
    right_weight[valid] = left_difference / (left_difference + right_difference)
    left_weight[valid] = 1.0 - right_weight[valid]
    below = index < 0
    left_weight[below] = 1.0
    index[below] = 0.0
    above = index >= reg_max
    right_weight[above] = 1.0
    index[above] = reg_max - 0.1
    return index, right_weight, left_weight


def bbox2distance(points, boxes_xyxy, reg_max, reg_scale, up):
    scale = torch.abs(reg_scale)
    left = (points[:, 0] - boxes_xyxy[:, 0]) / (points[:, 2] / scale + 1e-16) - 0.5 * scale
    top = (points[:, 1] - boxes_xyxy[:, 1]) / (points[:, 3] / scale + 1e-16) - 0.5 * scale
    right = (boxes_xyxy[:, 2] - points[:, 0]) / (points[:, 2] / scale + 1e-16) - 0.5 * scale
    bottom = (boxes_xyxy[:, 3] - points[:, 1]) / (points[:, 3] / scale + 1e-16) - 0.5 * scale
    distance = torch.stack((left, top, right, bottom), dim=-1)
    index, right_weight, left_weight = _translate_distance(
        distance, reg_max, reg_scale, up
    )
    return index.clamp(0, reg_max - 0.1).detach(), right_weight.detach(), left_weight.detach()


def distance2bbox(points, distance, reg_scale):
    scale = torch.abs(reg_scale)
    x1 = points[..., 0] - (0.5 * scale + distance[..., 0]) * (points[..., 2] / scale)
    y1 = points[..., 1] - (0.5 * scale + distance[..., 1]) * (points[..., 3] / scale)
    x2 = points[..., 0] + (0.5 * scale + distance[..., 2]) * (points[..., 2] / scale)
    y2 = points[..., 1] + (0.5 * scale + distance[..., 3]) * (points[..., 3] / scale)
    return box_xyxy_to_cxcywh(torch.stack((x1, y1, x2, y2), dim=-1))


class RTDETRUFD:
    """Mixin used by :class:`HiLoRSetCriterion`."""

    def _init_ufd(
        self,
        *,
        ufd_enabled,
        ufd_cdf_path,
        ufd_total_epochs,
        ufd_k,
        ufd_base_size,
        ufd_alpha_h,
        ufd_alpha_u_max,
        ufd_lambda_max,
        ufd_lambda_ramp_fraction,
        ufd_reg_max,
    ):
        self.ufd_enabled = bool(ufd_enabled)
        self.ufd_total_epochs = int(ufd_total_epochs)
        self.ufd_k = int(ufd_k)
        self.ufd_base_size = float(ufd_base_size)
        self.ufd_alpha_h = float(ufd_alpha_h)
        self.ufd_alpha_u_max = float(ufd_alpha_u_max)
        self.ufd_lambda_max = float(ufd_lambda_max)
        self.ufd_lambda_ramp_fraction = float(ufd_lambda_ramp_fraction)
        self.ufd_reg_max = int(ufd_reg_max)
        if self.ufd_alpha_u_max != 0.0:
            raise ValueError(
                "formal RT-DETR host has no boundary entropy; ufd_alpha_u_max must be 0"
            )
        if self.ufd_total_epochs <= 0 or self.ufd_k <= 0:
            raise ValueError("UFD epochs and token quota must be positive")

        if self.ufd_enabled:
            if not ufd_cdf_path:
                raise ValueError("ufd_cdf_path is required when UFD is enabled")
            with open(ufd_cdf_path, "r", encoding="utf-8") as handle:
                cdf = json.load(handle)
            area = torch.tensor(cdf["area_640_sorted"], dtype=torch.float32)
            aspect = torch.tensor(cdf["aspect_difficulty_sorted"], dtype=torch.float32)
            ioa = torch.tensor(cdf["max_ioa_sorted"], dtype=torch.float32)
            if not area.numel() or not (area.numel() == aspect.numel() == ioa.numel()):
                raise RuntimeError("invalid frozen Train difficulty distributions")
        else:
            area = aspect = ioa = torch.empty(0, dtype=torch.float32)
        self.register_buffer("_ufd_area_cdf", area, persistent=False)
        self.register_buffer("_ufd_aspect_cdf", aspect, persistent=False)
        self.register_buffer("_ufd_ioa_cdf", ioa, persistent=False)

    @staticmethod
    def _empirical_cdf(values, sorted_samples):
        return torch.searchsorted(sorted_samples, values, right=True).to(values.dtype) / float(
            sorted_samples.numel()
        )

    @staticmethod
    def _max_ioa(boxes_xyxy):
        count = int(boxes_xyxy.shape[0])
        if count <= 1:
            return boxes_xyxy.new_zeros(count)
        left_top = torch.maximum(boxes_xyxy[:, None, :2], boxes_xyxy[None, :, :2])
        right_bottom = torch.minimum(boxes_xyxy[:, None, 2:], boxes_xyxy[None, :, 2:])
        wh = (right_bottom - left_top).clamp_min(0.0)
        intersection = wh[..., 0] * wh[..., 1]
        size = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp_min(0.0)
        area = size[:, 0] * size[:, 1]
        ioa = intersection / area[:, None].clamp_min(1.0e-12)
        ioa.fill_diagonal_(0.0)
        return ioa.max(dim=1).values

    @torch.no_grad()
    def _difficulty(self, boxes):
        if boxes.numel() == 0:
            return boxes.new_zeros(0)
        width = boxes[:, 2].clamp_min(1.0e-12)
        height = boxes[:, 3].clamp_min(1.0e-12)
        area = width * height * self.ufd_base_size ** 2
        aspect = torch.abs(torch.log(width / height))
        overlap = self._max_ioa(box_cxcywh_to_xyxy(boxes.float()))
        return (
            1.0 - self._empirical_cdf(area, self._ufd_area_cdf)
            + self._empirical_cdf(aspect, self._ufd_aspect_cdf)
            + self._empirical_cdf(overlap, self._ufd_ioa_cdf)
        ).div(3.0).clamp(0.0, 1.0).detach()

    def _lambda(self, epoch, step, epoch_step):
        progress = (float(epoch) + float(step) / max(float(epoch_step), 1.0)) / float(
            self.ufd_total_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        if self.ufd_lambda_ramp_fraction > 0.0:
            scale = min(progress / self.ufd_lambda_ramp_fraction, 1.0)
        else:
            scale = 1.0
        return self.ufd_lambda_max * scale

    @torch.no_grad()
    def _candidate_queues(
        self, refs, valid_mask, spatial_shapes, gt_xyxy, difficulty, reg_scale,
        minimum_distance, maximum_distance,
    ):
        shapes = spatial_shapes.detach().cpu().tolist()
        if len(shapes) != 3:
            raise RuntimeError("UFD expects F8/F16/F32 memory")
        ranges, start = [], 0
        for height, width in shapes:
            end = start + int(height) * int(width)
            ranges.append((start, end))
            start = end
        if start != refs.shape[0]:
            raise RuntimeError("spatial shape/token mismatch")

        num_gt = len(gt_xyxy)
        queues = [[] for _ in range(num_gt)]
        unresolved = set(range(num_gt))
        difficulty_cpu = difficulty.detach().cpu()
        scale = torch.abs(reg_scale).to(refs)
        for begin, end in ranges:
            if not unresolved:
                break
            level_refs = refs[begin:end]
            centers = level_refs[:, :2]
            center_x, center_y = centers[:, 0][None], centers[:, 1][None]
            x1, y1 = gt_xyxy[:, 0][:, None], gt_xyxy[:, 1][:, None]
            x2, y2 = gt_xyxy[:, 2][:, None], gt_xyxy[:, 3][:, None]
            inside = (
                valid_mask[begin:end][None]
                & (center_x >= x1) & (center_x <= x2)
                & (center_y >= y1) & (center_y <= y2)
            )
            ref_x, ref_y = level_refs[:, 0][None], level_refs[:, 1][None]
            ref_w, ref_h = level_refs[:, 2][None], level_refs[:, 3][None]
            scale_w, scale_h = ref_w / scale + 1e-16, ref_h / scale + 1e-16
            distances = torch.stack(
                ((ref_x - x1) / scale_w - 0.5 * scale,
                 (ref_y - y1) / scale_h - 0.5 * scale,
                 (x2 - ref_x) / scale_w - 0.5 * scale,
                 (y2 - ref_y) / scale_h - 0.5 * scale),
                dim=-1,
            )
            reachable = ((distances >= minimum_distance) & (distances <= maximum_distance)).all(-1)
            candidate = inside & reachable
            dl, dt = (center_x - x1).clamp_min(0.0), (center_y - y1).clamp_min(0.0)
            dr, db = (x2 - center_x).clamp_min(0.0), (y2 - center_y).clamp_min(0.0)
            centerness = torch.sqrt(
                (torch.minimum(dl, dr) / torch.maximum(dl, dr).clamp_min(1e-12)
                 * torch.minimum(dt, db) / torch.maximum(dt, db).clamp_min(1e-12)).clamp_min(0.0)
            )
            candidate_cpu, centerness_cpu = candidate.cpu(), centerness.cpu()
            resolved = []
            for gt_index in sorted(unresolved):
                token_offset = torch.nonzero(candidate_cpu[gt_index], as_tuple=False).flatten()
                if token_offset.numel() == 0:
                    continue
                values = centerness_cpu[gt_index, token_offset]
                order = torch.argsort(values, descending=True, stable=True)
                token_offset, values = token_offset[order], values[order]
                hard = float(difficulty_cpu[gt_index])
                queues[gt_index] = [
                    (int(offset + begin), float(value), float(value) * (1.0 + self.ufd_alpha_h * hard))
                    for offset, value in zip(token_offset.tolist(), values.tolist())
                ]
                resolved.append(gt_index)
            unresolved.difference_update(resolved)
        return queues

    @torch.no_grad()
    def _resolve_conflicts(self, queues):
        assigned = [[] for _ in queues]
        cursor = [0 for _ in queues]
        claimed = set()
        for _ in range(self.ufd_k):
            pending = {
                index for index, queue in enumerate(queues)
                if len(assigned[index]) < self.ufd_k and len(assigned[index]) < len(queue)
            }
            while pending:
                proposals, exhausted = {}, []
                for gt_index in sorted(pending):
                    queue = queues[gt_index]
                    while cursor[gt_index] < len(queue) and queue[cursor[gt_index]][0] in claimed:
                        cursor[gt_index] += 1
                    if cursor[gt_index] >= len(queue):
                        exhausted.append(gt_index)
                        continue
                    token, center, priority = queue[cursor[gt_index]]
                    proposals.setdefault(token, []).append((gt_index, center, priority))
                for gt_index in exhausted:
                    pending.discard(gt_index)
                if not proposals:
                    break
                for token in sorted(proposals):
                    candidates = proposals[token]
                    winner = max(candidates, key=lambda item: (item[2], -item[0]))
                    gt_index, center = winner[0], winner[1]
                    assigned[gt_index].append((token, center))
                    claimed.add(token)
                    cursor[gt_index] += 1
                    pending.discard(gt_index)
                    for loser, _, _ in candidates:
                        if loser != gt_index:
                            cursor[loser] += 1
        return assigned

    @staticmethod
    def _aligned_giou(boxes1, boxes2):
        # Preserve the ordinary checkpoint-generating diagonal implementation
        # for sparse batches. Avoid its quadratic matrix only for the same
        # dense-batch threshold used by the optimized donor.
        if boxes1.shape[0] <= 4096:
            return torch.diagonal(generalized_box_iou(boxes1, boxes2))
        if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
            raise ValueError("aligned GIoU requires equal [N,4] tensors")
        if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
            raise ValueError("boxes1 contains a degenerate box")
        if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
            raise ValueError("boxes2 contains a degenerate box")
        area1 = ((boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0)).prod(-1)
        area2 = ((boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0)).prod(-1)
        intersection_size = (
            torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
            - torch.maximum(boxes1[:, :2], boxes2[:, :2])
        ).clamp_min(0.0)
        intersection = intersection_size.prod(-1)
        union = area1 + area2 - intersection
        iou = intersection / union
        enclosing_size = (
            torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
            - torch.minimum(boxes1[:, :2], boxes2[:, :2])
        ).clamp_min(0.0)
        enclosing_area = enclosing_size.prod(-1)
        return iou - (enclosing_area - union) / enclosing_area

    @staticmethod
    def _average_denominator(value):
        denominator = value.detach().clone()
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(denominator)
            denominator = denominator / get_world_size()
        return denominator.clamp_min(1.0e-12)

    def _ufd_loss(self, outputs, targets, *, epoch, step, epoch_step):
        if not self.ufd_enabled:
            return {}
        logits = outputs["ufd_logits"]
        zero = logits.sum() * 0.0
        lambda_ufd = self._lambda(epoch, step, epoch_step)
        if lambda_ufd <= 0.0:
            return {"loss_ufd_fgl": zero, "loss_ufd_bbox": zero, "loss_ufd_giou": zero}

        refs = outputs["ufd_ref_points"].float()
        valid = outputs["ufd_valid_mask"].bool()
        shapes = outputs["ufd_spatial_shapes"]
        up = outputs["ufd_up"].float()
        scale = outputs["ufd_reg_scale"].float()
        if logits.shape[1] != refs.shape[0]:
            raise RuntimeError("UFD logit/reference token mismatch")
        projection = weighting_function(self.ufd_reg_max, up, scale).float()
        minimum, maximum = float(projection.min().item()), float(projection.max().item())

        batch_ids, token_ids, gt_selected, mass_selected, difficulty_selected = [], [], [], [], []
        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"].float()
            if gt_boxes.numel() == 0:
                continue
            difficulty = self._difficulty(gt_boxes)
            queues = self._candidate_queues(
                refs, valid, shapes, box_cxcywh_to_xyxy(gt_boxes), difficulty,
                scale, minimum, maximum,
            )
            for gt_index, selection in enumerate(self._resolve_conflicts(queues)):
                if not selection:
                    continue
                center = torch.tensor([item[1] for item in selection], device=logits.device)
                mass = center / center.sum() if float(center.sum()) > 1e-12 else torch.ones_like(center) / len(center)
                for position, (token, _) in enumerate(selection):
                    batch_ids.append(batch_index)
                    token_ids.append(token)
                    gt_selected.append(gt_boxes[gt_index])
                    mass_selected.append(mass[position])
                    difficulty_selected.append(difficulty[gt_index])

        if batch_ids:
            batch_ids = torch.tensor(batch_ids, dtype=torch.long, device=logits.device)
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
            prediction = logits[batch_ids, token_ids].float().reshape(-1, 4, self.ufd_reg_max + 1)
            selected_refs = refs[token_ids]
            gt_boxes = torch.stack(gt_selected).float()
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            mass = torch.stack(mass_selected).float()
            difficulty = torch.stack(difficulty_selected).float()
            weight = mass * (1.0 + self.ufd_alpha_h * difficulty)
            targets_distance, right_weight, left_weight = bbox2distance(
                selected_refs, gt_xyxy, self.ufd_reg_max, scale, up
            )
            prediction_flat = prediction.reshape(-1, self.ufd_reg_max + 1)
            target_flat = targets_distance.reshape(-1)
            fgl = (
                F.cross_entropy(prediction_flat, target_flat.long(), reduction="none")
                * left_weight.reshape(-1)
                + F.cross_entropy(prediction_flat, target_flat.long() + 1, reduction="none")
                * right_weight.reshape(-1)
            )
            side_weight = weight[:, None].expand(-1, 4).reshape(-1)
            fgl_numerator = (fgl * side_weight).sum()
            side_denominator = side_weight.sum()
            distance = (
                F.softmax(prediction_flat, dim=-1) * projection.reshape(1, -1)
            ).sum(-1).reshape(-1, 4)
            pred_boxes = distance2bbox(selected_refs, distance, scale)
            bbox_numerator = (torch.abs(pred_boxes - gt_boxes).sum(-1) * weight).sum()
            # The auxiliary distribution spans signed offsets. Before it has
            # converged, an extreme pair of opposite-side predictions can
            # decode to a negative width or height. Keep the FGL and L1 terms
            # on that raw prediction, but make the GIoU input a valid box.
            # This is an identity operation for every ordinary decoded box.
            giou_boxes = torch.cat(
                (pred_boxes[:, :2], pred_boxes[:, 2:].clamp_min(0.0)), dim=-1
            )
            giou = 1.0 - self._aligned_giou(
                box_cxcywh_to_xyxy(giou_boxes), gt_xyxy
            )
            giou_numerator = (giou * weight).sum()
            box_denominator = weight.sum()
        else:
            fgl_numerator = bbox_numerator = giou_numerator = zero
            side_denominator = box_denominator = zero.detach()

        return {
            "loss_ufd_fgl": fgl_numerator / self._average_denominator(side_denominator) * 0.15 * lambda_ufd,
            "loss_ufd_bbox": bbox_numerator / self._average_denominator(box_denominator) * 5.0 * lambda_ufd,
            "loss_ufd_giou": giou_numerator / self._average_denominator(box_denominator) * 2.0 * lambda_ufd,
        }
