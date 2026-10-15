"""Query-competition conditional recalibration for the RT-DETR host.

The implementation is deliberately host-agnostic: it consumes the complete
``[B, Q, C]`` raw-logit tensor and normalized ``cxcywh`` boxes, changes scores
only, and leaves RT-DETR's native global query-class Top-K step untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "QueryCompetitionConditionalRecalibration",
    "apply_qccr",
    "compute_competition_risk",
    "compute_pairwise_iou",
]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        dim=-1,
    )


def _pairwise_iou_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be [B,Q,4], got {tuple(boxes.shape)}")

    left_top = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    right_bottom = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    wh = (right_bottom - left_top).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
    areas = sizes[..., 0] * sizes[..., 1]
    union = areas[:, :, None] + areas[:, None, :] - intersection
    return intersection / union.clamp_min(1.0e-9)


@torch.no_grad()
def compute_pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Return the FP32 normalized-box IoU matrix ``[B,Q,Q]``."""
    return _pairwise_iou_xyxy(_cxcywh_to_xyxy(boxes.float()))


@torch.no_grad()
def compute_competition_risk(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    iou_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Maximum overlap with a higher-ranked same-class query.

    Strictly larger logits lead. Exact ties use the smaller query index, which
    makes the result deterministic without perturbing the logits.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,Q,C], got {tuple(logits.shape)}")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be [B,Q,4], got {tuple(boxes.shape)}")
    if logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("logits and boxes must share [B,Q]")
    if logits.device != boxes.device:
        raise ValueError("logits and boxes must share one device")

    batch_size, num_queries, num_classes = logits.shape
    if iou_matrix is None:
        iou_matrix = compute_pairwise_iou(boxes)
    expected = (batch_size, num_queries, num_queries)
    if (
        iou_matrix.shape != expected
        or iou_matrix.device != logits.device
        or iou_matrix.dtype != torch.float32
    ):
        raise ValueError("precomputed IoU must be FP32 [B,Q,Q] on logits.device")

    query_index = torch.arange(num_queries, device=logits.device)
    candidate_index = query_index.view(1, 1, num_queries, 1)
    rival_index = query_index.view(1, 1, 1, num_queries)
    class_logits = logits.float().transpose(1, 2)
    candidate_logit = class_logits.unsqueeze(-1)
    rival_logit = class_logits.unsqueeze(-2)
    higher_ranked = (rival_logit > candidate_logit) | (
        (rival_logit == candidate_logit) & (rival_index < candidate_index)
    )
    masked_iou = iou_matrix.unsqueeze(1).expand(
        -1, num_classes, -1, -1
    ).masked_fill(~higher_ranked, -1.0)
    risk = masked_iou.max(dim=-1).values.clamp_(0.0, 1.0)
    return risk.transpose(1, 2).contiguous().to(dtype=logits.dtype)


def apply_qccr(
    logits: torch.Tensor,
    competition_risk: torch.Tensor,
    beta0,
    beta_positive,
    beta_negative,
    duplicate_penalty,
) -> torch.Tensor:
    if logits.shape != competition_risk.shape:
        raise ValueError("logits and competition_risk must have identical shape")
    return (
        beta0
        - beta_positive * F.softplus(-logits)
        + beta_negative * F.softplus(logits)
        - duplicate_penalty * competition_risk
    )


class QueryCompetitionConditionalRecalibration(nn.Module):
    """Frozen fitted QCCR mapping."""

    def __init__(
        self,
        beta0: float = 0.0,
        beta_positive: float = 1.0,
        beta_negative: float = 1.0,
        duplicate_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if beta_positive <= 0.0 or beta_negative <= 0.0:
            raise ValueError("QCCR beta_positive and beta_negative must be positive")
        if duplicate_penalty < 0.0:
            raise ValueError("QCCR duplicate_penalty must be non-negative")
        self.register_buffer(
            "_qccr_parameters",
            torch.tensor(
                [beta0, beta_positive, beta_negative, duplicate_penalty],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # Fitting captures are FP32. Keep the active QCCR path in FP32 as well
        # so QCCR-only and QCCR–UMR use one numerical protocol. The completely
        # disabled postprocessor still bypasses this module bit-for-bit.
        working_logits = logits.float()
        risk = compute_competition_risk(
            working_logits.detach(), boxes.detach().float()
        )
        parameters = self._qccr_parameters.to(working_logits)
        return apply_qccr(working_logits, risk, *parameters.unbind())
