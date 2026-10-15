"""
Query-Competition Conditional Recalibration (QCCR).

Inference-only score recalibration:
- never changes boxes;
- never deletes queries;
- never changes predicted classes directly;
- computes one query-query IoU matrix per image;
- suppresses only non-leading same-class candidates;
- uses deterministic query-index tie breaking.
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
    """Convert [..., 4] boxes from cxcywh to xyxy."""
    cx, cy, w, h = boxes.unbind(dim=-1)

    return torch.stack(
        (
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ),
        dim=-1,
    )


def _pairwise_iou_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for batched boxes.

    Args:
        boxes: [B, Q, 4] in xyxy format.

    Returns:
        IoU matrix [B, Q, Q].
    """
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(
            "boxes must have shape [B, Q, 4], "
            f"got {tuple(boxes.shape)}"
        )

    left_top = torch.maximum(
        boxes[:, :, None, :2],
        boxes[:, None, :, :2],
    )
    right_bottom = torch.minimum(
        boxes[:, :, None, 2:],
        boxes[:, None, :, 2:],
    )

    wh = (right_bottom - left_top).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]

    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
    areas = sizes[..., 0] * sizes[..., 1]

    union = (
        areas[:, :, None]
        + areas[:, None, :]
        - intersection
    )

    return intersection / union.clamp_min(1.0e-9)


@torch.no_grad()
def compute_pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Original FP32 normalized-box IoU, reusable within one forward only."""
    return _pairwise_iou_xyxy(_cxcywh_to_xyxy(boxes.float()))


@torch.no_grad()
def compute_competition_risk(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    iou_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute same-class higher-score overlap risk.

    For candidate (query i, class c), the risk is:

        m_ic = max IoU(box_i, box_k)

    over queries k whose class-c logit is strictly higher. Exact score
    ties are resolved deterministically: the smaller query index wins.

    The highest-ranked candidate of each class always receives risk 0.

    Args:
        logits: [B, Q, C] raw class logits.
        boxes: [B, Q, 4] normalized cxcywh boxes.

    Returns:
        risk: [B, Q, C], bounded to [0, 1].
    """
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [B, Q, C], "
            f"got {tuple(logits.shape)}"
        )

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(
            "boxes must have shape [B, Q, 4], "
            f"got {tuple(boxes.shape)}"
        )

    if logits.shape[:2] != boxes.shape[:2]:
        raise ValueError(
            "logits and boxes must share [B, Q], got "
            f"{tuple(logits.shape)} and {tuple(boxes.shape)}"
        )

    if logits.device != boxes.device:
        raise ValueError(
            "logits and boxes must be on the same device, got "
            f"{logits.device} and {boxes.device}"
        )

    batch_size, num_queries, num_classes = logits.shape

    if iou_matrix is None:
        iou_matrix = compute_pairwise_iou(boxes)
    elif (
        iou_matrix.shape != (batch_size, num_queries, num_queries)
        or iou_matrix.device != logits.device
        or iou_matrix.dtype != torch.float32
    ):
        raise ValueError("precomputed IoU must be FP32 [B,Q,Q] on logits.device")

    query_index = torch.arange(
        num_queries,
        device=logits.device,
    )

    candidate_index = query_index.view(1, 1, num_queries, 1)
    rival_index = query_index.view(1, 1, 1, num_queries)

    # Batch the class comparisons. The rival dimension and its reduction
    # remain unchanged; no summation or score arithmetic is reordered.
    class_logits = logits.float().transpose(1, 2)
    candidate_logit = class_logits.unsqueeze(-1)
    rival_logit = class_logits.unsqueeze(-2)
    higher_ranked = (rival_logit > candidate_logit) | (
        (rival_logit == candidate_logit) & (rival_index < candidate_index)
    )
    masked_iou = iou_matrix.unsqueeze(1).expand(
        -1, num_classes, -1, -1
    ).masked_fill(~higher_ranked, -1.0)
    risk = masked_iou.max(dim=-1).values.clamp(min=0.0, max=1.0)
    risk = risk.transpose(1, 2).contiguous()

    if risk.shape != (
        batch_size,
        num_queries,
        num_classes,
    ):
        raise RuntimeError(
            f"unexpected risk shape: {tuple(risk.shape)}"
        )

    return risk.to(dtype=logits.dtype)


def apply_qccr(
    logits: torch.Tensor,
    competition_risk: torch.Tensor,
    beta0,
    beta_positive,
    beta_negative,
    duplicate_penalty,
) -> torch.Tensor:
    """Apply numerically stable QCCR in raw-logit space.

    Formula:

        z_tilde =
            beta0
            - beta_positive * softplus(-z)
            + beta_negative * softplus(z)
            - duplicate_penalty * m

    Constraints used during fitting:

        beta_positive > 0
        beta_negative > 0
        duplicate_penalty >= 0

    Exact identity is obtained with:

        beta0 = 0
        beta_positive = 1
        beta_negative = 1
        duplicate_penalty = 0
    """
    if logits.shape != competition_risk.shape:
        raise ValueError(
            "logits and competition_risk must have identical "
            f"shape, got {tuple(logits.shape)} and "
            f"{tuple(competition_risk.shape)}"
        )

    return (
        beta0
        - beta_positive * F.softplus(-logits)
        + beta_negative * F.softplus(logits)
        - duplicate_penalty * competition_risk
    )


class QueryCompetitionConditionalRecalibration(nn.Module):
    """Frozen inference module using fitted scalar parameters."""

    def __init__(
        self,
        beta0: float = 0.0,
        beta_positive: float = 1.0,
        beta_negative: float = 1.0,
        duplicate_penalty: float = 0.0,
    ) -> None:
        super().__init__()

        if beta_positive <= 0.0:
            raise ValueError("beta_positive must be positive")

        if beta_negative <= 0.0:
            raise ValueError("beta_negative must be positive")

        if duplicate_penalty < 0.0:
            raise ValueError(
                "duplicate_penalty must be non-negative"
            )

        values = torch.tensor(
            [
                beta0,
                beta_positive,
                beta_negative,
                duplicate_penalty,
            ],
            dtype=torch.float32,
        )
        self.register_buffer(
            "_qccr_parameters",
            values,
            persistent=False,
        )

    @property
    def beta0(self) -> torch.Tensor:
        return self._qccr_parameters[0]

    @property
    def beta_positive(self) -> torch.Tensor:
        return self._qccr_parameters[1]

    @property
    def beta_negative(self) -> torch.Tensor:
        return self._qccr_parameters[2]

    @property
    def duplicate_penalty(self) -> torch.Tensor:
        return self._qccr_parameters[3]

    def _calibrate(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return calibrated logits and their detached competition risk."""
        risk = compute_competition_risk(
            logits.detach(),
            boxes.detach(),
        )

        parameters = self._qccr_parameters.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        calibrated = apply_qccr(
            logits=logits,
            competition_risk=risk,
            beta0=parameters[0],
            beta_positive=parameters[1],
            beta_negative=parameters[2],
            duplicate_penalty=parameters[3],
        )
        return calibrated, risk

    def forward(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        calibrated, _ = self._calibrate(logits, boxes)
        return calibrated

    def forward_with_risk(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._calibrate(logits, boxes)

    def extra_repr(self) -> str:
        values = self._qccr_parameters.tolist()

        return (
            f"beta0={values[0]:.6g}, "
            f"beta_positive={values[1]:.6g}, "
            f"beta_negative={values[2]:.6g}, "
            f"duplicate_penalty={values[3]:.6g}"
        )
