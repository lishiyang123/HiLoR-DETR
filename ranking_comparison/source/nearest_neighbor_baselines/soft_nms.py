"""Deterministic class-wise linear and Gaussian Soft-NMS."""

from __future__ import annotations

import numpy as np

from .candidate_adapter import CandidateSet, pairwise_iou, stable_descending_order


def _soft_nms_one_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    method: str,
    iou_threshold: float,
    sigma: float,
) -> np.ndarray:
    if method not in {"linear", "gaussian"}:
        raise ValueError("method must be 'linear' or 'gaussian'")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    updated = np.asarray(scores, dtype=np.float64).copy()
    overlap_matrix = pairwise_iou(boxes, boxes)
    remaining = np.arange(len(updated), dtype=np.int64)
    while len(remaining) > 0:
        order = stable_descending_order(updated[remaining], candidate_ids[remaining])
        best = remaining[order[0]]
        remaining = remaining[order[1:]]
        if len(remaining) == 0:
            break
        overlaps = overlap_matrix[best, remaining]
        if method == "linear":
            decay = np.ones_like(overlaps)
            affected = overlaps > iou_threshold
            decay[affected] = 1.0 - overlaps[affected]
        else:
            decay = np.exp(-(overlaps * overlaps) / sigma)
        updated[remaining] *= decay
    return updated


def soft_nms_scores(
    candidates: CandidateSet,
    *,
    method: str,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
) -> np.ndarray:
    """Return new scores without changing any box, label, or candidate id."""

    result = candidates.scores.copy()
    for label in np.unique(candidates.labels):
        indices = np.flatnonzero(candidates.labels == label)
        result[indices] = _soft_nms_one_class(
            candidates.boxes_xyxy[indices],
            candidates.scores[indices],
            candidates.candidate_ids[indices],
            method=method,
            iou_threshold=iou_threshold,
            sigma=sigma,
        )
    return result
