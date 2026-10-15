"""Adapters for the frozen ``Q x C`` detector-candidate representation.

The formal detector emits one box per query and one logit per query/class
pair.  This module expands those tensors into a flat, auditable candidate
pool without changing coordinates, labels, or scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    cx, cy, width, height = np.moveaxis(boxes, -1, 0)
    return np.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        axis=-1,
    )


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for ``xyxy`` boxes, with invalid areas mapped to 0."""

    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0.0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0.0), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def stable_descending_order(scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    """Sort by descending score and then ascending immutable candidate id."""

    scores = np.asarray(scores, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    return np.lexsort((candidate_ids, -scores))


@dataclass(frozen=True)
class CandidateSet:
    image_id: int
    boxes_xyxy: np.ndarray
    raw_logits: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    query_ids: np.ndarray
    candidate_ids: np.ndarray
    image_height: int
    image_width: int

    def __post_init__(self) -> None:
        count = len(self.scores)
        fields = (
            self.boxes_xyxy,
            self.raw_logits,
            self.labels,
            self.query_ids,
            self.candidate_ids,
        )
        if any(len(field) != count for field in fields):
            raise ValueError("candidate fields have inconsistent lengths")
        if self.boxes_xyxy.shape != (count, 4):
            raise ValueError("boxes_xyxy must have shape [N, 4]")
        if not np.all(np.isfinite(self.raw_logits)):
            raise ValueError("candidate logits contain non-finite values")
        if not np.all(np.isfinite(self.boxes_xyxy)):
            raise ValueError("candidate boxes contain non-finite values")

    def topk(
        self,
        scores: np.ndarray | None = None,
        k: int = 300,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        values = self.scores if scores is None else np.asarray(scores, dtype=np.float64)
        if values.shape != self.scores.shape:
            raise ValueError("replacement scores have the wrong shape")
        if active_mask is None:
            eligible = np.arange(len(values), dtype=np.int64)
        else:
            mask = np.asarray(active_mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError("active_mask has the wrong shape")
            eligible = np.flatnonzero(mask)
        if len(eligible) == 0:
            return eligible
        local_order = stable_descending_order(values[eligible], self.candidate_ids[eligible])
        return eligible[local_order[: min(int(k), len(local_order))]]

    def to_coco(self, indices: np.ndarray, scores: np.ndarray | None = None) -> list[dict]:
        values = self.scores if scores is None else np.asarray(scores, dtype=np.float64)
        output: list[dict] = []
        for index in np.asarray(indices, dtype=np.int64):
            x1, y1, x2, y2 = self.boxes_xyxy[index]
            output.append(
                {
                    "image_id": int(self.image_id),
                    "category_id": int(self.labels[index]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(values[index]),
                    "query_id": int(self.query_ids[index]),
                    "candidate_id": int(self.candidate_ids[index]),
                }
            )
        return output


def candidates_from_record(record: Mapping[str, Any]) -> CandidateSet:
    logits = _as_numpy(record["pred_logits"], np.float64)
    query_boxes = _as_numpy(record["pred_boxes"], np.float64)
    if logits.ndim != 2:
        raise ValueError("pred_logits must have shape [Q, C]")
    num_queries, num_classes = logits.shape
    if query_boxes.shape != (num_queries, 4):
        raise ValueError("pred_boxes must have shape [Q, 4]")

    orig_size = _as_numpy(record["orig_size"], np.int64).reshape(-1)
    if orig_size.shape != (2,):
        raise ValueError("orig_size must contain [height, width]")
    height, width = [int(value) for value in orig_size]

    boxes_norm = cxcywh_to_xyxy(query_boxes)
    boxes_abs = boxes_norm * np.asarray([width, height, width, height], dtype=np.float64)
    boxes_flat = np.repeat(boxes_abs, num_classes, axis=0)
    query_ids = np.repeat(np.arange(num_queries, dtype=np.int64), num_classes)
    labels = np.tile(np.arange(num_classes, dtype=np.int64), num_queries)
    candidate_ids = query_ids * num_classes + labels
    raw_logits = logits.reshape(-1)

    return CandidateSet(
        image_id=int(record["image_id"]),
        boxes_xyxy=boxes_flat,
        raw_logits=raw_logits,
        scores=sigmoid(raw_logits),
        labels=labels,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        image_height=height,
        image_width=width,
    )
