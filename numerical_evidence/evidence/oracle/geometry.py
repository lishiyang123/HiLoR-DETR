"""Class- and score-independent oracle coverage of the full final query pool."""

import numpy as np


def best_iou(pred_xyxy, gt_xyxy):
    pred = np.asarray(pred_xyxy, dtype=np.float64)
    gt = np.asarray(gt_xyxy, dtype=np.float64)
    for name, boxes in (("pred", pred), ("gt", gt)):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"{name}: expected [N,4] xyxy")
        if not np.isfinite(boxes).all():
            raise ValueError(f"{name}: nonfinite box")
        if name == "gt" and (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("gt: degenerate or reversed box")
    if len(pred) == 0:
        raise ValueError("Full query pool cannot be empty")
    if len(gt) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    intersection_wh = np.maximum(
        np.minimum(gt[:, None, 2:], pred[None, :, 2:])
        - np.maximum(gt[:, None, :2], pred[None, :, :2]), 0.0
    )
    intersection = intersection_wh.prod(axis=2)
    union = (gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None] + np.maximum(
        pred[:, 2:] - pred[:, :2], 0.0
    ).prod(axis=1)[None, :] - intersection
    ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    ious[:, (pred[:, 2:] <= pred[:, :2]).any(axis=1)] = 0.0
    query_ids = ious.argmax(axis=1)
    return ious[np.arange(len(gt)), query_ids], query_ids


def cxcywh_to_pixel_xyxy(boxes, width, height):
    boxes = np.asarray(boxes, dtype=np.float64)
    center, size = boxes[..., :2], boxes[..., 2:]
    xyxy = np.concatenate((center - size / 2, center + size / 2), axis=-1)
    return xyxy * np.array([width, height, width, height], dtype=np.float64)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite per-GT vector")
    return {
        "gt_count": len(values),
        "mean_best_iou": float(values.mean()),
        "median_best_iou": float(np.median(values)),
        "oracle_coverage_iou_ge_075": float((values >= 0.75).mean()),
    }
