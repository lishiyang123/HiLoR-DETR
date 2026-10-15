"""Validation and hashing for frozen detector capture files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    import torch

    payload = torch.load(
        Path(path), map_location="cpu", weights_only=False, mmap=bool(mmap)
    )
    if not isinstance(payload, dict):
        raise ValueError("capture root must be a mapping")
    if int(payload.get("format_version", -1)) != 2:
        raise ValueError("expected capture format_version 2")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("capture contains no records")
    required = {
        "image_id",
        "pred_logits",
        "pred_boxes",
        "orig_size",
        "gt_boxes",
        "gt_labels",
    }
    for index, record in enumerate(records):
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"record {index} is missing fields: {missing}")
    return payload


def record_ground_truth(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    boxes = record["gt_boxes"]
    labels = record["gt_labels"]
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    boxes_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(boxes_array) != len(labels_array):
        raise ValueError("GT boxes and labels have inconsistent lengths")
    return boxes_array, labels_array
