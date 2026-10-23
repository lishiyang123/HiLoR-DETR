#!/usr/bin/env python3
"""Extract the reported metrics from saved COCO evaluation arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


ORDER = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)


def mean_valid(values: object) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[array > -1]
    return float(array.mean()) if array.size else float("nan")


def load_evaluation(path: Path) -> dict[str, object]:
    evaluation = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(evaluation, dict) and "bbox" in evaluation:
        evaluation = evaluation["bbox"]
    if (
        isinstance(evaluation, dict)
        and "eval" in evaluation
        and "precision" not in evaluation
    ):
        evaluation = evaluation["eval"]
    if not isinstance(evaluation, dict):
        raise RuntimeError("unrecognized eval.pth object")
    required = {"precision", "recall", "params"}
    if not required <= set(evaluation):
        raise RuntimeError("eval.pth lacks precision, recall, or params")
    return evaluation


def extract(path: Path) -> dict[str, float]:
    evaluation = load_evaluation(path)
    precision = np.asarray(evaluation["precision"], dtype=np.float64)
    recall = np.asarray(evaluation["recall"], dtype=np.float64)
    params = evaluation["params"]
    field = (
        (lambda key: params[key])
        if isinstance(params, dict)
        else (lambda key: getattr(params, key))
    )
    ious = np.asarray(field("iouThrs"), dtype=np.float64)
    areas = list(field("areaRngLbl"))
    max_dets = [int(value) for value in field("maxDets")]

    def threshold(value: float) -> int:
        found = np.flatnonzero(np.isclose(ious, value, rtol=0, atol=1e-12))
        if len(found) != 1:
            raise RuntimeError(f"IoU threshold missing or duplicated: {value}")
        return int(found[0])

    def ap(
        iou: float | None = None,
        area: str = "all",
        max_det: int = 100,
    ) -> float:
        area_index = areas.index(area)
        max_det_index = max_dets.index(max_det)
        values = precision[:, :, :, area_index, max_det_index]
        if iou is not None:
            values = precision[
                threshold(iou), :, :, area_index, max_det_index
            ]
        return mean_valid(values)

    def ar(area: str = "all", max_det: int = 100) -> float:
        return mean_valid(
            recall[:, :, areas.index(area), max_dets.index(max_det)]
        )

    return {
        "AP": ap(),
        "AP50": ap(0.50),
        "AP75": ap(0.75),
        "APs": ap(area="small"),
        "APm": ap(area="medium"),
        "APl": ap(area="large"),
        "AR1": ar(max_det=1),
        "AR10": ar(max_det=10),
        "AR100": ar(max_det=100),
        "ARs": ar(area="small"),
        "ARm": ar(area="medium"),
        "ARl": ar(area="large"),
        "AP80": ap(0.80),
        "AP85": ap(0.85),
        "AP90": ap(0.90),
        "AP95": ap(0.95),
    }
