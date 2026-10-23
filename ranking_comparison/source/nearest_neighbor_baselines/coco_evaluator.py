"""COCO evaluation with the same max-detection contract as the paper."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_api():
    from importlib.metadata import version
    from faster_coco_eval import COCO, COCOeval_faster
    if version('faster-coco-eval') != '1.7.2':
        raise RuntimeError('Expected faster-coco-eval 1.7.2, matching the main results')
    return COCO, COCOeval_faster


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(np.mean(valid)) if len(valid) else float("nan")


def evaluate_coco_detections(
    annotation_path: Path,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    COCO, COCOeval = _load_api()
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(Path(annotation_path)))
        serializable = [
            {key: value for key, value in detection.items() if key in {
                "image_id", "category_id", "bbox", "score"
            }}
            for detection in detections
        ]
        predictions = ground_truth.loadRes(serializable)
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    thresholds = np.asarray(evaluator.params.iouThrs)
    threshold_ap = {}
    for threshold_index, threshold in enumerate(thresholds):
        values = precision[threshold_index, :, :, 0, 2]
        threshold_ap[f"AP{int(round(float(threshold) * 100)):02d}"] = (
            100.0 * _mean_valid(values)
        )

    metrics = {
        "AP": 100.0 * float(evaluator.stats[0]),
        "AP50": 100.0 * float(evaluator.stats[1]),
        "AP75": 100.0 * float(evaluator.stats[2]),
        "AR100": 100.0 * float(evaluator.stats[8]),
        "threshold_ap": threshold_ap,
        "image_count": int(len(ground_truth.imgs)),
        "detection_count": int(len(detections)),
    }
    return metrics


def write_detections(path: Path, detections: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
