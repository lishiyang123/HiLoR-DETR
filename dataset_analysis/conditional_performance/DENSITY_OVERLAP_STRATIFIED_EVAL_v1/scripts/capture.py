#!/usr/bin/env python3
"""Capture one formal prediction list through the official test-only path."""

from __future__ import annotations

import argparse
import math
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from common import file_record, require, sha256, write_json


def load_eval(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "eval" in payload and "precision" not in payload:
        payload = payload["eval"]
    require(isinstance(payload, dict), "Unrecognized eval payload: %s" % path)
    require({"precision", "recall", "params"} <= set(payload), "Incomplete eval payload: %s" % path)
    return payload


def exact_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    precision = np.asarray(payload["precision"], dtype=np.float64)
    recall = np.asarray(payload["recall"], dtype=np.float64)
    params = payload["params"]
    get = (lambda key: params[key]) if isinstance(params, dict) else (lambda key: getattr(params, key))
    ious = np.asarray(get("iouThrs"), dtype=np.float64)
    areas = list(get("areaRngLbl"))
    max_dets = [int(value) for value in get("maxDets")]

    def valid_mean(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    def threshold(value):
        found = np.flatnonzero(np.isclose(ious, value, rtol=0.0, atol=1.0e-12))
        require(len(found) == 1, "Missing IoU threshold: %s" % value)
        return int(found[0])

    max_index = max_dets.index(100)

    def ap(iou=None, area="all"):
        area_index = areas.index(area)
        values = precision[:, :, :, area_index, max_index]
        if iou is not None:
            values = precision[threshold(iou), :, :, area_index, max_index]
        return valid_mean(values)

    def ar(area="all", maximum=100):
        return valid_mean(recall[:, :, areas.index(area), max_dets.index(maximum)])

    return {
        "AP": ap(), "AP50": ap(0.50), "AP75": ap(0.75),
        "APs": ap(area="small"), "APm": ap(area="medium"), "APl": ap(area="large"),
        "AR1": ar(maximum=1), "AR10": ar(maximum=10), "AR100": ar(maximum=100),
        "ARs": ar(area="small"), "ARm": ar(area="medium"), "ARl": ar(area="large"),
        "AP80": ap(0.80), "AP85": ap(0.85), "AP90": ap(0.90), "AP95": ap(0.95),
    }


def same_metric(left: float, right: float, tolerance: float = 2.0e-9) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("B0", "Full"), required=True)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--formal-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(torch.cuda.is_available(), "CUDA GPU is required for official-path prediction capture")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Capture is locked to one-process evaluation")
    code_root = args.code_root.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    formal_eval = args.formal_eval.resolve()
    output = args.output_dir.resolve()
    for path in (code_root, config, checkpoint, formal_eval):
        require(path.exists(), "Missing capture input: %s" % path)
    observed_sha = sha256(checkpoint)
    require(observed_sha == args.expected_checkpoint_sha, "Checkpoint SHA mismatch")
    require(not output.exists() or not any(output.iterdir()), "Refusing nonempty output directory: %s" % output)
    output.mkdir(parents=True, exist_ok=True)

    os.chdir(str(code_root))
    sys.path.insert(0, str(code_root))
    from engine.core import YAMLConfig
    from engine.data import CocoEvaluator
    from train import main as official_main

    resolved = YAMLConfig(str(config)).yaml_cfg
    post = resolved.get("PostProcessor", {})
    observed_switches = (bool(post.get("qccr_enabled", False)), bool(post.get("umr_enabled", False)))
    expected_switches = (args.method == "Full", args.method == "Full")
    require(
        observed_switches == expected_switches,
        "%s switch mismatch: observed=%r expected=%r" % (args.method, observed_switches, expected_switches),
    )

    captured = {}
    original_update = CocoEvaluator.update

    def capture_after_official_update(evaluator, predictions):
        result = original_update(evaluator, predictions)
        for image_id, prediction in predictions.items():
            captured[int(image_id)] = {
                key: value.detach().cpu() if hasattr(value, "detach") else value
                for key, value in prediction.items()
            }
        return result

    CocoEvaluator.update = capture_after_official_update
    try:
        official_main(Namespace(
            config=str(config), resume=str(checkpoint), tuning=None,
            device=None, seed=args.detector_seed, use_amp=False,
            output_dir=str(output), summary_dir=None, test_only=True,
            update=None, print_method="builtin", print_rank=0, local_rank=None,
        ))
    finally:
        CocoEvaluator.update = original_update

    replay_eval = output / "eval.pth"
    require(replay_eval.is_file(), "Official test-only path did not produce eval.pth")
    replay_metrics = exact_metrics(load_eval(replay_eval))
    formal_metrics = exact_metrics(load_eval(formal_eval))
    mismatches = []
    for name, expected in formal_metrics.items():
        observed = replay_metrics[name]
        if not same_metric(observed, expected):
            mismatches.append((name, observed, expected))
    require(not mismatches, "Formal full-Test metric replay mismatch: %r" % mismatches[:6])

    detections = []
    for image_id in sorted(captured):
        prediction = captured[image_id]
        for label, box, score in zip(
            prediction["labels"].tolist(), prediction["boxes"].tolist(), prediction["scores"].tolist()
        ):
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    detections.sort(key=lambda row: (row["image_id"], -row["score"], row["category_id"]))
    image_ids = sorted(captured)
    predictions_path = output / "predictions.json"
    write_json(predictions_path, {
        "format_version": 1,
        "dataset": "CDW1630 official Test",
        "method": args.method,
        "detector_seed": args.detector_seed,
        "capture_policy": "after CocoEvaluator.update on official train.py test-only path",
        "checkpoint_sha256": observed_sha,
        "config_sha256": sha256(config),
        "formal_eval_sha256": sha256(formal_eval),
        "image_count": len(image_ids),
        "detection_count": len(detections),
        "image_ids": image_ids,
        "detections": detections,
        "formal_full_test_metric_replay": "PASS",
        "test_used_for_selection": False,
    })
    write_json(output / "PREDICTION_SUMMARY.json", {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "detector_seed": args.detector_seed,
        "checkpoint": file_record(checkpoint),
        "config": file_record(config),
        "formal_eval": file_record(formal_eval),
        "replay_eval": file_record(replay_eval),
        "predictions": file_record(predictions_path),
        "image_count": len(image_ids),
        "detection_count": len(detections),
        "full_test_exact_metrics": replay_metrics,
        "formal_full_test_metric_replay": "PASS",
        "test_used_for_selection": False,
    })
    print("%s seed%d OFFICIAL-PATH PREDICTION CAPTURE: PASS" % (args.method, args.detector_seed))


if __name__ == "__main__":
    main()
