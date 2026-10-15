#!/usr/bin/env python3
"""Replay a frozen Test capture through one baseline and run COCO evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np

from nearest_neighbor_baselines.candidate_adapter import candidates_from_record
from nearest_neighbor_baselines.capture_io import load_capture, sha256
from nearest_neighbor_baselines.coco_evaluator import (
    evaluate_coco_detections,
    write_detections,
)
from nearest_neighbor_baselines.gilg_iou_aware import (
    ConditionalBetaCalibrator,
    gilg_scores,
)
from nearest_neighbor_baselines.formal_replay import FormalPostprocessorReplay
from nearest_neighbor_baselines.iou_regression import IoURegressorReplay
from nearest_neighbor_baselines.soft_nms import soft_nms_scores


def percentile(values: list[float], percentage: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentage))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--method",
        choices=(
            "native", "soft_nms_linear", "soft_nms_gaussian",
            "gilg_original", "iou_reg_rescore", "qccr", "qccr_umr",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--gilg-params", type=Path)
    parser.add_argument("--near-duplicate-iou", type=float, default=0.99)
    parser.add_argument("--keep-complete-pool", action="store_true")
    parser.add_argument("--formal-source", type=Path)
    parser.add_argument("--qccr-params", type=Path)
    parser.add_argument("--umr-checkpoint", type=Path)
    parser.add_argument("--umr-metadata", type=Path)
    parser.add_argument("--formal-replay-device", default="cpu")
    parser.add_argument("--iou-reg-checkpoint", type=Path)
    parser.add_argument("--iou-reg-metadata", type=Path)
    parser.add_argument("--iou-reg-device", default="cpu")
    args = parser.parse_args()

    capture = load_capture(args.capture)
    if capture.get("split") != args.expected_split:
        raise RuntimeError(
            f"expected a {args.expected_split!r} capture, got {capture.get('split')!r}"
        )
    calibrator = None
    if args.method == "gilg_original":
        if args.gilg_params is None:
            raise RuntimeError("--gilg-params is required for gilg_original")
        calibrator = ConditionalBetaCalibrator.load(args.gilg_params)
    formal_replayer = None
    if args.method in {"qccr", "qccr_umr"}:
        if args.formal_source is None or args.qccr_params is None:
            raise RuntimeError(
                "--formal-source and --qccr-params are required for formal replay"
            )
        formal_replayer = FormalPostprocessorReplay(
            method=args.method,
            formal_source=args.formal_source,
            qccr_params=args.qccr_params,
            capture_checkpoint_sha256=capture.get("checkpoint_sha256"),
            umr_checkpoint=args.umr_checkpoint,
            umr_metadata=args.umr_metadata,
            device=args.formal_replay_device,
        )
    iou_replayer = None
    if args.method == "iou_reg_rescore":
        if args.iou_reg_checkpoint is None or args.iou_reg_metadata is None:
            raise RuntimeError(
                "--iou-reg-checkpoint and --iou-reg-metadata are required"
            )
        iou_replayer = IoURegressorReplay(
            checkpoint=args.iou_reg_checkpoint,
            metadata=args.iou_reg_metadata,
            capture_checkpoint_sha256=capture.get("checkpoint_sha256"),
            device=args.iou_reg_device,
        )

    detections = []
    timings_ms = []
    prefilter = None if args.keep_complete_pool else args.near_duplicate_iou
    for record in capture["records"]:
        candidates = candidates_from_record(record)
        if formal_replayer is not None:
            formal_replayer.synchronize()
        if iou_replayer is not None:
            iou_replayer.synchronize()
        started = time.perf_counter_ns()
        active = np.ones(len(candidates.scores), dtype=bool)
        if args.method == "native":
            scores = candidates.scores
        elif args.method == "soft_nms_linear":
            scores = soft_nms_scores(
                candidates,
                method="linear",
                iou_threshold=args.iou_threshold,
                sigma=args.sigma,
            )
        elif args.method == "soft_nms_gaussian":
            scores = soft_nms_scores(
                candidates,
                method="gaussian",
                iou_threshold=args.iou_threshold,
                sigma=args.sigma,
            )
        elif args.method == "gilg_original":
            scores, active, _ = gilg_scores(
                candidates, calibrator, near_duplicate_iou=prefilter
            )
        elif formal_replayer is not None:
            scores = formal_replayer.scores(record)
        else:
            scores = iou_replayer.scores(record)
        if formal_replayer is not None:
            formal_replayer.synchronize()
        if iou_replayer is not None:
            iou_replayer.synchronize()
        selected = candidates.topk(scores=scores, k=args.topk, active_mask=active)
        timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        detections.extend(candidates.to_coco(selected, scores=scores))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detection_path = args.output_dir / f"{args.method}_{args.expected_split}_detections.json"
    write_detections(detection_path, detections)
    metrics = evaluate_coco_detections(args.annotation, detections)
    summary = {
        "method": args.method,
        "split": args.expected_split,
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256(args.capture),
        "annotation": str(args.annotation.resolve()),
        "topk": int(args.topk),
        "parameters": {
            "iou_threshold": float(args.iou_threshold),
            "sigma": float(args.sigma),
            "near_duplicate_iou": prefilter,
            "gilg_params": None if args.gilg_params is None else str(args.gilg_params.resolve()),
            "formal_source": None if args.formal_source is None else str(args.formal_source.resolve()),
            "qccr_params": None if args.qccr_params is None else str(args.qccr_params.resolve()),
            "umr_checkpoint": None if args.umr_checkpoint is None else str(args.umr_checkpoint.resolve()),
            "umr_metadata": None if args.umr_metadata is None else str(args.umr_metadata.resolve()),
            "formal_replay_device": args.formal_replay_device,
            "iou_reg_checkpoint": None if args.iou_reg_checkpoint is None else str(args.iou_reg_checkpoint.resolve()),
            "iou_reg_metadata": None if args.iou_reg_metadata is None else str(args.iou_reg_metadata.resolve()),
            "iou_reg_device": args.iou_reg_device,
        },
        "formal_asset_binding": (
            None if formal_replayer is None else formal_replayer.binding.as_dict()
        ),
        "iou_regressor_binding": (
            None if iou_replayer is None else iou_replayer.binding()
        ),
        "cpu_replay_latency_ms": {
            "mean": float(np.mean(timings_ms)),
            "median": float(median(timings_ms)),
            "p95": percentile(timings_ms, 95.0),
            "note": "CPU capture-replay timing; not the formal GPU deployment latency.",
        },
        "metrics": metrics,
    }
    summary_path = args.output_dir / f"{args.method}_{args.expected_split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.method.upper()} {args.expected_split.upper()} EVALUATION: PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
