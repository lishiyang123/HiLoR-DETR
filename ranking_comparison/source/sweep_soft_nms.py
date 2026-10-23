#!/usr/bin/env python3
"""Evaluate a disclosed Soft-NMS grid on one frozen capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nearest_neighbor_baselines.candidate_adapter import candidates_from_record
from nearest_neighbor_baselines.capture_io import load_capture, sha256
from nearest_neighbor_baselines.coco_evaluator import evaluate_coco_detections
from nearest_neighbor_baselines.soft_nms import soft_nms_scores


def parse_values(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one grid value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--method", choices=("linear", "gaussian"), required=True)
    parser.add_argument("--values", type=parse_values, required=True)
    parser.add_argument("--fixed-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-sigma", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = load_capture(args.capture)
    if capture.get("split") != args.split:
        raise RuntimeError(
            f"expected split {args.split!r}, got {capture.get('split')!r}"
        )

    adapted = [candidates_from_record(record) for record in capture["records"]]
    configurations = []
    for value in args.values:
        iou_threshold = value if args.method == "linear" else args.fixed_iou_threshold
        sigma = value if args.method == "gaussian" else args.fixed_sigma
        detections = []
        for candidates in adapted:
            scores = soft_nms_scores(
                candidates,
                method=args.method,
                iou_threshold=iou_threshold,
                sigma=sigma,
            )
            selected = candidates.topk(scores=scores, k=args.topk)
            detections.extend(candidates.to_coco(selected, scores=scores))
        metrics = evaluate_coco_detections(args.annotation, detections)
        configurations.append(
            {
                "grid_value": float(value),
                "iou_threshold": float(iou_threshold),
                "sigma": float(sigma),
                "metrics": metrics,
            }
        )
        print(
            f"{args.method} value={value:g}: "
            f"AP={metrics['AP']:.6f} AP75={metrics['AP75']:.6f}"
        )

    ranking = sorted(
        range(len(configurations)),
        key=lambda index: (
            -configurations[index]["metrics"]["AP"],
            -configurations[index]["metrics"]["AP75"],
            configurations[index]["grid_value"],
        ),
    )
    payload = {
        "format_version": 1,
        "method": f"soft_nms_{args.method}",
        "split": args.split,
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256(args.capture),
        "annotation": str(args.annotation.resolve()),
        "topk": int(args.topk),
        "selection_rule": "highest AP, then AP75, then smaller grid value",
        "selected_index": int(ranking[0]),
        "selected": configurations[ranking[0]],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"SOFT-NMS {args.method.upper()} GRID: PASS")


if __name__ == "__main__":
    main()
