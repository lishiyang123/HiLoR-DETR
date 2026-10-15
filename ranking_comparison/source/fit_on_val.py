#!/usr/bin/env python3
"""Fit the Gilg-original conditional Beta map on a frozen Val capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nearest_neighbor_baselines.candidate_adapter import candidates_from_record
from nearest_neighbor_baselines.capture_io import load_capture, record_ground_truth, sha256
from nearest_neighbor_baselines.gilg_iou_aware import (
    ConditionalBetaCalibrator,
    greedy_tp_labels,
    minimum_jaccard_to_higher,
    official_near_duplicate_keep,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--label-iou", type=float, default=0.5)
    parser.add_argument("--near-duplicate-iou", type=float, default=0.99)
    parser.add_argument(
        "--keep-complete-pool",
        action="store_true",
        help="Disable the reference implementation's IoU=0.99 cleanup.",
    )
    args = parser.parse_args()

    capture = load_capture(args.capture)
    if capture.get("split") != "val":
        raise RuntimeError("Gilg parameters must be fitted from a Val capture")

    all_inputs = []
    all_targets = []
    image_summaries = []
    prefilter = None if args.keep_complete_pool else args.near_duplicate_iou
    started = time.perf_counter()
    for record in capture["records"]:
        candidates = candidates_from_record(record)
        gt_boxes, gt_labels = record_ground_truth(record)
        jaccard_min = minimum_jaccard_to_higher(candidates)
        active = official_near_duplicate_keep(candidates, prefilter)
        targets = greedy_tp_labels(
            candidates,
            gt_boxes,
            gt_labels,
            iou_threshold=args.label_iou,
            active_mask=active,
        )
        all_inputs.append(np.column_stack((candidates.scores[active], jaccard_min[active])))
        all_targets.append(targets[active])
        image_summaries.append(
            {
                "image_id": int(candidates.image_id),
                "candidate_count": int(len(candidates.scores)),
                "active_count": int(np.sum(active)),
                "positive_count": int(np.sum(targets[active])),
            }
        )

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    calibrator = ConditionalBetaCalibrator().fit(
        inputs, targets, seed=args.seed, maxiter=args.maxiter
    )
    calibrator.fit_summary.update(
        {
            "capture": str(args.capture.resolve()),
            "capture_sha256": sha256(args.capture),
            "split": "val",
            "label_iou": float(args.label_iou),
            "near_duplicate_iou": prefilter,
            "elapsed_seconds": float(time.perf_counter() - started),
            "image_count": int(len(image_summaries)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.output)
    audit_path = args.output.with_suffix(".fit_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "model": calibrator.to_dict(),
                "images": image_summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("GILG VAL FIT: PASS")
    print(json.dumps(calibrator.fit_summary, indent=2))


if __name__ == "__main__":
    main()
