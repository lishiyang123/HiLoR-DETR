#!/usr/bin/env python3
"""Run every disclosed frozen-output comparison and aggregate the results."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--linear-iou-threshold", type=float, required=True)
    parser.add_argument("--gaussian-sigma", type=float, required=True)
    parser.add_argument("--gilg-reference-params", type=Path, required=True)
    parser.add_argument("--gilg-complete-pool-params", type=Path, required=True)
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--qccr-params", type=Path, required=True)
    parser.add_argument("--umr-checkpoint", type=Path, required=True)
    parser.add_argument("--umr-metadata", type=Path, required=True)
    parser.add_argument("--formal-replay-device", default="cuda")
    parser.add_argument("--iou-reg-checkpoint", type=Path, required=True)
    parser.add_argument("--iou-reg-metadata", type=Path, required=True)
    parser.add_argument("--iou-reg-device", default="cuda")
    args = parser.parse_args()

    evaluator = Path(__file__).resolve().with_name("evaluate_on_test.py")
    aggregator = Path(__file__).resolve().with_name("aggregate_results.py")
    common = [
        sys.executable, "-B", str(evaluator),
        "--capture", str(args.capture.resolve()),
        "--annotation", str(args.annotation.resolve()),
        "--expected-split", args.split,
    ]
    jobs = [
        ("native", "native", []),
        (
            "soft_nms_linear",
            "soft_nms_linear",
            ["--iou-threshold", str(args.linear_iou_threshold)],
        ),
        (
            "soft_nms_gaussian",
            "soft_nms_gaussian",
            ["--sigma", str(args.gaussian_sigma)],
        ),
        (
            "gilg_reference",
            "gilg_original",
            ["--gilg-params", str(args.gilg_reference_params.resolve())],
        ),
        (
            "gilg_complete_pool",
            "gilg_original",
            [
                "--gilg-params", str(args.gilg_complete_pool_params.resolve()),
                "--keep-complete-pool",
            ],
        ),
        (
            "qccr",
            "qccr",
            [
                "--formal-source", str(args.formal_source.resolve()),
                "--qccr-params", str(args.qccr_params.resolve()),
                "--formal-replay-device", args.formal_replay_device,
            ],
        ),
        (
            "qccr_umr",
            "qccr_umr",
            [
                "--formal-source", str(args.formal_source.resolve()),
                "--qccr-params", str(args.qccr_params.resolve()),
                "--umr-checkpoint", str(args.umr_checkpoint.resolve()),
                "--umr-metadata", str(args.umr_metadata.resolve()),
                "--formal-replay-device", args.formal_replay_device,
            ],
        ),
        (
            "iou_reg_rescore",
            "iou_reg_rescore",
            [
                "--iou-reg-checkpoint", str(args.iou_reg_checkpoint.resolve()),
                "--iou-reg-metadata", str(args.iou_reg_metadata.resolve()),
                "--iou-reg-device", args.iou_reg_device,
            ],
        ),
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for directory, method, extras in jobs:
        command = common + [
            "--method", method,
            "--output-dir", str((args.output_root / directory).resolve()),
        ] + extras
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    subprocess.run(
        [
            sys.executable, "-B", str(aggregator),
            "--input-root", str(args.output_root.resolve()),
            "--output-dir", str((args.output_root / "aggregate").resolve()),
        ],
        check=True,
    )
    print("FROZEN COMPARISON: PASS / 8 disclosed replay paths")


if __name__ == "__main__":
    main()
