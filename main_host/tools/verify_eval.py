#!/usr/bin/env python3
"""Verify exact COCO metrics against a frozen JSON reference."""

from __future__ import annotations

import argparse
import json
import math
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--label", default="evaluation")
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = reference["metrics"]
    current = extract(args.eval)
    errors: dict[str, float | None] = {}
    checks: dict[str, bool] = {}

    print(f"===== {args.label} EXACT METRICS =====")
    for name in ORDER:
        actual = current[name]
        target = expected[name]
        if target is None:
            passed = not math.isfinite(actual)
            errors[name] = None
            print(f"{name:5s}: N/A expected=N/A {'PASS' if passed else 'FAIL'}")
        else:
            error = abs(actual - float(target))
            passed = error <= args.tolerance
            errors[name] = error
            print(
                f"{name:5s}: {actual:.9f} error={error:.3e} "
                f"{'PASS' if passed else 'FAIL'}"
            )
        checks[name] = passed

    passed = all(checks.values())
    report = {
        "passed": passed,
        "label": args.label,
        "eval": str(args.eval.resolve()),
        "reference": str(args.reference.resolve()),
        "tolerance": args.tolerance,
        "metrics": {
            name: value if math.isfinite(value) else None
            for name, value in current.items()
        },
        "absolute_errors": errors,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{args.label} EXACT VERIFICATION:", "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
