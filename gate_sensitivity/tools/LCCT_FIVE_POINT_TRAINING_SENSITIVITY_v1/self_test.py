#!/usr/bin/env python3
"""Dependency-light integrity tests for the LCCT queue manager."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from manager import CENTER, GRID, QUEUE, delta_label, parse_training_log, sha256


ROOT = Path(__file__).resolve().parent


def main() -> None:
    for name in ("manager.py", "self_test.py"):
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    if GRID != [0.005, 0.01, 0.02, 0.04, 0.08] or CENTER != 0.02:
        raise RuntimeError("Frozen five-point grid/center mismatch")
    if QUEUE != [0.01, 0.04, 0.005, 0.08] or CENTER in QUEUE:
        raise RuntimeError("Four-run queue/order mismatch")
    if len({delta_label(value) for value in GRID}) != 5:
        raise RuntimeError("Delta directory labels are not unique")

    protocol_path = ROOT / "protocol/TRAINING_GRID_AMENDMENT_v1_2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_FIVE_POINT_MEASUREMENT":
        raise RuntimeError("Training-grid amendment is not frozen")
    if protocol["grid"] != GRID or protocol["new_training_values"] != QUEUE:
        raise RuntimeError("Training-grid amendment values mismatch")
    if protocol["test_used_for_parameter_selection"] is not False:
        raise RuntimeError("Test-selection guard is absent")

    source = (ROOT / "manager.py").read_text(encoding="utf-8")
    required = (
        'MAIN_NAME = "main_host"',
        'TARGET_NAME = "gate_sensitivity"',
        '"test_accessed": False',
        'start_new_session=True',
        'archive_partial(target, delta)',
        'compact_completed_run(run_dir, delta)',
        'full_dataset_hash=True',
        '"CUDA_VISIBLE_DEVICES": "0"',
        '"OMP_NUM_THREADS": "1"',
        '"MKL_NUM_THREADS": "1"',
        "import faster_coco_eval",
    )
    for token in required:
        if token not in source:
            raise RuntimeError("Missing safety mechanism: " + token)
    for forbidden in ('project / "data/CDW1630', "diagnostics/CDW_LCCT_BOUNDED", "--test-only",
                      "import pycocotools"):
        if forbidden in source:
            raise RuntimeError("Non-final or Test workflow leaked into manager: " + forbidden)

    with tempfile.TemporaryDirectory() as temp:
        log = Path(temp) / "log.txt"
        rows = []
        for epoch in range(102):
            ap = 0.5 + epoch / 1000
            rows.append(json.dumps({"epoch": epoch, "test_coco_eval_bbox": [ap, 0.9, 0.8, 0, 0, 0, 0, 0, 0.85]}))
        log.write_text("\n".join(rows) + "\n", encoding="utf-8")
        parsed = parse_training_log(log, require_complete=True)
        if parsed["row_count"] != 102 or parsed["best_epoch"] != 101 or parsed["selected_stage"] != "stage2":
            raise RuntimeError("Training-log completion/selection logic mismatch")
        if sha256(log) != sha256(log):
            raise RuntimeError("SHA function is unstable")

    print("LCCT FIVE-POINT TRAINING SENSITIVITY SELF-TEST: PASS")


if __name__ == "__main__":
    main()
