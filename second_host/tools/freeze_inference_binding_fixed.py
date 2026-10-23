#!/usr/bin/env python3
"""Guard the fixed QCCR--UMR tuple, then delegate to the locked binder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from fit_common import atomic_json, require_output_path, require_sha
    from fixed_protocol_common import (
        LOCK_RELATIVE,
        POLICY_NAME,
        QCCR_LAMBDA_REG,
        UMR_BETA,
        UMR_HEAD_SEED,
        UMR_POSITIVE_WEIGHT,
        UMR_TAU,
        UMR_TOPK,
        fixed_lock_record,
        load_fixed_protocol,
    )
    from freeze_inference_binding import main as legacy_binding_main
    from train_umr_fixed import validate_fixed_qccr
except ImportError:  # imported as tools.freeze_inference_binding_fixed in tests
    from tools.fit_common import atomic_json, require_output_path, require_sha
    from tools.fixed_protocol_common import (
        LOCK_RELATIVE,
        POLICY_NAME,
        QCCR_LAMBDA_REG,
        UMR_BETA,
        UMR_HEAD_SEED,
        UMR_POSITIVE_WEIGHT,
        UMR_TAU,
        UMR_TOPK,
        fixed_lock_record,
        load_fixed_protocol,
    )
    from tools.freeze_inference_binding import main as legacy_binding_main
    from tools.train_umr_fixed import validate_fixed_qccr


def validate_fixed_umr_decision(
    selection: dict,
    *,
    detector_seed: int,
    checkpoint_sha: str,
    train_capture_sha: str,
    val_capture_sha: str,
    data_lock_sha: str,
    qccr_sha: str,
    runtime_hashes: dict[str, str],
    umr_sha: str,
    implementation_sha: str,
    fixed_lock_sha: str,
) -> None:
    policy = selection.get("selection_policy") or {}
    lock = selection.get("fixed_protocol_lock") or {}
    fixed_tuple = policy.get("fixed_tuple")
    expected_tuple = {
        "positive_weight": UMR_POSITIVE_WEIGHT,
        "head_seed": UMR_HEAD_SEED,
        "tau": UMR_TAU,
        "beta": UMR_BETA,
        "topk": UMR_TOPK,
    }
    if (
        selection.get("schema") != "rtdetr-hilor-umr-val-selection-v1"
        or selection.get("decision") != "VAL_GO"
        or selection.get("test_accessed") is not False
        or selection.get("official_test_accessed") is not False
        or selection.get("adaptive_candidate_evidence_used") is not False
        or selection.get("val_metrics_computed") is not False
        or selection.get("compatibility_fields_are_not_observed_gates") is not True
        or policy.get("name") != POLICY_NAME
        or policy.get("decision") != "FIXED_BEFORE_TEST"
        or policy.get("adaptive_selection_performed") is not False
        or policy.get("adaptive_candidate_evidence_used") is not False
        or policy.get("val_grid_read") is not False
        or policy.get("val_metrics_computed") is not False
        or policy.get("compatibility_fields_are_not_observed_gates") is not True
        or fixed_tuple != expected_tuple
        or lock.get("path") != LOCK_RELATIVE.as_posix()
        or lock.get("sha256") != fixed_lock_sha
    ):
        raise RuntimeError("UMR decision is not the fixed pre-Test decision")
    forbidden = {
        "grids",
        "baseline_qccr_val",
        "top5_by_val_score",
        "eligible_candidate_count",
        "candidate_count_including_beta0",
    }
    if forbidden.intersection(selection):
        raise RuntimeError("fixed UMR decision contains adaptive Val-grid evidence")
    expected = {
        "detector_seed": detector_seed,
        "source_checkpoint_sha256": checkpoint_sha,
        "train_capture_sha256": train_capture_sha,
        "val_capture_sha256": val_capture_sha,
        "source_data_split_lock_sha256": data_lock_sha,
        "qccr_artifact_sha256": qccr_sha,
        "qccr_source_capture_sha256": val_capture_sha,
        "runtime_source_sha256": runtime_hashes,
        "implementation_lock_sha256": implementation_sha,
    }
    for name, value in expected.items():
        if selection.get(name) != value:
            raise RuntimeError(f"fixed UMR provenance mismatch: {name}")
    selected = selection.get("selected") or {}
    for name, value in expected_tuple.items():
        if selected.get(name) != value:
            raise RuntimeError(f"fixed UMR tuple mismatch: {name}")
    if selected.get("pw") != UMR_POSITIVE_WEIGHT:
        raise RuntimeError("fixed UMR tuple mismatch: pw")
    if (selection.get("selected_model") or {}).get("sha256") != umr_sha:
        raise RuntimeError("fixed UMR decision and checkpoint SHA disagree")


def mark_fixed_binding(binding: dict, fixed_lock: dict[str, str]) -> dict:
    """Add visible fixed-policy provenance without changing runtime semantics."""

    binding = dict(binding)
    binding["fixed_protocol_lock"] = dict(fixed_lock)
    binding["adaptive_candidate_evidence_used"] = False
    binding["val_metrics_computed"] = False
    binding["compatibility_fields_are_not_observed_gates"] = True
    binding["selection_policy"] = {
        "name": POLICY_NAME,
        "decision": "FIXED_BEFORE_TEST",
        "adaptive_selection_performed": False,
        "adaptive_candidate_evidence_used": False,
        "val_grid_read": False,
        "val_metrics_computed": False,
        "compatibility_fields_are_not_observed_gates": True,
        "fixed_tuple": {
            "qccr_lambda_reg": QCCR_LAMBDA_REG,
            "umr_positive_weight": UMR_POSITIVE_WEIGHT,
            "umr_head_seed": UMR_HEAD_SEED,
            "umr_tau": UMR_TAU,
            "umr_beta": UMR_BETA,
            "umr_topk": UMR_TOPK,
        },
    }
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-protocol-lock", type=Path, required=True)
    parser.add_argument("--expected-fixed-protocol-lock-sha", required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-detector-sha", required=True)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--expected-train-capture-sha", required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--qccr-artifact", type=Path, required=True)
    parser.add_argument("--expected-qccr-sha", required=True)
    parser.add_argument("--umr-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-umr-sha", required=True)
    parser.add_argument("--umr-selection", type=Path, required=True)
    parser.add_argument("--expected-umr-selection-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = (
        Path(__file__).resolve().parents[1]
        if args.adapter_root is None
        else args.adapter_root.resolve()
    )
    policy, lock_sha, implementation_sha, data_lock_sha = load_fixed_protocol(
        args.fixed_protocol_lock,
        args.expected_fixed_protocol_lock_sha,
        adapter_root_path=root,
        tool_relative="tools/freeze_inference_binding_fixed.py",
        required_locked_tools=("tools/freeze_inference_binding.py",),
    )
    checkpoint = args.detector_checkpoint.resolve()
    checkpoint_sha = require_sha(
        checkpoint, args.expected_detector_sha, "detector checkpoint"
    )
    qccr_path = require_output_path(
        args.qccr_artifact,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed QCCR input",
    )
    qccr_sha = require_sha(qccr_path, args.expected_qccr_sha, "fixed QCCR artifact")
    qccr = json.loads(qccr_path.read_text(encoding="utf-8"))
    validate_fixed_qccr(
        qccr,
        detector_seed=args.detector_seed,
        checkpoint_sha=checkpoint_sha,
        val_capture_sha=args.expected_val_capture_sha,
        data_lock_sha=data_lock_sha,
        implementation_sha=implementation_sha,
        fixed_lock_sha=lock_sha,
        qccr_runtime_sha=policy["fit_runtime_sources"]["hilor_qccr.py"],
    )
    selection_path = require_output_path(
        args.umr_selection,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed UMR decision input",
    )
    require_sha(
        selection_path,
        args.expected_umr_selection_sha,
        "fixed UMR decision",
    )
    umr_path = require_output_path(
        args.umr_checkpoint,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed UMR checkpoint input",
    )
    umr_sha = require_sha(umr_path, args.expected_umr_sha, "fixed UMR checkpoint")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_fixed_umr_decision(
        selection,
        detector_seed=args.detector_seed,
        checkpoint_sha=checkpoint_sha,
        train_capture_sha=args.expected_train_capture_sha,
        val_capture_sha=args.expected_val_capture_sha,
        data_lock_sha=data_lock_sha,
        qccr_sha=qccr_sha,
        runtime_hashes=policy["fit_runtime_sources"],
        umr_sha=umr_sha,
        implementation_sha=implementation_sha,
        fixed_lock_sha=lock_sha,
    )

    output = require_output_path(
        args.output,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed inference binding",
    )
    temporary_output = output.with_name(output.name + ".legacy-binder.tmp")
    if output.exists() or temporary_output.exists():
        raise RuntimeError("refusing to overwrite fixed inference binding evidence")
    forwarded = [
        "freeze_inference_binding.py",
        "--adapter-root", str(root),
        "--source-root", str(args.source_root),
        "--mode", "qccr-umr",
        "--detector-seed", str(args.detector_seed),
        "--detector-checkpoint", str(checkpoint),
        "--expected-detector-sha", args.expected_detector_sha,
        "--train-capture", str(args.train_capture),
        "--expected-train-capture-sha", args.expected_train_capture_sha,
        "--val-capture", str(args.val_capture),
        "--expected-val-capture-sha", args.expected_val_capture_sha,
        "--qccr-artifact", str(qccr_path),
        "--expected-qccr-sha", args.expected_qccr_sha,
        "--umr-checkpoint", str(umr_path),
        "--expected-umr-sha", args.expected_umr_sha,
        "--umr-selection", str(selection_path),
        "--expected-umr-selection-sha", args.expected_umr_selection_sha,
        "--output", str(temporary_output),
    ]
    original = sys.argv
    try:
        sys.argv = forwarded
        legacy_binding_main()
        legacy = json.loads(temporary_output.read_text(encoding="utf-8"))
        atomic_json(
            output,
            mark_fixed_binding(legacy, fixed_lock_record(root, lock_sha)),
        )
    finally:
        sys.argv = original
        if temporary_output.exists():
            temporary_output.unlink()
    print("FIXED QCCR--UMR POLICY GUARD: PASS")


if __name__ == "__main__":
    main()
