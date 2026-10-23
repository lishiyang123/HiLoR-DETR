#!/usr/bin/env python3
"""Fit one predeclared UMR head on Train; perform no Val grid or gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from fit_common import (
        EXPECTED_TOPK,
        atomic_json,
        atomic_torch_save,
        formal_fit_guard,
        install_source_root,
        load_capture,
        require_output_path,
        require_sha,
        runtime_source_hashes,
        runtime_source_path,
        sha256_file,
    )
    from fixed_protocol_common import (
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
    from train_umr_select_val import FEATURE_SCHEMA, build_train_data, train_one
except ImportError:  # imported as tools.train_umr_fixed in unit tests
    from tools.fit_common import (
        EXPECTED_TOPK,
        atomic_json,
        atomic_torch_save,
        formal_fit_guard,
        install_source_root,
        load_capture,
        require_output_path,
        require_sha,
        runtime_source_hashes,
        runtime_source_path,
        sha256_file,
    )
    from tools.fixed_protocol_common import (
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
    from tools.train_umr_select_val import (
        FEATURE_SCHEMA,
        build_train_data,
        train_one,
    )


COMPATIBILITY_SCHEMA = "rtdetr-hilor-umr-val-selection-v1"
COMPATIBILITY_PROTOCOL = "CDW1630 Train candidates / Val-only selection"


def validate_fixed_qccr(
    artifact: dict,
    *,
    detector_seed: int,
    checkpoint_sha: str,
    val_capture_sha: str,
    data_lock_sha: str,
    implementation_sha: str,
    fixed_lock_sha: str,
    qccr_runtime_sha: str,
) -> dict[str, float]:
    """Validate the compatibility artifact plus its fixed-policy evidence."""

    policy = artifact.get("selection_policy") or {}
    lock = artifact.get("fixed_protocol_lock") or {}
    if (
        artifact.get("schema") != "rtdetr-hilor-qccr-frozen-artifact-v1"
        or artifact.get("method") != "QCCR-v1"
        or artifact.get("dataset") != "CDW1630"
        or artifact.get("selection_decision") != "OOF_VAL_GO"
        or artifact.get("test_accessed") is not False
        or artifact.get("test_used_for_selection") is not False
        or artifact.get("adaptive_candidate_evidence_used") is not False
        or artifact.get("val_metrics_computed") is not False
        or artifact.get("compatibility_fields_are_not_observed_gates") is not True
        or policy.get("name") != POLICY_NAME
        or policy.get("decision") != "FIXED_BEFORE_TEST"
        or policy.get("adaptive_selection_performed") is not False
        or policy.get("adaptive_candidate_evidence_used") is not False
        or policy.get("oof_gate_read") is not False
        or policy.get("val_metrics_computed") is not False
        or policy.get("compatibility_fields_are_not_observed_gates") is not True
        or lock.get("sha256") != fixed_lock_sha
        or float(artifact.get("selected_lambda_reg", -1.0)) != QCCR_LAMBDA_REG
    ):
        raise RuntimeError("QCCR artifact is not the fixed pre-Test artifact")
    forbidden_adaptive_evidence = {
        "oof_selection_sha256",
        "val_baseline_metrics",
        "selected_oof_metrics",
        "candidate_decisions",
    }
    if forbidden_adaptive_evidence.intersection(artifact):
        raise RuntimeError("fixed QCCR artifact contains adaptive OOF evidence")
    expected = {
        "detector_seed": detector_seed,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_capture_sha256": val_capture_sha,
        "source_data_split_lock_sha256": data_lock_sha,
        "implementation_lock_sha256": implementation_sha,
        "runtime_source_sha256": {"hilor_qccr.py": qccr_runtime_sha},
    }
    for name, value in expected.items():
        if artifact.get(name) != value:
            raise RuntimeError(f"fixed QCCR provenance mismatch: {name}")
    parameters = artifact.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("fixed QCCR parameters are missing")
    return {name: float(value) for name, value in parameters.items()}


def build_fixed_decision(
    *,
    detector_seed: int,
    checkpoint_sha: str,
    train_capture_sha: str,
    val_capture_sha: str,
    data_lock_sha: str,
    qccr_sha: str,
    qccr_source_capture_sha: str,
    runtime_hashes: dict[str, str],
    model_path: Path,
    model_sha: str,
    implementation_sha: str,
    fixed_lock: dict[str, str],
) -> dict:
    """Build the legacy-compatible record for the single fixed tuple."""

    fixed_tuple = {
        "positive_weight": UMR_POSITIVE_WEIGHT,
        "head_seed": UMR_HEAD_SEED,
        "tau": UMR_TAU,
        "beta": UMR_BETA,
        "topk": UMR_TOPK,
    }
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "protocol": COMPATIBILITY_PROTOCOL,
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": detector_seed,
        "test_accessed": False,
        "official_test_accessed": False,
        "adaptive_candidate_evidence_used": False,
        "val_metrics_computed": False,
        "compatibility_fields_are_not_observed_gates": True,
        "feature_schema": FEATURE_SCHEMA,
        "source_checkpoint_sha256": checkpoint_sha,
        "train_capture_sha256": train_capture_sha,
        "val_capture_sha256": val_capture_sha,
        "source_data_split_lock_sha256": data_lock_sha,
        "qccr_artifact_sha256": qccr_sha,
        "qccr_source_capture_sha256": qccr_source_capture_sha,
        "runtime_source_sha256": runtime_hashes,
        # Compatibility value consumed by the frozen runtime loader.  No Val
        # outcome is consulted to produce it.
        "decision": "VAL_GO",
        "selection_policy": {
            "name": POLICY_NAME,
            "decision": "FIXED_BEFORE_TEST",
            "adaptive_selection_performed": False,
            "adaptive_candidate_evidence_used": False,
            "val_grid_read": False,
            "val_metrics_computed": False,
            "compatibility_fields_are_not_observed_gates": True,
            "fixed_tuple": fixed_tuple,
        },
        "fixed_protocol_lock": fixed_lock,
        "selected": {"pw": UMR_POSITIVE_WEIGHT, **fixed_tuple},
        "selected_model": {
            "recommended_formal_seed": UMR_HEAD_SEED,
            "path": str(model_path),
            "sha256": model_sha,
        },
        "detector_retrained": False,
        "qccr_refitted": False,
        "implementation_lock_sha256": implementation_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-protocol-lock", type=Path, required=True)
    parser.add_argument("--expected-fixed-protocol-lock-sha", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--expected-train-capture-sha", required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--qccr-artifact", type=Path, required=True)
    parser.add_argument("--expected-qccr-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
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
        tool_relative="tools/train_umr_fixed.py",
        required_locked_tools=("tools/train_umr_select_val.py",),
    )
    source_root = args.source_root.resolve()
    formal_evidence = formal_fit_guard(
        tool_relative="tools/train_umr_select_val.py",
        adapter_root_path=root,
        source_root=source_root,
        require_umr=True,
    )
    if formal_evidence["implementation_lock_sha256"] != implementation_sha:
        raise RuntimeError("fixed protocol and formal fit guard disagree")
    runtime_hashes = runtime_source_hashes(source_root, require_umr=True)
    if runtime_hashes != policy.get("fit_runtime_sources"):
        raise RuntimeError("fixed protocol uses another QCCR/UMR runtime")
    if EXPECTED_TOPK != UMR_TOPK:
        raise RuntimeError("runtime candidate Top-k differs from fixed protocol")

    output_dir = require_output_path(
        args.output_dir,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed UMR output directory",
    )
    if output_dir.exists():
        raise RuntimeError(f"refusing to reuse fixed UMR directory: {output_dir}")
    checkpoint = args.checkpoint.resolve()
    expected_checkpoint = (root / "runs" / f"seed{args.detector_seed}" / "best_val.pth").resolve()
    if checkpoint != expected_checkpoint:
        raise RuntimeError("detector checkpoint is not the seed-bound best_val.pth")
    checkpoint_sha = require_sha(
        checkpoint, args.expected_checkpoint_sha, "detector checkpoint"
    )
    train_path = args.train_capture.resolve()
    val_path = args.val_capture.resolve()
    train_capture = load_capture(
        train_path,
        expected_sha=args.expected_train_capture_sha,
        split="train",
        expected_checkpoint_sha=checkpoint_sha,
        adapter_root_path=root,
        detector_seed=args.detector_seed,
    )
    val_capture = load_capture(
        val_path,
        expected_sha=args.expected_val_capture_sha,
        split="val",
        expected_checkpoint_sha=checkpoint_sha,
        adapter_root_path=root,
        detector_seed=args.detector_seed,
    )
    if (
        train_capture.get("data_split_lock_sha256") != data_lock_sha
        or val_capture.get("data_split_lock_sha256") != data_lock_sha
    ):
        raise RuntimeError("Train/Val captures do not use the fixed data lock")

    qccr_path = require_output_path(
        args.qccr_artifact,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed QCCR input",
    )
    qccr_sha = require_sha(qccr_path, args.expected_qccr_sha, "fixed QCCR artifact")
    qccr_artifact = json.loads(qccr_path.read_text(encoding="utf-8"))
    qccr = validate_fixed_qccr(
        qccr_artifact,
        detector_seed=args.detector_seed,
        checkpoint_sha=checkpoint_sha,
        val_capture_sha=args.expected_val_capture_sha,
        data_lock_sha=data_lock_sha,
        implementation_sha=implementation_sha,
        fixed_lock_sha=lock_sha,
        qccr_runtime_sha=runtime_hashes["hilor_qccr.py"],
    )

    inputs = {
        "detector_checkpoint": checkpoint,
        "fixed_protocol_lock": args.fixed_protocol_lock.resolve(),
        "train_capture": train_path,
        "val_capture": val_path,
        "qccr_artifact": qccr_path,
        "runtime_qccr": runtime_source_path(source_root, "hilor_qccr.py"),
        "runtime_umr": runtime_source_path(source_root, "hilor_umr.py"),
    }
    before = {name: sha256_file(path) for name, path in inputs.items()}
    apply_qccr, compute_competition_risk, UMRHead, build_image_features = (
        install_source_root(source_root, require_umr=True)
    )
    train_data, train_image_ids, positive_count, candidate_count = build_train_data(
        train_capture,
        qccr,
        apply_qccr,
        compute_competition_risk,
        build_image_features,
    )
    model, history, best_epoch, best_holdout, fit_count, holdout_count = train_one(
        UMRHead,
        train_data,
        train_image_ids,
        UMR_HEAD_SEED,
        UMR_POSITIVE_WEIGHT,
    )

    models_dir = output_dir / "models"
    reports_dir = output_dir / "training_reports"
    results_dir = output_dir / "results"
    model_path = models_dir / "umr_seed0_pw4_best.pt"
    atomic_torch_save(model_path, model.state_dict())
    model_sha = sha256_file(model_path)
    fixed_lock = fixed_lock_record(root, lock_sha)
    report = {
        "schema": "rtdetr-hilor-umr-training-report-v1",
        "test_accessed": False,
        "adaptive_candidate_evidence_used": False,
        "val_metrics_computed": False,
        "selection_policy": {
            "name": POLICY_NAME,
            "decision": "FIXED_BEFORE_TEST",
            "adaptive_candidate_evidence_used": False,
            "val_metrics_computed": False,
        },
        "fixed_protocol_lock": fixed_lock,
        "seed": UMR_HEAD_SEED,
        "detector_seed": args.detector_seed,
        "positive_weight": UMR_POSITIVE_WEIGHT,
        "source_split": "CDW1630 Train",
        "source_checkpoint_sha256": checkpoint_sha,
        "train_capture_sha256": args.expected_train_capture_sha,
        "val_capture_sha256": args.expected_val_capture_sha,
        "val_used_for_fitting_or_selection": False,
        "source_data_split_lock_sha256": data_lock_sha,
        "qccr_artifact_sha256": qccr_sha,
        "qccr_source_capture_sha256": qccr_artifact["source_capture_sha256"],
        "runtime_source_sha256": runtime_hashes,
        "weight_path": str(model_path),
        "weight_sha256": model_sha,
        "feature_schema": FEATURE_SCHEMA,
        "head_parameters": sum(value.numel() for value in model.parameters()),
        "train_image_count": len(train_image_ids),
        "fit_image_count": fit_count,
        "holdout_image_count": holdout_count,
        "epochs_run": len(history["train"]),
        "best_epoch": best_epoch,
        "best_holdout_qfl": best_holdout,
        "positive_target_rate_m300": positive_count / max(candidate_count, 1),
        "history": history,
        "implementation_lock_sha256": implementation_sha,
    }
    atomic_json(reports_dir / "umr_seed0_pw4.json", report)
    decision = build_fixed_decision(
        detector_seed=args.detector_seed,
        checkpoint_sha=checkpoint_sha,
        train_capture_sha=args.expected_train_capture_sha,
        val_capture_sha=args.expected_val_capture_sha,
        data_lock_sha=data_lock_sha,
        qccr_sha=qccr_sha,
        qccr_source_capture_sha=qccr_artifact["source_capture_sha256"],
        runtime_hashes=runtime_hashes,
        model_path=model_path,
        model_sha=model_sha,
        implementation_sha=implementation_sha,
        fixed_lock=fixed_lock,
    )
    atomic_json(results_dir / "VAL_SELECTION_DECISION.json", decision)
    after = {name: sha256_file(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("one or more retained inputs changed during fixed UMR fit")
    atomic_json(
        results_dir / "INPUT_PROVENANCE.json",
        {
            "schema": "rtdetr-hilor-umr-input-provenance-v1",
            "test_accessed": False,
            "adaptive_candidate_evidence_used": False,
            "val_metrics_computed": False,
            "selection_policy": {
                "name": POLICY_NAME,
                "decision": "FIXED_BEFORE_TEST",
                "adaptive_candidate_evidence_used": False,
                "val_metrics_computed": False,
            },
            "fixed_protocol_lock": fixed_lock,
            "data_policy": {
                "fit": ["CDW1630 Train frozen detector capture"],
                "provenance_only": ["CDW1630 Val frozen detector capture"],
                "forbidden": ["CDW1630 Test"],
            },
            "inputs": {
                name: {"path": str(path), "sha256": before[name]}
                for name, path in sorted(inputs.items())
            },
            "inputs_unchanged": True,
            "candidate_training": [report],
        },
    )
    print("RT-DETR UMR FIXED-PROTOCOL TRAINING: PASS")
    print("Tuple: positive_weight=4 head_seed=0 tau=0 beta=2 topk=300")
    print("Val grid/gate: NOT RUN")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
