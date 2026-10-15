#!/usr/bin/env python3
"""Freeze a checkpoint-bound RT-DETR QCCR or QCCR–UMR manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from fit_common import (
        atomic_json,
        formal_fit_guard,
        load_capture,
        require_output_path,
        require_sha,
        runtime_source_hashes,
    )
except ImportError:  # imported as tools.freeze_inference_binding in unit tests
    from tools.fit_common import (
        atomic_json,
        formal_fit_guard,
        load_capture,
        require_output_path,
        require_sha,
        runtime_source_hashes,
    )


SCHEMA = "cdw1630-rtdetr-r18-qccr-umr-inference-binding-v1"
FEATURE_SCHEMA = "umr-25d-rtdetr-final-hidden-v1"
UMR_SELECTION_SCHEMA = "rtdetr-hilor-umr-val-selection-v1"
UMR_SELECTION_PROTOCOL = "CDW1630 Train candidates / Val-only selection"


def relative_to_manifest(path: Path, output: Path) -> str:
    try:
        return os.path.relpath(path, output.parent).replace("\\", "/")
    except ValueError:
        return str(path)


def _require_mode_arguments(args, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) in (None, "")]
    if missing:
        raise RuntimeError(
            "qccr-umr mode requires: "
            + ", ".join(name.replace("_", "-") for name in missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("qccr", "qccr-umr"), required=True)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-detector-sha", required=True)
    parser.add_argument("--train-capture", type=Path)
    parser.add_argument("--expected-train-capture-sha")
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--qccr-artifact", type=Path, required=True)
    parser.add_argument("--expected-qccr-sha", required=True)
    parser.add_argument("--umr-checkpoint", type=Path)
    parser.add_argument("--expected-umr-sha")
    parser.add_argument("--umr-selection", type=Path)
    parser.add_argument("--expected-umr-selection-sha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal_evidence = formal_fit_guard(
        tool_relative="tools/freeze_inference_binding.py",
        adapter_root_path=args.adapter_root,
        source_root=args.source_root,
        require_umr=args.mode == "qccr-umr",
        check_runtime=False,
    )
    implementation_sha = formal_evidence["implementation_lock_sha256"]

    use_umr = args.mode == "qccr-umr"
    umr_arguments = (
        "train_capture",
        "expected_train_capture_sha",
        "umr_checkpoint",
        "expected_umr_sha",
        "umr_selection",
        "expected_umr_selection_sha",
    )
    if use_umr:
        _require_mode_arguments(args, umr_arguments)
    elif any(getattr(args, name) not in (None, "") for name in umr_arguments):
        raise RuntimeError("qccr mode must not bind Train or UMR artifacts")

    output = require_output_path(
        args.output,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="inference binding",
    )
    detector = args.detector_checkpoint.resolve()
    if detector.parent.name != f"seed{args.detector_seed}":
        raise RuntimeError("detector checkpoint path belongs to another seed")
    detector_sha = require_sha(
        detector, args.expected_detector_sha, "detector checkpoint"
    )
    val_path = args.val_capture.resolve()
    val = load_capture(
        val_path,
        expected_sha=args.expected_val_capture_sha,
        split="val",
        expected_checkpoint_sha=detector_sha,
        adapter_root_path=args.adapter_root,
        detector_seed=args.detector_seed,
    )
    data_lock_sha = val["data_split_lock_sha256"]
    captures = {
        "val": {
            "sha256": args.expected_val_capture_sha,
            "detector_checkpoint_sha256": detector_sha,
            "data_split_lock_sha256": data_lock_sha,
        }
    }
    if use_umr:
        train_path = args.train_capture.resolve()
        train = load_capture(
            train_path,
            expected_sha=args.expected_train_capture_sha,
            split="train",
            expected_checkpoint_sha=detector_sha,
            adapter_root_path=args.adapter_root,
            detector_seed=args.detector_seed,
        )
        if train.get("data_split_lock_sha256") != data_lock_sha:
            raise RuntimeError("Train and Val captures use different data-split locks")
        captures["train"] = {
            "sha256": args.expected_train_capture_sha,
            "detector_checkpoint_sha256": detector_sha,
            "data_split_lock_sha256": data_lock_sha,
        }

    source_root = args.source_root.resolve()
    runtime_hashes = runtime_source_hashes(source_root, require_umr=use_umr)
    qccr_path = require_output_path(
        args.qccr_artifact,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="frozen QCCR input",
    )
    qccr_sha = require_sha(qccr_path, args.expected_qccr_sha, "QCCR artifact")
    qccr = json.loads(qccr_path.read_text(encoding="utf-8"))
    if (
        qccr.get("schema") != "rtdetr-hilor-qccr-frozen-artifact-v1"
        or qccr.get("method") != "QCCR-v1"
        or qccr.get("dataset") != "CDW1630"
        or qccr.get("selection_decision") != "OOF_VAL_GO"
        or qccr.get("detector_seed") != args.detector_seed
        or qccr.get("implementation_lock_sha256") != implementation_sha
    ):
        raise RuntimeError("invalid QCCR artifact")
    if qccr.get("source_checkpoint_sha256") != detector_sha:
        raise RuntimeError("QCCR artifact belongs to another detector")
    if qccr.get("test_used_for_selection") is not False:
        raise RuntimeError("QCCR artifact does not prove Val-only selection")
    if qccr.get("source_capture_sha256") != args.expected_val_capture_sha:
        raise RuntimeError("QCCR artifact was not fitted from this exact Val capture")
    if qccr.get("source_data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("QCCR artifact uses another data-split lock")
    qccr_runtime = qccr.get("runtime_source_sha256")
    if qccr_runtime != {"hilor_qccr.py": runtime_hashes["hilor_qccr.py"]}:
        raise RuntimeError("QCCR artifact uses another QCCR runtime source")

    parameters = qccr.get("parameters") or {}
    try:
        beta0 = float(parameters["beta0"])
        beta_positive = float(parameters["beta_positive"])
        beta_negative = float(parameters["beta_negative"])
        duplicate_penalty = float(parameters["duplicate_penalty"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("QCCR artifact has invalid parameters") from error
    if not all(
        math.isfinite(value)
        for value in (beta0, beta_positive, beta_negative, duplicate_penalty)
    ):
        raise RuntimeError("QCCR artifact has non-finite parameters")
    if beta_positive <= 0.0 or beta_negative <= 0.0:
        raise RuntimeError("QCCR beta parameters must be strictly positive")
    if duplicate_penalty < 0.0:
        raise RuntimeError("QCCR duplicate penalty must be non-negative")
    binding = {
        "schema": SCHEMA,
        "format_version": 2,
        "mode": args.mode,
        "detector_seed": args.detector_seed,
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "test_accessed_for_fitting_or_selection": False,
        "fit_and_selection_splits": ["train", "val"] if use_umr else ["val"],
        "detector_checkpoint": str(detector),
        "detector_checkpoint_sha256": detector_sha,
        "source_data_split_lock_sha256": data_lock_sha,
        "source_val_capture_sha256": args.expected_val_capture_sha,
        "runtime_source_sha256": runtime_hashes,
        "implementation_lock_sha256": implementation_sha,
        "captures": captures,
        "qccr_artifact": {
            "path": relative_to_manifest(qccr_path, output),
            "sha256": qccr_sha,
            "source_capture_sha256": qccr["source_capture_sha256"],
            "source_data_split_lock_sha256": data_lock_sha,
            "runtime_source_sha256": {
                "hilor_qccr.py": runtime_hashes["hilor_qccr.py"]
            },
        },
    }

    if use_umr:
        umr_path = require_output_path(
            args.umr_checkpoint,
            namespace="inference",
            detector_seed=args.detector_seed,
            adapter_root_path=args.adapter_root,
            label="UMR checkpoint input",
        )
        selection_path = require_output_path(
            args.umr_selection,
            namespace="inference",
            detector_seed=args.detector_seed,
            adapter_root_path=args.adapter_root,
            label="UMR selection input",
        )
        umr_sha = require_sha(umr_path, args.expected_umr_sha, "UMR checkpoint")
        selection_sha = require_sha(
            selection_path, args.expected_umr_selection_sha, "UMR selection"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema") != UMR_SELECTION_SCHEMA:
            raise RuntimeError("invalid UMR Val-selection schema")
        if selection.get("decision") != "VAL_GO":
            raise RuntimeError("UMR selection is not VAL_GO")
        if (
            selection.get("test_accessed") is not False
            or selection.get("official_test_accessed") is not False
        ):
            raise RuntimeError("UMR selection does not prove Test exclusion")
        if selection.get("protocol") != UMR_SELECTION_PROTOCOL:
            raise RuntimeError("UMR selection uses another Train/Val protocol")
        if selection.get("implementation_lock_sha256") != implementation_sha:
            raise RuntimeError("UMR selection uses another implementation lock")
        selected_model = selection.get("selected_model") or {}
        if selected_model.get("sha256") != umr_sha:
            raise RuntimeError("UMR selection and checkpoint SHA disagree")
        expected = {
            "source_checkpoint_sha256": detector_sha,
            "train_capture_sha256": args.expected_train_capture_sha,
            "val_capture_sha256": args.expected_val_capture_sha,
            "source_data_split_lock_sha256": data_lock_sha,
            "qccr_artifact_sha256": qccr_sha,
            "qccr_source_capture_sha256": args.expected_val_capture_sha,
            "runtime_source_sha256": runtime_hashes,
            "feature_schema": FEATURE_SCHEMA,
            "detector_seed": args.detector_seed,
        }
        for name, value in expected.items():
            if selection.get(name) != value:
                raise RuntimeError(f"UMR selection provenance mismatch: {name}")
        selected = selection.get("selected") or {}
        tau = float(selected["tau"])
        beta = float(selected["beta"])
        if (
            not math.isfinite(tau)
            or not math.isfinite(beta)
            or beta <= 0.0
        ):
            raise RuntimeError("UMR selection has invalid tau/beta values")
        selected_model_path = Path(str(selected_model.get("path", "")))
        if not selected_model_path.is_absolute():
            selected_model_path = selection_path.parent / selected_model_path
        if selected_model_path.resolve() != umr_path:
            raise RuntimeError("UMR selected-model path does not match supplied checkpoint")
        binding["umr"] = {
            "path": relative_to_manifest(umr_path, output),
            "sha256": umr_sha,
            "selection_path": relative_to_manifest(selection_path, output),
            "selection_sha256": selection_sha,
            "feature_schema": FEATURE_SCHEMA,
            "num_classes": 5,
            "num_queries": 300,
            "topk": 300,
            "tau": tau,
            "beta": beta,
            "source_val_capture_sha256": args.expected_val_capture_sha,
            "runtime_source_sha256": runtime_hashes,
        }

    atomic_json(output, binding)
    print("RT-DETR INFERENCE BINDING: PASS")
    print("Mode:", args.mode)
    print("Detector SHA:", detector_sha)
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
