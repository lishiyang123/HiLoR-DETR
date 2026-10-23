#!/usr/bin/env python3
"""Freeze a detector-bound QCCR artifact selected exclusively on Val."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from evaluate_qccr_oof_val import SELECTION_SCHEMA
    from fit_common import (
        atomic_json, formal_fit_guard, require_output_path,
        require_sha,
    )
    from fit_qccr import FIT_SCHEMA
except ImportError:  # imported as tools.freeze_qccr in unit tests
    from tools.evaluate_qccr_oof_val import SELECTION_SCHEMA
    from tools.fit_common import (
        atomic_json, formal_fit_guard, require_output_path,
        require_sha,
    )
    from tools.fit_qccr import FIT_SCHEMA


PARAMETERS = ("beta0", "beta_positive", "beta_negative", "duplicate_penalty")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-selection", type=Path, required=True)
    parser.add_argument("--expected-oof-selection-sha", required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--expected-fits-sha", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    formal_evidence = formal_fit_guard(
        tool_relative="tools/freeze_qccr.py",
        adapter_root_path=args.adapter_root,
        check_runtime=False,
    )
    implementation_sha = formal_evidence["implementation_lock_sha256"]

    checkpoint = args.checkpoint.resolve()
    require_sha(checkpoint, args.expected_checkpoint_sha, "detector checkpoint")
    oof_path = require_output_path(
        args.oof_selection, namespace="inference",
        detector_seed=args.detector_seed, adapter_root_path=args.adapter_root,
        label="QCCR OOF selection input",
    )
    oof_sha = require_sha(
        oof_path, args.expected_oof_selection_sha, "OOF Val selection"
    )
    fit_path = require_output_path(
        args.fits, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root, label="QCCR fit input",
    )
    fit_sha = require_sha(fit_path, args.expected_fits_sha, "QCCR fits")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    fits = json.loads(fit_path.read_text(encoding="utf-8"))
    if oof.get("schema") != SELECTION_SCHEMA or oof.get("test_accessed") is not False:
        raise RuntimeError("invalid OOF Val selection artifact")
    if fits.get("schema") != FIT_SCHEMA or fits.get("test_accessed") is not False:
        raise RuntimeError("invalid QCCR fit artifact")
    for label, payload in (("OOF selection", oof), ("QCCR fits", fits)):
        if payload.get("implementation_lock_sha256") != implementation_sha:
            raise RuntimeError(f"{label} belongs to another implementation lock")
    if oof.get("decision") != "OOF_VAL_GO":
        raise RuntimeError("OOF Val did not select QCCR; Test fallback is forbidden")
    if oof.get("protocol") != "image-level-five-fold-OOF-Val":
        raise RuntimeError("OOF Val selection uses another protocol")
    if oof.get("fits_sha256") != fit_sha:
        raise RuntimeError("OOF selection and QCCR fits disagree")
    if oof.get("source_capture_sha256", oof.get("capture_sha256")) != fits.get(
        "source_capture_sha256"
    ):
        raise RuntimeError("OOF selection and QCCR fits use different Val captures")
    if oof.get("source_data_split_lock_sha256") != fits.get(
        "source_data_split_lock_sha256"
    ):
        raise RuntimeError("OOF selection and QCCR fits use different data locks")
    if oof.get("runtime_source_sha256") != fits.get("runtime_source_sha256"):
        raise RuntimeError("OOF selection and QCCR fits use different runtime sources")
    for payload in (oof, fits):
        if payload.get("source_checkpoint_sha256", payload.get("checkpoint_sha256")) != args.expected_checkpoint_sha:
            raise RuntimeError("QCCR evidence belongs to another detector")
        if payload.get("detector_seed") != args.detector_seed:
            raise RuntimeError("QCCR evidence belongs to another detector seed")

    selected_key = oof["selected_candidate_key"]
    if selected_key not in fits["candidates"]:
        raise RuntimeError("selected QCCR candidate is absent from fit artifact")
    selected = oof["selected_full_val_parameters"]
    expected_selected = fits["candidates"][selected_key]["full_val_fit"]["parameters"]
    selected_fit = fits["candidates"][selected_key]
    if selected_fit["full_val_fit"].get("optimizer_success") is not True:
        raise RuntimeError("selected full-Val QCCR optimizer did not succeed")
    if any(entry.get("optimizer_success") is not True for entry in selected_fit["folds"]):
        raise RuntimeError("selected QCCR fold optimizer did not succeed")
    if float(oof["selected_lambda_reg"]) != float(
        fits["candidates"][selected_key]["lambda_reg"]
    ):
        raise RuntimeError("selected QCCR lambda does not match the fit candidate")
    if fits.get("seed") != 0 or fits.get("fold_count") != 5:
        raise RuntimeError("QCCR fit does not use the locked seed/fold protocol")
    parameters = {}
    for name in PARAMETERS:
        value = float(selected[name])
        if value != float(expected_selected[name]):
            raise RuntimeError("selected QCCR parameters do not match full-Val fit")
        if not math.isfinite(value):
            raise RuntimeError(f"invalid QCCR parameter: {name}={value}")
        if name in {"beta_positive", "beta_negative"} and value <= 0.0:
            raise RuntimeError(f"QCCR {name} must be strictly positive")
        if name == "duplicate_penalty" and value < 0.0:
            raise RuntimeError("QCCR duplicate_penalty must be non-negative")
        parameters[name] = value
    output = require_output_path(
        args.output, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="frozen QCCR artifact",
    )
    artifact = {
        "schema": "rtdetr-hilor-qccr-frozen-artifact-v1",
        "format_version": 1,
        "method": "QCCR-v1",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": args.detector_seed,
        "source_method": "RT-DETR-R18+GCA-alpha_u=0",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": args.expected_checkpoint_sha,
        "source_capture_sha256": fits["source_capture_sha256"],
        "source_data_split_lock_sha256": fits["source_data_split_lock_sha256"],
        "runtime_source_sha256": fits["runtime_source_sha256"],
        "selection_protocol": "image-level-five-fold-OOF-Val",
        "selection_decision": "OOF_VAL_GO",
        "test_used_for_selection": False,
        "oof_selection_sha256": oof_sha,
        "fits_sha256": fit_sha,
        "selected_candidate_key": selected_key,
        "selected_lambda_reg": oof["selected_lambda_reg"],
        "parameters": parameters,
        "fit_candidate_count": fits["fit_candidate_count"],
        "val_baseline_metrics": oof["baseline_metrics"],
        "selected_oof_metrics": oof["candidate_decisions"][selected_key]["metrics"],
        "implementation_lock_sha256": implementation_sha,
    }
    atomic_json(output, artifact)
    print("RT-DETR QCCR OOF-VAL ARTIFACT FREEZE: PASS")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
