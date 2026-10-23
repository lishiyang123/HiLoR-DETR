#!/usr/bin/env python3
"""Freeze QCCR at the predeclared DEIM lambda without reading an OOF gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from fit_common import atomic_json, require_output_path, require_sha
    from fit_qccr import FIT_SCHEMA
    from fixed_protocol_common import (
        POLICY_NAME,
        QCCR_LAMBDA_REG,
        fixed_lock_record,
        load_fixed_protocol,
    )
except ImportError:  # imported as tools.freeze_qccr_fixed in unit tests
    from tools.fit_common import atomic_json, require_output_path, require_sha
    from tools.fit_qccr import FIT_SCHEMA
    from tools.fixed_protocol_common import (
        POLICY_NAME,
        QCCR_LAMBDA_REG,
        fixed_lock_record,
        load_fixed_protocol,
    )


PARAMETERS = ("beta0", "beta_positive", "beta_negative", "duplicate_penalty")
FROZEN_SCHEMA = "rtdetr-hilor-qccr-frozen-artifact-v1"


def _fixed_candidate(fits: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = fits.get("candidates")
    if not isinstance(candidates, dict):
        raise RuntimeError("QCCR fits lack candidate records")
    key = "1e-03"
    candidate = candidates.get(key)
    if not isinstance(candidate, dict):
        raise RuntimeError("QCCR fits lack the fixed lambda=0.001 full-Val fit")
    try:
        value = float(candidate.get("lambda_reg"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("fixed QCCR lambda record is invalid") from exc
    if value != QCCR_LAMBDA_REG:
        raise RuntimeError("fixed QCCR lambda record is not 0.001")
    return key, candidate


def build_fixed_artifact(
    fits: dict[str, Any],
    *,
    detector_seed: int,
    checkpoint: Path,
    checkpoint_sha: str,
    fits_sha: str,
    implementation_sha: str,
    data_lock_sha: str,
    qccr_runtime_sha: str,
    fixed_lock: dict[str, str],
) -> dict[str, Any]:
    """Convert bound fit evidence into a deployable, fixed-policy artifact."""

    if fits.get("schema") != FIT_SCHEMA or fits.get("test_accessed") is not False:
        raise RuntimeError("invalid QCCR fit artifact")
    if fits.get("implementation_lock_sha256") != implementation_sha:
        raise RuntimeError("QCCR fits belong to another implementation lock")
    if fits.get("source_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("QCCR fits belong to another detector")
    if fits.get("detector_seed") != detector_seed:
        raise RuntimeError("QCCR fits belong to another detector seed")
    if fits.get("source_data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("QCCR fits use another data-split lock")
    source_capture_sha = fits.get("source_capture_sha256")
    if not isinstance(source_capture_sha, str) or len(source_capture_sha) != 64:
        raise RuntimeError("QCCR fits lack a bound Val-capture SHA-256")
    runtime_hashes = fits.get("runtime_source_sha256")
    if runtime_hashes != {"hilor_qccr.py": qccr_runtime_sha}:
        raise RuntimeError("QCCR fits use another QCCR runtime source")

    candidate_key, candidate = _fixed_candidate(fits)
    full_fit = candidate.get("full_val_fit")
    if not isinstance(full_fit, dict) or full_fit.get("optimizer_success") is not True:
        raise RuntimeError("fixed-lambda full-Val QCCR optimizer did not succeed")
    raw_parameters = full_fit.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise RuntimeError("fixed-lambda QCCR fit lacks parameters")
    parameters: dict[str, float] = {}
    for name in PARAMETERS:
        try:
            value = float(raw_parameters[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid QCCR parameter: {name}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite QCCR parameter: {name}")
        if name in {"beta_positive", "beta_negative"} and value <= 0.0:
            raise RuntimeError(f"QCCR {name} must be strictly positive")
        if name == "duplicate_penalty" and value < 0.0:
            raise RuntimeError("QCCR duplicate_penalty must be non-negative")
        parameters[name] = value

    return {
        "schema": FROZEN_SCHEMA,
        "format_version": 1,
        "method": "QCCR-v1",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": detector_seed,
        "source_method": "RT-DETR-R18+GCCS-alpha_u=0",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_capture_sha256": source_capture_sha,
        "source_data_split_lock_sha256": data_lock_sha,
        "runtime_source_sha256": {
            "hilor_qccr.py": runtime_hashes["hilor_qccr.py"]
        },
        # Compatibility fields consumed by the already-frozen runtime loader.
        # They are policy markers, not the result of an OOF gate.
        "selection_protocol": POLICY_NAME,
        "selection_decision": "OOF_VAL_GO",
        "test_used_for_selection": False,
        "test_accessed": False,
        "adaptive_candidate_evidence_used": False,
        "val_metrics_computed": False,
        "compatibility_fields_are_not_observed_gates": True,
        "selection_policy": {
            "name": POLICY_NAME,
            "decision": "FIXED_BEFORE_TEST",
            "adaptive_selection_performed": False,
            "adaptive_candidate_evidence_used": False,
            "oof_gate_read": False,
            "val_metrics_computed": False,
            "compatibility_fields_are_not_observed_gates": True,
        },
        "fixed_protocol_lock": fixed_lock,
        "fits_sha256": fits_sha,
        "selected_candidate_key": candidate_key,
        "selected_lambda_reg": QCCR_LAMBDA_REG,
        "parameters": parameters,
        "fit_candidate_count": fits.get("fit_candidate_count"),
        "implementation_lock_sha256": implementation_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-protocol-lock", type=Path, required=True)
    parser.add_argument("--expected-fixed-protocol-lock-sha", required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--expected-fits-sha", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        tool_relative="tools/freeze_qccr_fixed.py",
        required_locked_tools=("tools/fit_qccr.py",),
    )
    checkpoint = args.checkpoint.resolve()
    expected_checkpoint = (root / "runs" / f"seed{args.detector_seed}" / "best_val.pth").resolve()
    if checkpoint != expected_checkpoint:
        raise RuntimeError("detector checkpoint is not the seed-bound best_val.pth")
    checkpoint_sha = require_sha(
        checkpoint, args.expected_checkpoint_sha, "detector checkpoint"
    )
    fit_path = require_output_path(
        args.fits,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="QCCR fit input",
    )
    fits_sha = require_sha(fit_path, args.expected_fits_sha, "QCCR fits")
    fits = json.loads(fit_path.read_text(encoding="utf-8"))
    artifact = build_fixed_artifact(
        fits,
        detector_seed=args.detector_seed,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        fits_sha=fits_sha,
        implementation_sha=implementation_sha,
        data_lock_sha=data_lock_sha,
        qccr_runtime_sha=policy["fit_runtime_sources"]["hilor_qccr.py"],
        fixed_lock=fixed_lock_record(root, lock_sha),
    )
    output = require_output_path(
        args.output,
        namespace="inference",
        detector_seed=args.detector_seed,
        adapter_root_path=root,
        label="fixed QCCR artifact",
    )
    atomic_json(output, artifact)
    print("RT-DETR QCCR FIXED-PROTOCOL FREEZE: PASS")
    print("Lambda: 0.001 (predeclared; no OOF gate read)")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
