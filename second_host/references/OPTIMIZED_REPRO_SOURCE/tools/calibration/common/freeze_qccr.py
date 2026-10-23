#!/usr/bin/env python3
"""Freeze a QCCR artifact selected exclusively by image-level OOF Val."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PARAMETERS = ("beta0", "beta_positive", "beta_negative", "duplicate_penalty")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-method", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    actual_sha = sha256(checkpoint)
    if actual_sha != args.expected_checkpoint_sha:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected={args.expected_checkpoint_sha}, "
            f"actual={actual_sha}"
        )
    oof = json.loads(args.oof_summary.read_text(encoding="utf-8"))
    fits = json.loads(args.fits.read_text(encoding="utf-8"))
    if oof.get("decision") != "OOF_VAL_GO":
        raise RuntimeError("QCCR OOF-Val selection is not GO; no Test fallback is allowed")
    selected_key = oof["selected_candidate_key"]
    selected = oof["selected_full_val_parameters"]
    parameters = {}
    for name in PARAMETERS:
        value = float(selected[name])
        if not math.isfinite(value) or (name != "beta0" and value < 0.0):
            raise RuntimeError(f"invalid QCCR parameter: {name}={value}")
        parameters[name] = value

    artifact = {
        "format_version": 1,
        "method": "QCCR-v1",
        "dataset": args.dataset_name,
        "source_method": args.source_method,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": actual_sha,
        "selection_protocol": "image-level-five-fold-OOF-Val",
        "selection_decision": "OOF_VAL_GO",
        "test_used_for_selection": False,
        "selected_candidate_key": selected_key,
        "selected_lambda_reg": oof["selected_lambda_reg"],
        "parameters": parameters,
        "fit_candidate_count": fits.get("fit_candidate_count"),
        "val_baseline_metrics": oof.get("baseline_metrics"),
        "selected_oof_metrics": oof["candidate_decisions"][selected_key]["metrics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Dataset        :", args.dataset_name)
    print("Selected lambda:", artifact["selected_lambda_reg"])
    print("Test accessed  : NO")
    print("QCCR OOF-VAL ARTIFACT FREEZE: PASS")


if __name__ == "__main__":
    main()
