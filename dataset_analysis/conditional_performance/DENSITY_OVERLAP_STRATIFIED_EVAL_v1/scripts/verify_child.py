#!/usr/bin/env python3
"""Verify an installed density/overlap child package, optionally recomputing it."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from analysis import run_analysis
from common import METHODS, SEEDS, read_json, require, sha256, verify_manifest


def resolve_parent_annotations(package: Path):
    dataset_package = package.parents[1]
    root = dataset_package / "inputs" / "annotations"
    return root / "instances_train2017.json", root / "instances_test2017.json"


def verify(package: Path, rerun: bool = False) -> int:
    package = package.resolve()
    require(package.name == "DENSITY_OVERLAP_STRATIFIED_EVAL_v1", "Unexpected child package name")
    count = verify_manifest(package)
    expected_top = {
        "README.md", "protocol.json", "input_bindings.json", "train_thresholds.json",
        "test_image_groups.json", "per_seed_results.csv", "summary_table.csv",
        "results.json", "predictions", "configs", "scripts", "manifest.json",
    }
    require({item.name for item in package.iterdir()} == expected_top, "Child top-level layout changed")
    protocol = read_json(package / "protocol.json")
    require(protocol.get("status") == "RULES_FROZEN_BEFORE_SUBGROUP_TEST_EVALUATION", "Protocol is not frozen")
    require(protocol.get("detector_seeds") == [0, 1, 2], "Detector-seed scope changed")
    require(protocol.get("methods") == ["B0", "Full"], "Method scope changed")
    require(
        protocol["evaluation"].get("test_used_for_model_checkpoint_threshold_or_parameter_selection") is False,
        "Protocol claims Test selection",
    )
    thresholds = read_json(package / "train_thresholds.json")
    require(thresholds["density"]["threshold"] == 5, "Bound Train density threshold changed")
    require(thresholds["overlap"]["strategy"] == "positive_median", "Bound overlap strategy changed")
    require(thresholds.get("test_metrics_consulted") is False, "Threshold freeze consulted Test metrics")

    groups = read_json(package / "test_image_groups.json")
    require(groups.get("test_image_count") == 237, "Test image count changed")
    for prefix in ("density", "overlap"):
        low = groups["groups"][prefix + "_low"]["image_ids"]
        high = groups["groups"][prefix + "_high"]["image_ids"]
        require(not set(low).intersection(high), "%s groups overlap" % prefix)
        require(len(set(low) | set(high)) == 237, "%s partition incomplete" % prefix)

    bindings = read_json(package / "input_bindings.json")
    require(bindings.get("test_used_for_selection") is False, "Bindings claim Test selection")
    require(bindings.get("new_training") is False and bindings.get("new_fitting") is False, "Unexpected training/fitting")
    for method in METHODS:
        for seed in SEEDS:
            key = "%s_seed%d" % (method, seed)
            pred = package / "predictions" / method / ("seed%d" % seed) / "predictions.json"
            summary = package / "predictions" / method / ("seed%d" % seed) / "PREDICTION_SUMMARY.json"
            require(pred.is_file() and summary.is_file(), "Missing prediction evidence: %s" % key)
            payload = read_json(pred)
            record = bindings["predictions"][key]
            require(payload.get("formal_full_test_metric_replay") == "PASS", "Formal replay missing: %s" % key)
            require(payload.get("test_used_for_selection") is False, "Test selection claimed: %s" % key)
            require(record["predictions_sha256"] == sha256(pred), "Prediction binding mismatch: %s" % key)
            require(record["summary_sha256"] == sha256(summary), "Summary binding mismatch: %s" % key)

    result = read_json(package / "results.json")
    require(result.get("test_used_for_selection") is False, "Result claims Test selection")
    require(len(result.get("summary_rows_percent", [])) == 4, "Expected four summary rows")
    require(set(result.get("raw_metrics", {})) == {
        "density_low", "density_high", "overlap_low", "overlap_high"
    }, "Result groups changed")

    if rerun:
        train_annotation, test_annotation = resolve_parent_annotations(package)
        require(train_annotation.is_file() and test_annotation.is_file(), "Parent annotations are unavailable")
        with tempfile.TemporaryDirectory(prefix="cdw_density_overlap_verify_") as directory:
            reproduced = run_analysis(
                train_annotation, test_annotation, package / "train_thresholds.json",
                package / "predictions", Path(directory),
            )
            require(
                reproduced["raw_metrics"] == result["raw_metrics"]
                and reproduced["summary_rows_percent"] == result["summary_rows_percent"],
                "CPU subgroup analysis did not reproduce",
            )
    print("===== DENSITY/OVERLAP STRATIFIED CHILD VERIFICATION =====")
    print("Groups          : density low/high + overlap low/high")
    print("Predictions     : B0/Full x seeds 0/1/2 / formal full-Test replay PASS")
    print("Threshold source: Train only")
    print("Evaluation unit : whole Test images / all GT retained")
    print("Test selection  : NO")
    print("Manifest        : PASS (%d files)" % count)
    print("CPU recompute   : %s" % ("PASS" if rerun else "NOT REQUESTED"))
    print("VERIFICATION: PASS")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    verify(args.package, rerun=args.rerun)


if __name__ == "__main__":
    main()
