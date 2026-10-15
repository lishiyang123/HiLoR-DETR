#!/usr/bin/env python3
"""Prepare, run, and verify the staged sorting-proposal budget evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from common import (
    BUDGETS,
    DATASET_PACKAGE,
    EXPECTED_CANDIDATES_PER_IMAGE,
    EXPECTED_TEST_IMAGES,
    IOU_THRESHOLDS,
    MAIN_BUDGET,
    MAIN_IOU,
    MAIN_PACKAGE,
    METHODS,
    PREDICTION_CHILD,
    RELEASE_NAME,
    SEEDS,
    STATE_PACKAGE,
    file_record,
    read_json,
    rebuild_manifest,
    require,
    sha256,
    verify_file_record,
    verify_manifest,
    verify_manifest_member,
    write_json,
    write_text,
)
from evaluate import RATE_METRICS, run_evaluation


HERE = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def paths(project: Path) -> Dict[str, Path]:
    project = project.resolve()
    release = project / RELEASE_NAME
    main = release / MAIN_PACKAGE
    data_package = release / DATASET_PACKAGE
    prediction_child = data_package / "conditional_performance" / PREDICTION_CHILD
    state = project / STATE_PACKAGE
    return {
        "project": project,
        "release": release,
        "release_verifier": release / "run_verify.sh",
        "main": main,
        "main_manifest": main / "manifests/FILE_MANIFEST.sha256.json",
        "annotation": main / "datasets/CDW1630/annotations/instances_test2017.json",
        "data_package": data_package,
        "prediction_child": prediction_child,
        "prediction_manifest": prediction_child / "manifest.json",
        "prediction_root": prediction_child / "predictions",
        "state": state,
        "state_protocol": state / "protocol/ACTION_BUDGET_PROTOCOL_LOCK.json",
        "bindings": state / "manifests/INPUT_BINDINGS.json",
        "results": state / "results/ACTION_BUDGET_RESULTS.json",
    }


def prediction_path(p: Dict[str, Path], method: str, seed: int) -> Path:
    return p["prediction_root"] / method / ("seed%d" % seed) / "predictions.json"


def prediction_summary_path(p: Dict[str, Path], method: str, seed: int) -> Path:
    return p["prediction_root"] / method / ("seed%d" % seed) / "PREDICTION_SUMMARY.json"


def run_release_verifier(p: Dict[str, Path]) -> None:
    result = subprocess.run(["bash", str(p["release_verifier"])], cwd=str(p["release"]), check=False)
    require(result.returncode == 0, "Final release verification failed")


def validate_inputs(project: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    p = paths(project)
    for path in (p["release"], p["main"], p["data_package"], p["prediction_child"], p["prediction_root"]):
        require(path.is_dir(), "Missing formal directory: %s" % path)
    for path in (p["release_verifier"], p["main_manifest"], p["annotation"], p["prediction_manifest"]):
        require(path.is_file(), "Missing formal file: %s" % path)
    verify_manifest_member(p["main_manifest"], p["main"], p["annotation"])

    annotation = read_json(p["annotation"])
    image_ids = sorted(int(row["id"]) for row in annotation.get("images", []))
    require(len(image_ids) == EXPECTED_TEST_IMAGES, "Expected 237 official Test images")
    require(len(set(image_ids)) == EXPECTED_TEST_IMAGES, "Official Test image ids are not unique")

    identities = {}
    for method in METHODS:
        for seed in SEEDS:
            key = "%s_seed%d" % (method, seed)
            prediction = prediction_path(p, method, seed)
            summary_path = prediction_summary_path(p, method, seed)
            require(prediction.is_file(), "Missing formal prediction: %s" % prediction)
            require(summary_path.is_file(), "Missing formal prediction summary: %s" % summary_path)
            verify_manifest_member(p["prediction_manifest"], p["prediction_child"], prediction)
            verify_manifest_member(p["prediction_manifest"], p["prediction_child"], summary_path)
            payload = read_json(prediction)
            summary = read_json(summary_path)
            require(payload.get("method") == method, "%s prediction method mismatch" % key)
            require(int(payload.get("detector_seed", -1)) == seed, "%s prediction seed mismatch" % key)
            require(payload.get("formal_full_test_metric_replay") == "PASS", "%s formal replay missing" % key)
            require(payload.get("test_used_for_selection") is False, "%s claims Test selection" % key)
            require([int(value) for value in payload.get("image_ids", [])] == image_ids, "%s image set mismatch" % key)
            require(int(payload.get("image_count", -1)) == EXPECTED_TEST_IMAGES, "%s image count mismatch" % key)
            require(int(payload.get("detection_count", -1)) == EXPECTED_TEST_IMAGES * EXPECTED_CANDIDATES_PER_IMAGE, "%s is not Top-300 per image" % key)
            require(summary.get("method") == method, "%s summary method mismatch" % key)
            require(int(summary.get("detector_seed", -1)) == seed, "%s summary seed mismatch" % key)
            require(summary.get("formal_full_test_metric_replay") == "PASS", "%s summary replay missing" % key)
            require(summary.get("test_used_for_selection") is False, "%s summary claims Test selection" % key)
            require(summary.get("predictions_sha256") == sha256(prediction), "%s summary/prediction SHA mismatch" % key)
            identities[key] = {
                "prediction": prediction,
                "summary": summary_path,
                "method": method,
                "seed": seed,
            }
    return p, identities


def preflight(project: Path) -> None:
    p, identities = validate_inputs(project)
    protocol = read_json(HERE / "protocol/ACTION_BUDGET_PROTOCOL.json")
    require(protocol.get("release_write") is False, "Protocol unexpectedly permits release writes")
    run_release_verifier(p)
    print("===== CDW1630 SORTING-PROPOSAL BUDGET PREFLIGHT =====")
    print("Inputs          : 6/6 formal Test prediction files SHA-bound")
    print("Methods/seeds   : B0 + Full / 0, 1, 2")
    print("Candidate queue : final Top-300 / no score threshold")
    print("Budgets         : K=1..10; main K=5")
    print("IoU thresholds  : 0.75 main / 0.80 sensitivity")
    print("Train/inference : NO / NO")
    print("GPU             : NOT USED")
    print("Release write   : NO (staged package only)")
    print("Formal inputs   : %d method-seed identities" % len(identities))
    print("PREFLIGHT: PASS")


def state_readme() -> str:
    return """# CDW1630 budgeted sorting-proposal evaluation — staged state

This staged package evaluates the six frozen CDW1630 Test prediction files as
score-ordered sorting-proposal queues. It does not train or run a detector and
does not write to the final paper reproduction release.

The main setting is `K=5` actions per image at `IoU >= 0.75`; `K=1..10` curves
and an `IoU >= 0.80` sensitivity analysis are also retained. Each method/seed
is evaluated separately on the complete Test set before three-seed means,
sample standard deviations, and paired Full-minus-B0 differences are formed.

The analysis is an offline application proxy. It does not measure grasp
success, physical recovery, robot timing, or a closed-loop sorting system.
"""


def build_bindings(p: Dict[str, Path], identities: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "format_version": 1,
        "created_utc": utc_now(),
        "release": RELEASE_NAME,
        "release_write": False,
        "main_manifest": file_record(p["main_manifest"], p["release"]),
        "prediction_package_manifest": file_record(p["prediction_manifest"], p["release"]),
        "test_annotation": file_record(p["annotation"], p["release"]),
        "predictions": {},
    }
    for key, identity in identities.items():
        payload["predictions"][key] = {
            "method": identity["method"],
            "seed": identity["seed"],
            "prediction": file_record(identity["prediction"], p["release"]),
            "summary": file_record(identity["summary"], p["release"]),
        }
    return payload


def prepare(project: Path) -> None:
    p, identities = validate_inputs(project)
    state = p["state"]
    for relative in ("protocol", "manifests", "scripts", "results"):
        (state / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "protocol/ACTION_BUDGET_PROTOCOL.json", p["state_protocol"])
    for name in ("common.py", "evaluate.py"):
        shutil.copy2(HERE / name, state / "scripts" / name)
    write_text(state / "README.md", state_readme())
    write_json(p["bindings"], build_bindings(p, identities))
    write_json(state / "STATUS.json", {
        "status": "PREPARED__NO_RELEASE_WRITE",
        "prepared_utc": utc_now(),
        "formal_prediction_pairs": 6,
        "training_executed": False,
        "inference_executed": False,
        "gpu_used": False,
        "release_write": False,
    })
    count = rebuild_manifest(state)
    print("===== SORTING-PROPOSAL BUDGET PACKAGE PREPARED =====")
    print("State package   : %s" % state)
    print("Formal inputs   : 6 prediction lists + Test GT / SHA-bound")
    print("Package files   : %d bound" % count)
    print("Release write   : NO")
    print("PREPARE: PASS")


def verify_external_bindings(p: Dict[str, Path]) -> None:
    bindings = read_json(p["bindings"])
    require(bindings.get("release_write") is False, "Input binding unexpectedly permits release writes")
    for key in ("main_manifest", "prediction_package_manifest", "test_annotation"):
        verify_file_record(bindings[key], p["release"])
    require(set(bindings.get("predictions", {})) == {"%s_seed%d" % (m, s) for m in METHODS for s in SEEDS}, "Bound prediction grid is incomplete")
    for row in bindings["predictions"].values():
        verify_file_record(row["prediction"], p["release"])
        verify_file_record(row["summary"], p["release"])


def safe_clear_output_directory(p: Dict[str, Path], name: str) -> Path:
    target = p["state"] / name
    state = p["state"].resolve()
    if target.exists():
        resolved = target.resolve()
        require(state in resolved.parents and resolved.name == name, "Unsafe staged-output cleanup target")
        shutil.rmtree(resolved)
    target.mkdir(parents=True, exist_ok=True)
    return target


def run(project: Path) -> None:
    p, _identities = validate_inputs(project)
    if not p["bindings"].is_file() or not p["state_protocol"].is_file():
        prepare(project)
    verify_external_bindings(p)
    safe_clear_output_directory(p, "results")
    report = run_evaluation(
        p["annotation"], p["prediction_root"], p["state"], sha256(p["state_protocol"])
    )
    write_json(p["state"] / "STATUS.json", {
        "status": report["status"],
        "completed_utc": utc_now(),
        "formal_prediction_pairs": 6,
        "training_executed": False,
        "inference_executed": False,
        "gpu_used": False,
        "release_write": False,
    })
    rebuild_manifest(p["state"])
    verify(project)
    print("CDW1630 SORTING-PROPOSAL BUDGET EVALUATION: PASS")


def verify_result_payload(report: Dict[str, Any]) -> None:
    require(report.get("status") == "COMPLETE__OFFLINE_BUDGETED_SORTING_PROPOSAL_EVALUATION__STAGED_ONLY", "Result status mismatch")
    require(report.get("methods") == list(METHODS), "Result method list changed")
    require(report.get("detector_seeds") == list(SEEDS), "Result seed list changed")
    require(report.get("iou_thresholds") == list(IOU_THRESHOLDS), "Result IoU thresholds changed")
    require(report.get("action_budgets") == list(BUDGETS), "Result action budgets changed")
    require(report.get("main_setting") == {"tau": MAIN_IOU, "K": MAIN_BUDGET}, "Main setting changed")
    require(report.get("training_executed") is False and report.get("inference_executed") is False, "Unexpected training or inference")
    require(report.get("gpu_used") is False and report.get("release_write") is False, "Unexpected GPU or release write")
    require(report.get("test_used_for_reporting") is True, "Test reporting role is missing")
    require(report.get("test_used_for_model_checkpoint_threshold_parameter_or_protocol_selection") is False, "Result claims Test selection")
    annotation = report.get("annotation", {})
    require(int(annotation.get("image_count", -1)) == EXPECTED_TEST_IMAGES, "Result image count mismatch")
    require(int(annotation.get("noncrowd_gt_count", 0)) > 0, "Result GT denominator is invalid")
    audits = report.get("prediction_audit", {})
    require(len(audits) == len(METHODS) * len(SEEDS), "Prediction audit is incomplete")
    for key, row in audits.items():
        require(int(row.get("candidates_per_image_min", -1)) == EXPECTED_CANDIDATES_PER_IMAGE, "%s minimum candidate count changed" % key)
        require(int(row.get("candidates_per_image_max", -1)) == EXPECTED_CANDIDATES_PER_IMAGE, "%s maximum candidate count changed" % key)
        require(int(row.get("exact_duplicate_record_count", -1)) == 0, "%s contains exact duplicate records" % key)
        require(row.get("bbox_bounds_check") == "AUDITED__PRESERVED_UNCHANGED", "%s bbox audit policy changed" % key)
        require(int(row.get("out_of_bounds_box_count", -1)) >= 0, "%s bbox audit count is invalid" % key)
        require(float(row.get("maximum_outside_excursion_px", -1.0)) >= 0.0, "%s bbox excursion audit is invalid" % key)
        require(int(row.get("degenerate_box_count", -1)) >= 0, "%s degenerate-box audit count is invalid" % key)
        require(int(row.get("nonpositive_width_box_count", -1)) >= 0, "%s width audit count is invalid" % key)
        require(int(row.get("nonpositive_height_box_count", -1)) >= 0, "%s height audit count is invalid" % key)
        require(row.get("degenerate_bbox_policy") == "retained in the score queue, assigned IoU zero, and counted as an other-invalid background action", "%s degenerate-box policy changed" % key)
    rows = report.get("per_seed_rows", [])
    require(len(rows) == len(METHODS) * len(SEEDS) * len(IOU_THRESHOLDS) * len(BUDGETS), "Per-seed result grid is incomplete")
    for row in rows:
        executed = int(row["executed_actions"])
        success = int(row["successful_actions"])
        duplicate = int(row["duplicate_actions"])
        other = int(row["other_invalid_actions"])
        require(executed == EXPECTED_TEST_IMAGES * int(row["K"]), "Executed-action denominator mismatch")
        require(success + duplicate + other == executed, "Action partition mismatch")
        require(abs(float(row["valid_action_rate"]) + float(row["wasted_action_rate"]) - 1.0) <= 1.0e-12, "Rate complement mismatch")
        for metric in RATE_METRICS:
            require(math.isfinite(float(row[metric])) and 0.0 <= float(row[metric]) <= 1.0, "Invalid rate: %s" % metric)
    summary = report.get("three_seed_summary_rows", [])
    require(len(summary) == len(IOU_THRESHOLDS) * len(BUDGETS) * len(RATE_METRICS), "Three-seed summary grid is incomplete")


def verify(project: Path) -> None:
    p, _identities = validate_inputs(project)
    require(p["state"].is_dir(), "Staged package is missing; run --prepare")
    require(p["bindings"].is_file() and p["state_protocol"].is_file(), "Staged protocol or bindings are missing")
    verify_external_bindings(p)
    protocol = read_json(p["state_protocol"])
    require(protocol.get("release_write") is False, "Protocol release-write boundary changed")
    require(protocol.get("main_setting") == {"K": MAIN_BUDGET, "iou_threshold": MAIN_IOU, "selection_basis": "Train-set median GT count"}, "Locked main setting changed")
    report = read_json(p["results"])
    require(report.get("protocol_sha256") == sha256(p["state_protocol"]), "Result/protocol SHA mismatch")
    verify_result_payload(report)
    required = (
        "results/PER_SEED_METRICS.csv",
        "results/PER_IMAGE_COUNTS.csv",
        "results/THREE_SEED_SUMMARY.csv",
        "results/MAIN_TABLE_K5_TAU075.csv",
        "results/INTERPRETATION.md",
    )
    for relative in required:
        path = p["state"] / relative
        require(path.is_file() and path.stat().st_size > 0, "Missing staged output: %s" % relative)
    count = verify_manifest(p["state"])
    print("===== CDW1630 SORTING-PROPOSAL BUDGET VERIFICATION =====")
    print("Formal inputs    : PASS / 6 Test prediction lists")
    print("Image coverage   : PASS / 237 Test images")
    print("Candidate queues : PASS / 300 per image / no score threshold")
    print("Box coordinates  : AUDITED / exported values preserved unchanged")
    print("Degenerate boxes : RETAINED / IoU=0 / other-invalid background")
    print("Matching protocol: PASS / same class / non-crowd / one-to-one")
    print("Metrics          : PASS / target, valid, duplicate, other-invalid, wasted")
    print("Three-seed report: PASS / mean, sample SD, paired deltas")
    print("GPU/train/infer  : NOT USED / NOT EXECUTED / NOT EXECUTED")
    print("Release write    : NO")
    print("Package manifest : PASS / %d files" % count)
    print("VERIFICATION: PASS")


def summary(project: Path) -> None:
    p = paths(project)
    report = read_json(p["results"])
    verify_result_payload(report)
    rows = report["three_seed_summary_rows"]
    print("===== CDW1630 BUDGETED SORTING-PROPOSAL SUMMARY =====")
    for tau in IOU_THRESHOLDS:
        print("tau=%.2f / K=%d" % (tau, MAIN_BUDGET))
        for metric in RATE_METRICS:
            row = next(item for item in rows if item["tau"] == tau and item["K"] == MAIN_BUDGET and item["metric"] == metric)
            print(
                "  %-23s DEIM=%6.3f +/- %6.3f %%  HiLoR-DETR=%6.3f +/- %6.3f %%  delta=%+6.3f +/- %6.3f pp"
                % (
                    metric,
                    100.0 * float(row["B0_mean"]),
                    100.0 * float(row["B0_sample_std"]),
                    100.0 * float(row["Full_mean"]),
                    100.0 * float(row["Full_sample_std"]),
                    100.0 * float(row["paired_delta_mean_Full_minus_B0"]),
                    100.0 * float(row["paired_delta_sample_std"]),
                )
            )
    print("Interpretation   : offline detection-driven sorting-proposal proxy only")
    print("Release installed: NO")


def status(project: Path) -> None:
    p = paths(project)
    print("===== SORTING-PROPOSAL BUDGET PACKAGE STATUS =====")
    print("Prepared :", p["bindings"].is_file() and p["state_protocol"].is_file())
    print("Complete :", p["results"].is_file())
    print("Location :", p["state"])
    print("Installed:", "NO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "run", "verify", "summary", "status"))
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/BTC_DEIM_project"))
    args = parser.parse_args()
    project = args.project.resolve()
    actions = {
        "preflight": lambda: preflight(project),
        "prepare": lambda: prepare(project),
        "run": lambda: run(project),
        "verify": lambda: verify(project),
        "summary": lambda: summary(project),
        "status": lambda: status(project),
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
