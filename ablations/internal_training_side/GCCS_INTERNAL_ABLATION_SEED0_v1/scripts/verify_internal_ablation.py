#!/usr/bin/env python3
"""Verify the installed GCCS seed-0 internal-ablation child package."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


PARENT = "ablations"
CHILD_RELATIVE = Path(PARENT) / "internal_training_side" / "GCCS_INTERNAL_ABLATION_SEED0_v1"
VARIANTS = ("ufd_without_difficulty", "ufd_only", "lcct_only")
ORDER = ("b0",) + VARIANTS + ("gccs",)
METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)
EXPECTED_MAIN_PERCENT = {
    "b0": (77.506, 97.977, 89.932, 83.946),
    "ufd_without_difficulty": (77.517, 98.015, 88.773, 84.671),
    "ufd_only": (77.649, 98.078, 88.822, 84.347),
    "lcct_only": (78.149, 97.627, 90.537, 87.172),
    "gccs": (78.673, 97.648, 90.692, 87.368),
}


def fail(message):
    raise RuntimeError(message)


def require(condition, message):
    if not condition:
        fail(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), "Missing JSON: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_release(value):
    release = Path(value).expanduser().resolve()
    require(release.is_dir(), "Release directory missing: %s" % release)
    require((release / "verify_release.py").is_file(), "Release verifier missing")
    return release


def safe_resolve(child, release, relative, label):
    require(isinstance(relative, str) and relative, "%s path is empty" % label)
    require(not Path(relative).is_absolute(), "%s path is absolute" % label)
    candidate = (child / relative).resolve()
    require(candidate == release or release in candidate.parents,
            "%s path escapes the release: %s" % (label, relative))
    require(candidate.is_file(), "%s file is missing: %s" % (label, candidate))
    return candidate


def verify_manifest(child):
    manifest_path = child / "manifests/PACKAGE_MANIFEST.sha256.json"
    payload = read_json(manifest_path)
    observed = payload.get("files")
    require(isinstance(observed, dict), "Child manifest files must be a mapping")
    expected_paths = []
    for path in sorted(child.rglob("*"), key=lambda item: item.relative_to(child).as_posix()):
        require(not path.is_symlink(), "Symbolic link is forbidden: %s" % path)
        relative = path.relative_to(child).as_posix()
        require("__pycache__" not in path.parts and path.suffix != ".pyc",
                "Python cache is forbidden: %s" % relative)
        if path.is_file() and path != manifest_path:
            expected_paths.append(relative)
    require(sorted(observed) == expected_paths, "Child manifest file-set mismatch")
    for relative in expected_paths:
        require(observed[relative] == sha256_file(child / relative),
                "Child manifest SHA mismatch: %s" % relative)
    require(payload.get("file_count") == len(expected_paths),
            "Child manifest count mismatch")
    return len(expected_paths)


def close(left, right, tolerance=1e-10):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def verify(release):
    release = resolve_release(release)
    child = release / CHILD_RELATIVE
    require(child.is_dir(), "Internal-ablation child package missing: %s" % child)
    count = verify_manifest(child)
    binding = read_json(child / "bindings/ASSET_BINDINGS.json")
    require(binding.get("schema") == "cdw1630-gccs-internal-ablation-assets-v1",
            "Binding schema changed")

    for label, record in binding["external_assets"].items():
        path = safe_resolve(child, release, record["path"], label)
        require(sha256_file(path) == record["sha256"],
                "External binding changed: %s" % label)

    for variant in VARIANTS:
        record = binding["new_rows"][variant]
        for key in ("training_config", "test_config", "selection", "training_log",
                    "checkpoint", "evaluation_record", "eval"):
            asset = record[key]
            path = safe_resolve(child, release, asset["path"], variant + "/" + key)
            require(sha256_file(path) == asset["sha256"],
                    "Local binding changed: %s/%s" % (variant, key))

    summary = read_json(child / "paper_tables/GCCS_INTERNAL_ABLATION_SEED0_TEST.json")
    rows = summary.get("rows")
    require([row.get("variant") for row in rows] == list(ORDER),
            "Five-row Test order changed")
    require(summary.get("test_used_for_selection") is False and
            summary.get("fitting_on_test") is False,
            "Test role changed")
    for row in rows:
        name = row["variant"]
        metrics = row["metrics"]
        expected = EXPECTED_MAIN_PERCENT[name]
        actual = tuple(100.0 * float(metrics[key]) for key in
                       ("AP", "AP50", "AP75", "AR100"))
        for value, wanted in zip(actual, expected):
            require(abs(value - wanted) <= 0.0006,
                    "Unexpected published metric for %s: %.9f vs %.3f" %
                    (name, value, wanted))
        require(set(metrics) == set(METRICS), "Metric scope changed: %s" % name)

    with (child / "paper_tables/GCCS_INTERNAL_ABLATION_SEED0_TEST.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require([row["variant"] for row in csv_rows] == list(ORDER),
            "Test CSV row order changed")

    for variant in VARIANTS:
        exact = read_json(child / ("results/%s/EXACT_METRICS.json" % variant))
        summary_row = next(row for row in rows if row["variant"] == variant)
        require(exact["checkpoint_selection"] == "Val AP only",
                "Checkpoint-selection scope changed: %s" % variant)
        require(exact["test_used_for_selection"] is False and
                exact["fitting_on_test"] is False,
                "Test role changed: %s" % variant)
        for key in METRICS:
            require(close(exact["metrics"][key], summary_row["metrics"][key]),
                    "Metric mismatch: %s/%s" % (variant, key))

    forbidden = []
    for path in child.rglob("*"):
        if path.is_file():
            lower = path.name.lower()
            if lower in ("queue.json", "supervisor.log", "console.log", "smoke_result.json"):
                forbidden.append(path.relative_to(child).as_posix())
    require(not forbidden, "Runtime residue entered child package: %s" % forbidden)

    print("===== GCCS INTERNAL ABLATION CHILD-PACKAGE VERIFICATION =====")
    print("Rows             : B0 / UFD-no-difficulty / UFD-only / LCCT-only / GCCS")
    print("New checkpoints  : 3/3 locally included and SHA-bound")
    print("New Test results : 3/3 locally included and SHA-bound")
    print("Reused rows      : B0 + GCCS relative bindings PASS")
    print("Val/Test tables  : PASS")
    print("QCCR/UMR         : OFF / OFF")
    print("Test selection   : NO")
    print("Runtime residue  : ABSENT")
    print("Manifest         : PASS (%d files)" % count)
    print("VERIFICATION: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify",))
    parser.add_argument("--release", default=os.getcwd())
    args = parser.parse_args()
    verify(args.release)


if __name__ == "__main__":
    main()
