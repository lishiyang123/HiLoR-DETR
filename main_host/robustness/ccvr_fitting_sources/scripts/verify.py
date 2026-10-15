#!/usr/bin/env python3
"""Verify the installed CCVR candidate-value fitting-source analysis."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


TARGET_RELATIVE = Path(
    "main_host/robustness/ccvr_fitting_sources"
)
SEEDS = (0, 1, 2)
ROWS = (
    "qccr_only",
    "current_train_capture_umr",
    "val_oof_umr",
    "train_matched_umr",
)
EXPECTED_DELTA_AP_PP = {
    "qccr_only": 0.000,
    "current_train_capture_umr": 0.207,
    "val_oof_umr": 0.105,
    "train_matched_umr": 0.111,
}
EXPECTED_DELTA_AP75_PP = {
    "qccr_only": 0.000,
    "current_train_capture_umr": 0.244,
    "val_oof_umr": 0.085,
    "train_matched_umr": 0.097,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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


def verify_manifest(child):
    manifest_path = child / "manifests/PACKAGE_MANIFEST.sha256.json"
    payload = read_json(manifest_path)
    observed = payload.get("files")
    require(isinstance(observed, dict), "Package manifest files must be a mapping")
    expected = []
    for path in sorted(child.rglob("*"), key=lambda item: item.relative_to(child).as_posix()):
        require(not path.is_symlink(), "Symbolic link is forbidden: %s" % path)
        relative = path.relative_to(child).as_posix()
        require("__pycache__" not in path.parts and path.suffix != ".pyc",
                "Python cache is forbidden: %s" % relative)
        if path.is_file() and path != manifest_path:
            expected.append(relative)
    require(sorted(observed) == expected, "Package manifest file-set mismatch")
    for relative in expected:
        require(observed[relative] == sha256_file(child / relative),
                "Package manifest SHA mismatch: %s" % relative)
    require(payload.get("file_count") == len(expected), "Package manifest count mismatch")
    return len(expected)


def safe_external(child, release, record, label):
    relative = record.get("path")
    require(isinstance(relative, str) and relative, "%s path is empty" % label)
    require(not Path(relative).is_absolute(), "%s path is absolute" % label)
    path = (child / relative).resolve()
    require(release == path or release in path.parents,
            "%s path escaped final release: %s" % (label, relative))
    require(path.is_file(), "%s is missing: %s" % (label, path))
    require(sha256_file(path) == record.get("sha256"),
            "%s SHA-256 changed" % label)


def close(left, right, tolerance=1.1e-3):
    return abs(float(left) - float(right)) <= tolerance


def validate_summary(summary):
    require(summary.get("detector_seeds") == [0, 1, 2], "Detector seed scope changed")
    require(summary.get("fold_count") == 5, "Fold count changed")
    require(summary.get("rows") == list(ROWS), "Row scope changed")
    require(summary.get("test_accessed") is False, "Summary indicates Test access")
    require(summary.get("detector_retrained") is False,
            "Summary indicates detector retraining")
    require(summary.get("no_result_based_selection") is True,
            "Summary no-selection flag changed")
    results = summary.get("results")
    require([record.get("row") for record in results] == list(ROWS),
            "Summary result order changed")
    for record in results:
        row = record["row"]
        for metric, expected in (("AP", EXPECTED_DELTA_AP_PP[row]),
                                 ("AP75", EXPECTED_DELTA_AP75_PP[row])):
            block = record["delta_vs_qccr_only_pp"][metric]
            values = block["per_seed"]
            require(len(values) == 3, "%s/%s per-seed scope changed" % (row, metric))
            require(close(block["mean"], expected),
                    "%s/%s mean changed: %s" % (row, metric, block["mean"]))
            if row != "qccr_only":
                require(all(float(value) > 0.0 for value in values),
                        "%s/%s is not positive for every seed" % (row, metric))
        ar = record["delta_vs_qccr_only_pp"]["AR100"]["per_seed"]
        require(all(abs(float(value)) <= 1.0e-10 for value in ar),
                "%s AR100 behavior changed" % row)
    return results


def validate_seed(child, seed):
    seed_dir = child / ("seed%d" % seed)
    payload = read_json(seed_dir / "OOF_METRICS.json")
    require(payload.get("detector_seed") == seed, "Seed identity changed: %d" % seed)
    require(payload.get("rows") == list(ROWS), "Seed row scope changed: %d" % seed)
    require(payload.get("val_image_count") == 249, "Val image count changed: %d" % seed)
    require(payload.get("detections_per_row") == 249 * 300,
            "Detection count changed: %d" % seed)
    require(payload.get("test_accessed") is False, "Seed indicates Test access: %d" % seed)
    require(payload.get("detector_retrained") is False,
            "Seed indicates detector retraining: %d" % seed)
    require(payload.get("no_result_based_selection") is True,
            "Seed selection flag changed: %d" % seed)
    require(abs(float(payload.get("qccr_lambda_reg")) - 0.001) < 1.0e-12,
            "QCCR lambda changed: %d" % seed)
    require(payload.get("umr_tuple") == {
        "positive_weight": 4, "tau": 0.0, "beta": 2.0,
        "topk": 300, "head_seed": 0,
    }, "Fixed UMR tuple changed: %d" % seed)
    folds = payload.get("fold_reports")
    require([record.get("fold") for record in folds] == [0, 1, 2, 3, 4],
            "Fold reports changed: %d" % seed)
    for record in folds:
        require(record.get("target_fold_used_for_fitting") is False,
                "Target fold entered fitting: seed%d/fold%d" % (seed, record["fold"]))
        target = int(record["target_image_count"])
        outer = int(record["outer_val_fit_image_count"])
        matched = int(record["matched_train_image_count"])
        require(target + outer == 249 and matched == outer,
                "Fold sample-size control changed: seed%d/fold%d" % (seed, record["fold"]))
        for key in ("val_oof_head", "train_matched_head"):
            head = record[key]
            relative = head["path"]
            require(relative.startswith("seed%d/heads/" % seed),
                    "Unexpected head path: %s" % relative)
            path = child / relative
            require(path.is_file() and sha256_file(path) == head["sha256"],
                    "Fold head binding changed: %s" % relative)
    heads = list((seed_dir / "heads").glob("*.pt"))
    require(len(heads) == 10, "Expected ten fold heads for seed%d" % seed)
    return payload


def verify(release_value):
    release = resolve_release(release_value)
    child = release / TARGET_RELATIVE
    require(child.is_dir(), "UMR Val-OOF child package missing: %s" % child)
    count = verify_manifest(child)
    protocol = read_json(child / "protocol/FITTING_SOURCE_PROTOCOL.json")
    require(protocol.get("detector_seeds") == [0, 1, 2], "Protocol seeds changed")
    require(protocol.get("fold_count") == 5 and protocol.get("rows") == list(ROWS),
            "Protocol scope changed")
    require(protocol.get("outcome_policy", {}).get("no_parameter_selection") is True,
            "Protocol outcome policy changed")
    summary = read_json(child / "results/THREE_SEED_SUMMARY.json")
    validate_summary(summary)
    for seed in SEEDS:
        validate_seed(child, seed)

    binding = read_json(child / "bindings/ASSET_BINDINGS.json")
    require(binding.get("schema") == "cdw1630-umr-val-oof-robustness-assets-v1",
            "Binding schema changed")
    for label, record in binding.get("release_assets", {}).items():
        safe_external(child, release, record, label)
    captures = binding.get("capture_inputs", {})
    require(len(captures) == 6, "Capture hash scope changed")
    for label, record in captures.items():
        require(record.get("included") is False, "Capture unexpectedly duplicated: %s" % label)
        require(re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is not None,
                "Capture SHA is malformed: %s" % label)

    required = (
        "README.md", "scripts/common.py", "scripts/run_analysis.py",
        "scripts/verify.py", "scripts/build_inputs.py",
    )
    for relative in required:
        require((child / relative).is_file(), "Required compact asset missing: %s" % relative)
    forbidden = []
    for path in child.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if (lower.endswith(".csv") or lower in (
                "input_provenance.json", "status.json", "queue.json",
                "supervisor.log", "console.log", "run_server.sh", "manager.py",
                "self_test.py")):
            forbidden.append(path.relative_to(child).as_posix())
    require(not forbidden, "Redundant/runtime files entered child package: %s" % forbidden)

    print("===== CCVR FITTING-SOURCE VERIFICATION =====")
    print("Detector seeds : 3/3")
    print("OOF folds      : 5 per seed")
    print("Rows           : mapping-only / Train captures / Val-OOF / matched Train")
    print("Val-OOF dAP    : +0.105 +/- 0.034 pp / all three seeds positive")
    print("Matched dAP    : +0.111 +/- 0.050 pp / all three seeds positive")
    print("Fold heads     : 30/30 included and SHA-bound")
    print("Duplicate CSV  : ABSENT")
    print("Runtime residue: ABSENT")
    print("Test access    : NO")
    print("Manifest       : PASS (%d files)" % count)
    print("VERIFICATION: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify",))
    parser.add_argument("--release", default=os.getcwd())
    args = parser.parse_args()
    verify(args.release)


if __name__ == "__main__":
    main()
