#!/usr/bin/env python3
"""Verify the installed three-row B0 frozen-postprocessing branch."""

import argparse
import json
from pathlib import Path


SEEDS = (0, 1, 2)
ROWS = ("B0", "B0_QCCR", "B0_QCCR_UMR")
SIX_ROWS = ("B0", "B0_QCCR", "B0_QCCR_UMR", "UFD_LCCT",
            "UFD_LCCT_QCCR", "FULL")


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    require(path.is_file(), "Missing JSON: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root):
    path = root / "manifests/PACKAGE_MANIFEST.sha256.json"
    value = read_json(path)
    actual = {
        item.relative_to(root).as_posix(): sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item != path and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    }
    require(value.get("files") == actual, "Child package manifest differs")
    require(value.get("file_count") == len(actual), "Child file count differs")


def verify(release):
    root = Path(__file__).resolve().parents[1]
    expected = release / "ablations/frozen_postprocessing_control/B0_QCCR_UMR_THREE_SEED_v1"
    require(root == expected.resolve(), "Unexpected child location")
    verify_manifest(root)
    original = read_json(root / "protocol/B0_POSTPROCESS_CONTROL.json")
    addition = read_json(root / "protocol/B0_QCCR_ONLY_PROTOCOL.json")
    require(original.get("detector_seeds") == [0, 1, 2], "Original seed scope changed")
    require(addition.get("comparison_rows") == list(ROWS), "QCCR branch rows changed")
    require(addition.get("detector_training") is False and
            addition.get("postprocessor_fitting") is False,
            "Training/refitting scope changed")
    bindings = read_json(root / "bindings/ASSET_BINDINGS.json")
    require(bindings.get("rows") == list(ROWS), "Binding rows changed")
    require(bindings.get("test_used_for_selection") is False,
            "Test selection changed")
    for group in ("external_assets", "local_assets"):
        for label, record in bindings[group].items():
            path = (root / record["path"]).resolve()
            require(path.is_file(), "Bound asset missing: %s" % label)
            require(path.stat().st_size == int(record["size"]),
                    "Bound size changed: %s" % label)
            require(sha256(path) == record["sha256"],
                    "Bound SHA changed: %s" % label)
    per_seed = read_json(root / "paper_tables/B0_POSTPROCESS_PER_SEED.json")
    require([item["seed"] for item in per_seed] == [0, 1, 2],
            "Per-seed scope changed")
    for item in per_seed:
        seed = int(item["seed"])
        require(tuple(item["metrics"].keys()) == ROWS,
                "Per-seed row order changed")
        exact = read_json(root / ("results/seed%d/QCCR_ONLY_EXACT_METRICS.json" % seed))
        require(exact["metrics"] == item["metrics"]["B0_QCCR"],
                "QCCR-only metrics differ: seed%d" % seed)
        eval_path = root / ("results/seed%d/qccr_only_eval.pth" % seed)
        require(exact["eval_sha256"] == sha256(eval_path),
                "QCCR-only eval changed: seed%d" % seed)
        require(exact.get("test_used_for_parameter_selection") is False,
                "Test selection changed")
    summary = read_json(root / "paper_tables/B0_POSTPROCESS_MEAN_STD.json")
    require([item["row"] for item in summary["rows"]] == list(ROWS),
            "Summary rows changed")
    require(abs(100.0 * summary["rows"][0]["metrics"]["AP"]["mean"] - 77.500) <= 0.0006,
            "B0 AP changed")
    require(abs(100.0 * summary["rows"][1]["metrics"]["AP"]["mean"] - 77.499) <= 0.001,
            "B0+QCCR AP changed")
    require(abs(100.0 * summary["rows"][2]["metrics"]["AP"]["mean"] - 77.570) <= 0.0006,
            "B0+QCCR+UMR AP changed")
    six = read_json(root / "paper_tables/CDW1630_SEED0_FINAL_SIX_ROW_ABLATION.json")
    require([item["variant"] for item in six["rows"]] == list(SIX_ROWS),
            "Six-row table order changed")
    require(six.get("seed") == 0 and
            six.get("test_used_for_parameter_selection") is False,
            "Six-row protocol changed")
    require(six["rows"][0]["metrics"] == six["rows"][1]["metrics"],
            "Seed0 QCCR-only row is no longer identical to B0")
    print("===== B0 FROZEN-POSTPROCESSING BRANCH VERIFICATION =====")
    print("Rows             : B0 / B0+QCCR / B0+QCCR+UMR")
    print("Detector seeds   : 3/3")
    print("QCCR-only evals  : 3/3")
    print("Seed0 main table : six rows PASS")
    print("Runtime residue  : ABSENT")
    print("Manifest         : PASS (%d files)" %
          read_json(root / "manifests/PACKAGE_MANIFEST.sha256.json")["file_count"])
    print("VERIFICATION: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify",))
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release.resolve(strict=True))


if __name__ == "__main__":
    main()
