#!/usr/bin/env python3
"""Bind fixed UMR fusion settings to the new checkpoint's Train-fitted head."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing overwrite: " + str(args.output))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    pw = int(policy["umr_positive_weight"])
    tau = float(policy["umr_tau"])
    beta = float(policy["umr_beta"])
    topk = int(policy["umr_topk"])
    grid = args.candidate_dir / "results/VAL_GRID_MEAN.csv"
    provenance_path = args.candidate_dir / "results/INPUT_PROVENANCE.json"
    if not grid.is_file() or not provenance_path.is_file():
        raise RuntimeError("UMR Train/Val evidence is incomplete")
    rows = list(csv.DictReader(grid.open("r", encoding="utf-8", newline="")))
    matches = [row for row in rows if int(row["pw"]) == pw and abs(float(row["tau"]) - tau) < 1e-12 and abs(float(row["beta"]) - beta) < 1e-12]
    if len(matches) != 1:
        raise RuntimeError("Fixed UMR tuple is absent or duplicated")
    weight = args.candidate_dir / ("models/umr_seed%d_pw%d_best.pt" % (int(policy["umr_head_seed"]), pw))
    if not weight.is_file():
        raise RuntimeError("Fixed UMR Train weight is missing: " + str(weight))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("data_policy", {}).get("test_accessed") is not False:
        raise RuntimeError("UMR fit does not certify Test-free operation")
    if provenance["inputs"]["detector_checkpoint"]["sha256"] != args.checkpoint_sha:
        raise RuntimeError("UMR/checkpoint provenance mismatch")
    row = matches[0]
    decision = {
        "decision": "FIXED_BEFORE_TEST",
        "test_accessed": False,
        "selected": {"pw": pw, "tau": tau, "beta": beta, "topk": topk},
        "selected_model": {
            "head_seed": int(policy["umr_head_seed"]),
            "path": str(weight.resolve()),
            "sha256": sha256(weight),
            "source_checkpoint_sha256": args.checkpoint_sha,
        },
        "val_diagnostic": {key: row.get(key) for key in row},
        "grid": str(grid.resolve()),
        "grid_sha256": sha256(grid),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": sha256(provenance_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("===== FIXED UMR =====")
    print("UMR          : pw=%d tau=%.2f beta=%.2f topk=%d" % (pw, tau, beta, topk))
    print("Weight       : new-checkpoint Train fit / head seed%d" % int(policy["umr_head_seed"]))
    print("Val AP       : %.9f (diagnostic only)" % float(row["AP"]))
    print("Test accessed: NO")
    print("FIXED UMR: PASS")


if __name__ == "__main__":
    main()
