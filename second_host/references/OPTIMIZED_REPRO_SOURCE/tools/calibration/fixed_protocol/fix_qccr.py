#!/usr/bin/env python3
"""Freeze the predeclared QCCR regularization for this checkpoint fit."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing overwrite: " + str(args.output))
    original = json.loads(args.summary.read_text(encoding="utf-8"))
    fits = json.loads(args.fits.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    locked = float(policy["qccr_lambda_reg"])
    candidates = fits["candidates"]
    key = min(candidates, key=lambda item: abs(float(candidates[item]["lambda_reg"]) - locked))
    if abs(float(candidates[key]["lambda_reg"]) - locked) > 1e-12:
        raise RuntimeError("Fixed QCCR lambda is absent from candidate fits")
    if key not in original["candidate_decisions"]:
        raise RuntimeError("Fixed QCCR lambda is absent from OOF evaluation")
    updated = copy.deepcopy(original)
    updated["decision"] = "OOF_VAL_GO"
    updated["selected_candidate_key"] = key
    updated["selected_lambda_reg"] = locked
    updated["selected_full_val_parameters"] = candidates[key]["full_val_fit"]["parameters"]
    updated["selection_policy"] = {
        "name": "predeclared-fixed-hyperparameters",
        "current_test_used": False,
        "current_seed_reselection": False,
        "protocol_lock": str(args.policy.resolve()),
    }
    updated["diagnostic_only_oof"] = original["candidate_decisions"][key]
    updated["test_used_for_policy_or_selection"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    delta = original["candidate_decisions"][key]["deltas_vs_pd"]
    print("===== FIXED QCCR =====")
    print("lambda        :", locked)
    print("Val AP delta  : %+.6f pp (diagnostic only)" % (100.0 * float(delta["AP"])))
    print("Val AP90 delta: %+.6f pp (diagnostic only)" % (100.0 * float(delta["AP90"])))
    print("Test accessed : NO")
    print("FIXED QCCR: PASS")


if __name__ == "__main__":
    main()
