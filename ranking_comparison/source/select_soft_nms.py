#!/usr/bin/env python3
"""Select one disclosed Val configuration per Soft-NMS variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.sweeps]
    splits = {payload["split"] for payload in payloads}
    captures = {payload["capture_sha256"] for payload in payloads}
    if splits != {"val"}:
        raise RuntimeError(f"Soft-NMS parameters must be selected on Val, got {splits}")
    if len(captures) != 1:
        raise RuntimeError("all Soft-NMS sweeps must use the same frozen capture")

    by_method: dict[str, list[dict]] = {}
    sources: dict[str, list[str]] = {}
    for path, payload in zip(args.sweeps, payloads):
        method = payload["method"]
        by_method.setdefault(method, []).extend(payload["configurations"])
        sources.setdefault(method, []).append(str(path.resolve()))

    selected = {}
    for method, configurations in sorted(by_method.items()):
        configurations.sort(key=lambda item: item["grid_value"])
        ranking = sorted(
            range(len(configurations)),
            key=lambda index: (
                -float(configurations[index]["metrics"]["AP"]),
                -float(configurations[index]["metrics"]["AP75"]),
                float(configurations[index]["grid_value"]),
            ),
        )
        selected[method] = {
            "selection_rule": "highest Val AP, then AP75, then smaller grid value",
            "selected_index": ranking[0],
            "selected": configurations[ranking[0]],
            "all_configurations": configurations,
            "source_sweeps": sources[method],
        }

    result = {
        "format_version": 1,
        "split": "val",
        "capture_sha256": next(iter(captures)),
        "test_used_for_parameter_selection": False,
        "methods": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for method, decision in selected.items():
        item = decision["selected"]
        print(f"{method}: value={item['grid_value']:g} AP={item['metrics']['AP']:.6f}")
    print("SOFT-NMS VAL SELECTION: PASS")


if __name__ == "__main__":
    main()

