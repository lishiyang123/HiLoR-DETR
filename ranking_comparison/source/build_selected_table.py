#!/usr/bin/env python3
"""Build a concise table using the recorded Val-selected Soft-NMS settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("AP", "AP50", "AP75", "AP85", "AP90", "AP95", "AR100")


def metric(row, name):
    metrics = row["metrics"]
    return metrics.get(name, metrics.get("threshold_ap", {}).get(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--soft-nms-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.aggregate.read_text(encoding="utf-8"))["rows"]
    selection = json.loads(args.soft_nms_selection.read_text(encoding="utf-8"))
    chosen = {
        method: float(decision["selected"]["grid_value"])
        for method, decision in selection["methods"].items()
    }
    selected = []
    for row in rows:
        method = row["method"]
        if method == "soft_nms_linear":
            if float(row["parameters"]["iou_threshold"]) != chosen[method]:
                continue
        elif method == "soft_nms_gaussian":
            if float(row["parameters"]["sigma"]) != chosen[method]:
                continue
        selected.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_results.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "soft_nms_selection": str(args.soft_nms_selection.resolve()),
                "rows": selected,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    columns = ["method", "setting", *METRICS, "delta_AP_vs_native_pp"]
    with (args.output_dir / "selected_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "method": row["method"],
                    "setting": row["setting"],
                    **{name: metric(row, name) for name in METRICS},
                    "delta_AP_vs_native_pp": row["delta_AP_vs_native_pp"],
                }
            )

    header = ["Method", "Setting", *METRICS, "Delta AP"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in selected:
        lines.append(
            "| " + " | ".join(
                [
                    row["method"], row["setting"],
                    *(f"{float(metric(row, name)):.4f}" for name in METRICS),
                    f"{float(row['delta_AP_vs_native_pp']):+.4f}",
                ]
            ) + " |"
        )
    (args.output_dir / "selected_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"SELECTED RESULT TABLE: PASS / {len(selected)} rows")


if __name__ == "__main__":
    main()

