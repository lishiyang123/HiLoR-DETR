#!/usr/bin/env python3
"""Aggregate every disclosed replay result into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("AP", "AP50", "AP75", "AP80", "AP85", "AP90", "AP95", "AR100")


def _metric(metrics: dict[str, Any], name: str):
    if name in metrics:
        return metrics[name]
    return metrics.get("threshold_ap", {}).get(name)


def _rows_from_payload(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    base = {
        "split": payload.get("split", "unknown"),
        "capture_sha256": payload.get("capture_sha256", ""),
        "source": str(source.resolve()),
    }
    if "configurations" in payload:
        rows = []
        for config in payload["configurations"]:
            parameters = {
                "grid_value": config.get("grid_value"),
                "iou_threshold": config.get("iou_threshold"),
                "sigma": config.get("sigma"),
            }
            rows.append(
                {
                    **base,
                    "method": payload["method"],
                    "parameters": parameters,
                    "metrics": config["metrics"],
                    "cpu_replay_latency_ms": config.get("cpu_replay_latency_ms", {}),
                }
            )
        return rows
    if "metrics" not in payload or "method" not in payload:
        return []
    return [
        {
            **base,
            "method": payload["method"],
            "parameters": payload.get("parameters", {}),
            "metrics": payload["metrics"],
            "cpu_replay_latency_ms": payload.get("cpu_replay_latency_ms", {}),
        }
    ]


def _fingerprint(row: dict[str, Any]) -> str:
    parameters = row["parameters"]
    method = row["method"]
    if method == "native":
        parameters = {}
    elif method == "soft_nms_linear":
        parameters = {"iou_threshold": parameters.get("iou_threshold")}
    elif method == "soft_nms_gaussian":
        parameters = {"sigma": parameters.get("sigma")}
    elif method == "gilg_original":
        parameters = {
            "near_duplicate_iou": parameters.get("near_duplicate_iou"),
            "gilg_params": parameters.get("gilg_params"),
        }
    elif method == "iou_reg_rescore":
        parameters = {
            "iou_reg_checkpoint": Path(
                str(parameters.get("iou_reg_checkpoint", ""))
            ).name,
            "iou_reg_metadata": Path(
                str(parameters.get("iou_reg_metadata", ""))
            ).name,
        }
    return json.dumps(
        [row["split"], row["capture_sha256"], method, parameters],
        sort_keys=True,
        separators=(",", ":"),
    )


def _setting(row: dict[str, Any]) -> str:
    parameters = row["parameters"]
    if row["method"] == "soft_nms_linear":
        return f"IoU={float(parameters['iou_threshold']):g}"
    if row["method"] == "soft_nms_gaussian":
        return f"sigma={float(parameters['sigma']):g}"
    if row["method"] == "gilg_original":
        threshold = parameters.get("near_duplicate_iou")
        return "complete pool" if threshold is None else f"near-duplicate IoU={threshold:g}"
    if row["method"] == "iou_reg_rescore":
        return "Train fit / Val BCE selected"
    return "--"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows_by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(args.input_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for row in _rows_from_payload(payload, path):
            rows_by_key[_fingerprint(row)] = row

    rows = list(rows_by_key.values())
    native = {
        (row["split"], row["capture_sha256"]): float(row["metrics"]["AP"])
        for row in rows
        if row["method"] == "native"
    }
    for row in rows:
        baseline = native.get((row["split"], row["capture_sha256"]))
        row["delta_AP_vs_native_pp"] = (
            None if baseline is None else float(row["metrics"]["AP"]) - baseline
        )
        row["setting"] = _setting(row)
    rows.sort(
        key=lambda row: (
            row["split"],
            -float(row["metrics"].get("AP", float("-inf"))),
            row["method"],
            json.dumps(row["parameters"], sort_keys=True),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_results.json").write_text(
        json.dumps({"format_version": 1, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = ["split", "method", "setting", "parameters", *METRICS,
               "delta_AP_vs_native_pp", "cpu_mean_ms", "capture_sha256", "source"]
    with (args.output_dir / "all_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "method": row["method"],
                    "setting": row["setting"],
                    "parameters": json.dumps(row["parameters"], sort_keys=True),
                    **{name: _metric(row["metrics"], name) for name in METRICS},
                    "delta_AP_vs_native_pp": row["delta_AP_vs_native_pp"],
                    "cpu_mean_ms": row["cpu_replay_latency_ms"].get("mean"),
                    "capture_sha256": row["capture_sha256"],
                    "source": row["source"],
                }
            )

    header = ["Split", "Method", "Setting", "AP", "AP50", "AP75", "AP85", "AP90", "AP95", "AR100", "Delta AP"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        metrics = row["metrics"]
        delta = row["delta_AP_vs_native_pp"]
        lines.append(
            "| " + " | ".join(
                [
                    str(row["split"]), str(row["method"]), str(row["setting"]),
                    *(f"{float(_metric(metrics, name)):.4f}" for name in
                      ("AP", "AP50", "AP75", "AP85", "AP90", "AP95", "AR100")),
                    "--" if delta is None else f"{delta:+.4f}",
                ]
            ) + " |"
        )
    (args.output_dir / "all_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT AGGREGATION: PASS / {len(rows)} disclosed configurations")


if __name__ == "__main__":
    main()
