#!/usr/bin/env python3
"""Extract class-wise COCO metrics from the six frozen CDW1630 eval files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


METHODS = {
    "DEIM": "results/CDW1630/B0/seed{seed}/official_test/eval.pth",
    "HiLoR-DETR": "results/CDW1630/Ours/seed{seed}/final/eval.pth",
}
SEEDS = (0, 1, 2)
METRICS = ("AP", "AP75", "AR100")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_valid(values: np.ndarray, label: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1.0]
    if valid.size == 0:
        raise RuntimeError(f"{label} contains no valid COCO entries")
    return float(valid.mean())


def index_exact(values, target, label: str) -> int:
    array = np.asarray(values)
    matches = np.flatnonzero(np.isclose(array.astype(float), float(target)))
    if matches.size != 1:
        raise RuntimeError(f"expected one {label}={target}, found {matches.size}")
    return int(matches[0])


def extract_eval(path: Path, categories: list[dict]) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"params", "precision", "recall"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"invalid COCO eval payload: {path}")
    params = payload["params"]
    precision = np.asarray(payload["precision"])
    recall = np.asarray(payload["recall"])
    if precision.ndim != 5 or recall.ndim != 4:
        raise RuntimeError(f"unexpected precision/recall geometry: {path}")

    cat_ids = [int(value) for value in params.catIds]
    expected_ids = [int(category["id"]) for category in categories]
    if cat_ids != expected_ids:
        raise RuntimeError(
            f"category order mismatch in {path}: eval={cat_ids}, annotation={expected_ids}"
        )
    area_index = list(params.areaRngLbl).index("all")
    max_det_index = list(params.maxDets).index(100)
    iou75_index = index_exact(params.iouThrs, 0.75, "IoU threshold")

    rows = []
    for class_index, category in enumerate(categories):
        ap = mean_valid(
            precision[:, :, class_index, area_index, max_det_index], "AP"
        )
        ap75 = mean_valid(
            precision[iou75_index, :, class_index, area_index, max_det_index],
            "AP75",
        )
        ar100 = mean_valid(
            recall[:, class_index, area_index, max_det_index], "AR100"
        )
        rows.append(
            {
                "category_id": int(category["id"]),
                "category_name": str(category["display_name"]),
                "AP": 100.0 * ap,
                "AP75": 100.0 * ap75,
                "AR100": 100.0 * ar100,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/CDW1630/classwise_metrics"),
    )
    args = parser.parse_args()

    root = args.package_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    expected_output = (root / "results/CDW1630/classwise_metrics").resolve()
    if output != expected_output:
        raise RuntimeError(f"output must be {expected_output}")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")

    annotation_path = root / "datasets/CDW1630/annotations/instances_test2017.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(annotation["categories"], key=lambda item: int(item["id"]))
    for category in categories:
        name = str(category["name"])
        category["display_name"] = "Timber" if name == "timber" else name
    counts = Counter(int(item["category_id"]) for item in annotation["annotations"])
    if sum(counts.values()) != 1071:
        raise RuntimeError("unexpected Test annotation count")

    input_records = [
        {
            "role": "Test annotation",
            "path": annotation_path.relative_to(root).as_posix(),
            "sha256": sha256_file(annotation_path),
            "size": annotation_path.stat().st_size,
        }
    ]
    per_seed = []
    for method, pattern in METHODS.items():
        for seed in SEEDS:
            path = root / pattern.format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            input_records.append(
                {
                    "role": f"{method} seed {seed} COCO evaluation",
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
            for metrics in extract_eval(path, categories):
                per_seed.append(
                    {
                        "method": method,
                        "seed": seed,
                        "category_id": metrics["category_id"],
                        "category_name": metrics["category_name"],
                        "instance_count": counts[metrics["category_id"]],
                        **{name: f"{metrics[name]:.6f}" for name in METRICS},
                    }
                )

    count_rows = [
        {
            "category_id": int(category["id"]),
            "category_name": category["display_name"],
            "instance_count": counts[int(category["id"])],
        }
        for category in categories
    ]
    count_rows.append(
        {"category_id": "all", "category_name": "All", "instance_count": 1071}
    )

    summary_rows = []
    for category in categories:
        category_id = int(category["id"])
        for metric in METRICS:
            values = {}
            for method in METHODS:
                values[method] = np.asarray(
                    [
                        float(row[metric])
                        for row in per_seed
                        if row["method"] == method
                        and int(row["category_id"]) == category_id
                    ],
                    dtype=np.float64,
                )
                if values[method].size != len(SEEDS):
                    raise RuntimeError("incomplete three-seed aggregation")
            delta = values["HiLoR-DETR"] - values["DEIM"]
            summary_rows.append(
                {
                    "category_id": category_id,
                    "category_name": category["display_name"],
                    "instance_count": counts[category_id],
                    "metric": metric,
                    "DEIM_mean_percent": f"{values['DEIM'].mean():.6f}",
                    "DEIM_sample_sd_percent": f"{values['DEIM'].std(ddof=1):.6f}",
                    "HiLoR_DETR_mean_percent": f"{values['HiLoR-DETR'].mean():.6f}",
                    "HiLoR_DETR_sample_sd_percent": f"{values['HiLoR-DETR'].std(ddof=1):.6f}",
                    "paired_delta_mean_pp": f"{delta.mean():.6f}",
                    "paired_delta_sample_sd_pp": f"{delta.std(ddof=1):.6f}",
                }
            )

    output.mkdir(parents=True)
    write_csv(
        output / "class_instance_counts.csv",
        count_rows,
        ["category_id", "category_name", "instance_count"],
    )
    write_csv(
        output / "classwise_per_seed.csv",
        per_seed,
        [
            "method",
            "seed",
            "category_id",
            "category_name",
            "instance_count",
            *METRICS,
        ],
    )
    write_csv(
        output / "classwise_mean_std.csv",
        summary_rows,
        [
            "category_id",
            "category_name",
            "instance_count",
            "metric",
            "DEIM_mean_percent",
            "DEIM_sample_sd_percent",
            "HiLoR_DETR_mean_percent",
            "HiLoR_DETR_sample_sd_percent",
            "paired_delta_mean_pp",
            "paired_delta_sample_sd_pp",
        ],
    )
    manifest = {
        "schema": "cdw1630-classwise-metrics-v1",
        "dataset": "CDW1630 Test",
        "image_count": len(annotation["images"]),
        "annotation_count": len(annotation["annotations"]),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "units": {
            "AP": "percent",
            "AP75": "percent",
            "AR100": "percent",
            "paired_delta": "percentage points",
        },
        "definitions": {
            "AP": "mean precision over IoU=0.50:0.95, area=all, maxDets=100",
            "AP75": "mean precision at IoU=0.75, area=all, maxDets=100",
            "AR100": "mean recall over IoU=0.50:0.95, area=all, maxDets=100",
            "sample_sd": "three-seed sample standard deviation (ddof=1)",
            "paired_delta": "HiLoR-DETR minus DEIM for matching detector seeds",
        },
        "model_training_or_inference_rerun": False,
        "inputs": input_records,
        "outputs": [
            "results/CDW1630/classwise_metrics/class_instance_counts.csv",
            "results/CDW1630/classwise_metrics/classwise_per_seed.csv",
            "results/CDW1630/classwise_metrics/classwise_mean_std.csv",
        ],
    }
    (output / "CLASSWISE_METRICS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("CDW1630 CLASS-WISE METRICS: PASS")
    print(f"Inputs: {len(input_records)} frozen files")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
