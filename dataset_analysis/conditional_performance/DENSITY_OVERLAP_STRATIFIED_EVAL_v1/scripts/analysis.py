#!/usr/bin/env python3
"""Train-frozen density/overlap grouping and whole-image COCO evaluation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from common import METRICS, METHODS, SEEDS, mean, read_json, require, sample_std, sha256, write_json


def valid_annotations(annotation: Dict[str, Any]) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    image_ids = sorted(int(row["id"]) for row in annotation.get("images", []))
    require(len(image_ids) == len(set(image_ids)), "Image IDs are not unique")
    grouped = {image_id: [] for image_id in image_ids}
    for row in annotation.get("annotations", []):
        image_id = int(row["image_id"])
        require(image_id in grouped, "Annotation references unknown image: %s" % image_id)
        bbox = row.get("bbox", [])
        require(len(bbox) == 4, "Invalid bbox")
        x, y, width, height = [float(value) for value in bbox]
        require(all(math.isfinite(value) for value in (x, y, width, height)), "Non-finite bbox")
        require(width > 0.0 and height > 0.0, "Non-positive bbox")
        grouped[image_id].append(row)
    return image_ids, grouped


def intersection_over_own(box: Iterable[float], other: Iterable[float]) -> float:
    x1, y1, width1, height1 = [float(value) for value in box]
    x2, y2, width2, height2 = [float(value) for value in other]
    intersection_width = max(0.0, min(x1 + width1, x2 + width2) - max(x1, x2))
    intersection_height = max(0.0, min(y1 + height1, y2 + height2) - max(y1, y2))
    return intersection_width * intersection_height / (width1 * height1)


def image_scores(annotation: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    image_ids, grouped = valid_annotations(annotation)
    scores = {}
    for image_id in image_ids:
        rows = grouped[image_id]
        object_scores = []
        for index, row in enumerate(rows):
            competitors = [
                intersection_over_own(row["bbox"], other["bbox"])
                for other_index, other in enumerate(rows)
                if other_index != index
            ]
            object_scores.append(max(competitors) if competitors else 0.0)
        scores[image_id] = {
            "gt_count": len(rows),
            "mean_max_ioa": sum(object_scores) / float(len(object_scores)) if object_scores else 0.0,
        }
    return scores


def freeze_thresholds(train_annotation: Dict[str, Any]) -> Dict[str, Any]:
    scores = image_scores(train_annotation)
    densities = [row["gt_count"] for row in scores.values()]
    overlaps = [row["mean_max_ioa"] for row in scores.values()]
    require(bool(densities), "Train annotation contains no images")
    density = float(statistics.median(densities))
    overlap_median = float(statistics.median(overlaps))
    overlap_q75 = float(statistics.quantiles(overlaps, n=4, method="inclusive")[2])
    if overlap_median > 0.0:
        overlap_threshold = overlap_median
        strategy = "positive_median"
    elif overlap_q75 > 0.0:
        overlap_threshold = overlap_q75
        strategy = "positive_q75_after_zero_median"
    else:
        overlap_threshold = 0.0
        strategy = "zero_versus_positive"
    return {
        "format_version": 1,
        "source_split": "Train",
        "train_image_count": len(scores),
        "density": {
            "threshold": int(density) if density.is_integer() else density,
            "strategy": "median",
            "low_rule": "N_i <= threshold",
            "high_rule": "N_i > threshold",
        },
        "overlap": {
            "threshold": overlap_threshold,
            "strategy": strategy,
            "train_median": overlap_median,
            "train_q75": overlap_q75,
            "zero_score_images": sum(value == 0.0 for value in overlaps),
            "positive_score_images": sum(value > 0.0 for value in overlaps),
            "low_rule": "O_i <= threshold",
            "high_rule": "O_i > threshold",
        },
        "test_metrics_consulted": False,
    }


def same_thresholds(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return (
        left["density"]["threshold"] == right["density"]["threshold"]
        and left["overlap"]["strategy"] == right["overlap"]["strategy"]
        and abs(float(left["overlap"]["threshold"]) - float(right["overlap"]["threshold"])) <= 1.0e-15
    )


def make_groups(test_annotation: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    scores = image_scores(test_annotation)
    density_threshold = float(thresholds["density"]["threshold"])
    overlap_threshold = float(thresholds["overlap"]["threshold"])
    groups = {
        "density_low": [], "density_high": [],
        "overlap_low": [], "overlap_high": [],
    }
    for image_id, row in sorted(scores.items()):
        groups["density_low" if row["gt_count"] <= density_threshold else "density_high"].append(image_id)
        groups["overlap_low" if row["mean_max_ioa"] <= overlap_threshold else "overlap_high"].append(image_id)
    all_ids = set(scores)
    require(set(groups["density_low"]).isdisjoint(groups["density_high"]), "Density groups overlap")
    require(set(groups["density_low"]) | set(groups["density_high"]) == all_ids, "Density partition incomplete")
    require(set(groups["overlap_low"]).isdisjoint(groups["overlap_high"]), "Overlap groups overlap")
    require(set(groups["overlap_low"]) | set(groups["overlap_high"]) == all_ids, "Overlap partition incomplete")
    return {
        "format_version": 1,
        "test_image_count": len(scores),
        "thresholds": thresholds,
        "groups": {
            name: {
                "image_ids": ids,
                "image_count": len(ids),
                "gt_count": sum(int(scores[image_id]["gt_count"]) for image_id in ids),
            }
            for name, ids in groups.items()
        },
        "image_scores": {
            str(image_id): {
                "gt_count": int(row["gt_count"]),
                "mean_max_ioa": float(row["mean_max_ioa"]),
            }
            for image_id, row in sorted(scores.items())
        },
        "selection_policy": "rules and thresholds frozen from Train before subgroup Test metrics",
        "test_used_for_selection": False,
    }


def metric_values(evaluation: Dict[str, Any], params: Any) -> Dict[str, float]:
    precision = np.asarray(evaluation["precision"], dtype=np.float64)
    recall = np.asarray(evaluation["recall"], dtype=np.float64)
    ious = np.asarray(params.iouThrs, dtype=np.float64)
    area = list(params.areaRngLbl).index("all")
    max100 = [int(value) for value in params.maxDets].index(100)

    def average(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[values > -1]
        require(bool(values.size), "COCO metric has no valid values")
        return float(values.mean())

    def ap(iou=None):
        values = precision[:, :, :, area, max100]
        if iou is not None:
            found = np.flatnonzero(np.isclose(ious, iou, rtol=0.0, atol=1.0e-12))
            require(len(found) == 1, "Missing IoU threshold: %s" % iou)
            values = values[int(found[0])]
        return average(values)

    return {
        "AP": ap(),
        "AP50": ap(0.50),
        "AP75": ap(0.75),
        "AR100": average(recall[:, :, area, max100]),
    }


def evaluate(annotation_path: Path, detections: List[Dict[str, Any]], image_ids: List[int]) -> Dict[str, float]:
    from faster_coco_eval import COCO, COCOeval_faster

    selected = set(int(value) for value in image_ids)
    subset_detections = [row for row in detections if int(row["image_id"]) in selected]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation_path))
        dt = gt.loadRes(subset_detections)
        evaluator = COCOeval_faster(
            gt, iouType="bbox", print_function=lambda *_args, **_kwargs: None, separate_eval=True
        )
        evaluator.cocoDt = dt
        evaluator.params.imgIds = sorted(selected)
        evaluator.evaluate()
        evaluator.accumulate()
    return metric_values(evaluator.eval, evaluator.params)


def prediction_path(root: Path, method: str, seed: int) -> Path:
    return root / method / ("seed%d" % seed) / "predictions.json"


def load_prediction(root: Path, method: str, seed: int, expected_ids: List[int]) -> Dict[str, Any]:
    path = prediction_path(root, method, seed)
    payload = read_json(path)
    require(payload.get("method") == method, "Prediction method mismatch: %s" % path)
    require(int(payload.get("detector_seed", -1)) == seed, "Prediction seed mismatch: %s" % path)
    require(payload.get("formal_full_test_metric_replay") == "PASS", "Prediction lacks formal replay: %s" % path)
    require(payload.get("test_used_for_selection") is False, "Prediction claims Test selection: %s" % path)
    require([int(value) for value in payload.get("image_ids", [])] == expected_ids, "Prediction image IDs differ: %s" % path)
    return payload


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    train_annotation_path: Path,
    test_annotation_path: Path,
    thresholds_path: Path,
    prediction_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    train_annotation = read_json(train_annotation_path)
    test_annotation = read_json(test_annotation_path)
    frozen = read_json(thresholds_path)
    recomputed = freeze_thresholds(train_annotation)
    require(same_thresholds(frozen, recomputed), "Frozen Train thresholds do not reproduce")
    groups = make_groups(test_annotation, frozen)
    test_ids = sorted(int(row["id"]) for row in test_annotation.get("images", []))
    require(len(test_ids) == 237, "Expected 237 official Test images")

    predictions = {
        (method, seed): load_prediction(prediction_root, method, seed, test_ids)
        for method in METHODS for seed in SEEDS
    }
    per_seed = []
    raw = {}
    for group_name, group in groups["groups"].items():
        grouping, subset = group_name.split("_", 1)
        raw[group_name] = {}
        for seed in SEEDS:
            metrics_by_method = {}
            for method in METHODS:
                metrics_by_method[method] = evaluate(
                    test_annotation_path,
                    predictions[(method, seed)]["detections"],
                    group["image_ids"],
                )
            raw[group_name][str(seed)] = metrics_by_method
            row = {
                "grouping": grouping,
                "subset": subset,
                "seed": seed,
                "image_count": group["image_count"],
                "gt_count": group["gt_count"],
            }
            for metric in METRICS:
                row["B0_%s" % metric] = 100.0 * metrics_by_method["B0"][metric]
                row["Full_%s" % metric] = 100.0 * metrics_by_method["Full"][metric]
                row["delta_%s_pp" % metric] = 100.0 * (
                    metrics_by_method["Full"][metric] - metrics_by_method["B0"][metric]
                )
            per_seed.append(row)

    summary_rows = []
    for grouping in ("density", "overlap"):
        for subset in ("low", "high"):
            selected = [row for row in per_seed if row["grouping"] == grouping and row["subset"] == subset]
            require(len(selected) == 3, "Incomplete three-seed subgroup: %s/%s" % (grouping, subset))
            row = {
                "grouping": grouping,
                "subset": subset,
                "image_count": selected[0]["image_count"],
                "gt_count": selected[0]["gt_count"],
            }
            for metric in METRICS:
                for method in METHODS:
                    values = [float(item["%s_%s" % (method, metric)]) for item in selected]
                    row["%s_%s_mean" % (method, metric)] = mean(values)
                    row["%s_%s_sample_std" % (method, metric)] = sample_std(values)
                deltas = [float(item["delta_%s_pp" % metric]) for item in selected]
                row["delta_%s_mean_pp" % metric] = mean(deltas)
                row["delta_%s_sample_std_pp" % metric] = sample_std(deltas)
            summary_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_fields = ["grouping", "subset", "seed", "image_count", "gt_count"]
    for metric in METRICS:
        per_seed_fields.extend(["B0_%s" % metric, "Full_%s" % metric, "delta_%s_pp" % metric])
    summary_fields = ["grouping", "subset", "image_count", "gt_count"]
    for metric in METRICS:
        summary_fields.extend([
            "B0_%s_mean" % metric, "B0_%s_sample_std" % metric,
            "Full_%s_mean" % metric, "Full_%s_sample_std" % metric,
            "delta_%s_mean_pp" % metric, "delta_%s_sample_std_pp" % metric,
        ])
    write_csv(output_dir / "per_seed_results.csv", per_seed, per_seed_fields)
    write_csv(output_dir / "summary_table.csv", summary_rows, summary_fields)
    write_json(output_dir / "test_image_groups.json", groups)
    result = {
        "format_version": 1,
        "analysis": "CDW1630 density/overlap whole-image stratified evaluation",
        "train_annotation_sha256": sha256(train_annotation_path),
        "test_annotation_sha256": sha256(test_annotation_path),
        "thresholds_sha256": sha256(thresholds_path),
        "prediction_files": {
            "%s_seed%d" % (method, seed): {
                "sha256": sha256(prediction_path(prediction_root, method, seed)),
                "full_test_metric_replay": "PASS",
            }
            for method in METHODS for seed in SEEDS
        },
        "thresholds": frozen,
        "raw_metrics": raw,
        "summary_rows_percent": summary_rows,
        "test_used_for_selection": False,
        "claim_boundary": "condition-specific full-system behavior; no component-level causal attribution",
    }
    write_json(output_dir / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotation", type=Path, required=True)
    parser.add_argument("--test-annotation", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_analysis(
        args.train_annotation.resolve(), args.test_annotation.resolve(),
        args.thresholds.resolve(), args.prediction_root.resolve(), args.output_dir.resolve(),
    )
    print("CDW1630 DENSITY/OVERLAP STRATIFIED ANALYSIS: PASS")


if __name__ == "__main__":
    main()
