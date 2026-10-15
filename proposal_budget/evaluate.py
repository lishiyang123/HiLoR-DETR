#!/usr/bin/env python3
"""Offline budgeted sorting-proposal evaluation for frozen CDW1630 detections."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import (
    BUDGETS,
    EXPECTED_CANDIDATES_PER_IMAGE,
    EXPECTED_TEST_IMAGES,
    IOU_THRESHOLDS,
    MAIN_BUDGET,
    MAIN_IOU,
    METHODS,
    SEEDS,
    mean,
    read_json,
    require,
    sample_std,
    sha256,
    write_json,
    write_text,
)


RATE_METRICS = (
    "target_recall",
    "valid_action_rate",
    "duplicate_action_rate",
    "other_invalid_rate",
    "wasted_action_rate",
)
COUNT_FIELDS = (
    "successful_actions",
    "duplicate_actions",
    "other_invalid_actions",
    "localization_insufficient_actions",
    "category_error_actions",
    "background_actions",
)
BOUND_TOLERANCE = 1.0e-3


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def numeric_bbox(
    value: Any,
    label: str,
    positive_size_required: bool = False,
) -> Tuple[float, float, float, float]:
    require(isinstance(value, list) and len(value) == 4, "%s bbox must contain four values" % label)
    box = tuple(float(item) for item in value)
    require(all(math.isfinite(item) for item in box), "%s bbox contains a non-finite value" % label)
    if positive_size_required:
        require(box[2] > 0.0 and box[3] > 0.0, "%s bbox has non-positive width or height" % label)
    return box


def require_inside_image(box: Sequence[float], width: float, height: float, label: str) -> None:
    require(bbox_outside_excursion(box, width, height) <= BOUND_TOLERANCE, "%s bbox exceeds image bounds" % label)


def bbox_outside_excursion(box: Sequence[float], width: float, height: float) -> float:
    """Return the largest pixel excursion beyond an image boundary."""
    x, y, box_width, box_height = [float(value) for value in box]
    return max(
        0.0,
        -x,
        -y,
        x + box_width - float(width),
        y + box_height - float(height),
    )


def iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    if lw <= 0.0 or lh <= 0.0 or rw <= 0.0 or rh <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


def load_annotation(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    images: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("images", []):
        image_id = int(row["id"])
        require(image_id not in images, "Duplicate image id in Test annotation: %d" % image_id)
        width = float(row["width"])
        height = float(row["height"])
        require(math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0, "Invalid image size")
        images[image_id] = {"id": image_id, "width": width, "height": height}
    require(len(images) == EXPECTED_TEST_IMAGES, "Expected %d Test images, got %d" % (EXPECTED_TEST_IMAGES, len(images)))

    categories = {int(row["id"]) for row in payload.get("categories", [])}
    require(bool(categories), "Test annotation has no categories")
    gt_by_image: Dict[int, List[Dict[str, Any]]] = {image_id: [] for image_id in images}
    crowd_count = 0
    noncrowd_count = 0
    annotation_ids = set()
    for position, row in enumerate(payload.get("annotations", [])):
        annotation_id = int(row.get("id", position + 1))
        require(annotation_id not in annotation_ids, "Duplicate annotation id: %d" % annotation_id)
        annotation_ids.add(annotation_id)
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        require(image_id in images, "GT references unknown image id: %d" % image_id)
        require(category_id in categories, "GT references unknown category id: %d" % category_id)
        box = numeric_bbox(row.get("bbox"), "GT %d" % annotation_id, positive_size_required=True)
        require_inside_image(box, images[image_id]["width"], images[image_id]["height"], "GT %d" % annotation_id)
        iscrowd = int(row.get("iscrowd", 0)) == 1
        crowd_count += int(iscrowd)
        noncrowd_count += int(not iscrowd)
        gt_by_image[image_id].append({
            "id": annotation_id,
            "category_id": category_id,
            "bbox": box,
            "iscrowd": iscrowd,
        })
    return {
        "payload": payload,
        "images": images,
        "image_ids": sorted(images),
        "categories": categories,
        "gt_by_image": gt_by_image,
        "noncrowd_gt_count": noncrowd_count,
        "crowd_gt_count": crowd_count,
        "empty_noncrowd_image_count": sum(
            not any(not gt["iscrowd"] for gt in rows) for rows in gt_by_image.values()
        ),
    }


def prediction_path(root: Path, method: str, seed: int) -> Path:
    return root / method / ("seed%d" % seed) / "predictions.json"


def validate_prediction_payload(
    path: Path,
    method: str,
    seed: int,
    annotation: Dict[str, Any],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    payload = read_json(path)
    require(payload.get("dataset") == "CDW1630 official Test", "Prediction dataset identity mismatch: %s" % path)
    require(payload.get("method") == method, "Prediction method mismatch: %s" % path)
    require(int(payload.get("detector_seed", -1)) == seed, "Prediction seed mismatch: %s" % path)
    require(payload.get("formal_full_test_metric_replay") == "PASS", "Prediction lacks formal Test replay: %s" % path)
    require(payload.get("test_used_for_selection") is False, "Prediction claims Test selection: %s" % path)
    image_ids = [int(value) for value in payload.get("image_ids", [])]
    require(image_ids == annotation["image_ids"], "Prediction image ids differ from official Test: %s" % path)
    require(int(payload.get("image_count", -1)) == EXPECTED_TEST_IMAGES, "Prediction image count mismatch: %s" % path)
    detections = payload.get("detections")
    require(isinstance(detections, list), "Prediction detections must be a list: %s" % path)

    grouped: Dict[int, List[Dict[str, Any]]] = {image_id: [] for image_id in annotation["image_ids"]}
    exact_records = set()
    exact_duplicates = 0
    out_of_bounds_boxes = 0
    maximum_outside_excursion = 0.0
    degenerate_boxes = 0
    nonpositive_width_boxes = 0
    nonpositive_height_boxes = 0
    minimum_width = math.inf
    minimum_height = math.inf
    for original_index, row in enumerate(detections):
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        require(image_id in grouped, "Prediction references unknown Test image: %d" % image_id)
        require(category_id in annotation["categories"], "Prediction references unknown category: %d" % category_id)
        score = float(row["score"])
        require(math.isfinite(score), "Prediction contains a non-finite score")
        require(0.0 <= score <= 1.0, "Prediction score is outside [0, 1]")
        box = numeric_bbox(row.get("bbox"), "Prediction row %d" % original_index)
        minimum_width = min(minimum_width, box[2])
        minimum_height = min(minimum_height, box[3])
        width_nonpositive = box[2] <= 0.0
        height_nonpositive = box[3] <= 0.0
        nonpositive_width_boxes += int(width_nonpositive)
        nonpositive_height_boxes += int(height_nonpositive)
        degenerate_boxes += int(width_nonpositive or height_nonpositive)
        image = annotation["images"][image_id]
        excursion = bbox_outside_excursion(box, image["width"], image["height"])
        if excursion > BOUND_TOLERANCE:
            out_of_bounds_boxes += 1
            maximum_outside_excursion = max(maximum_outside_excursion, excursion)
        identity = (image_id, category_id, box, score)
        if identity in exact_records:
            exact_duplicates += 1
        exact_records.add(identity)
        grouped[image_id].append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": box,
            "score": score,
            "original_index": original_index,
        })
    require(exact_duplicates == 0, "Prediction contains %d exact duplicate records: %s" % (exact_duplicates, path))
    declared_count = int(payload.get("detection_count", -1))
    require(declared_count == len(detections), "Declared detection count mismatch: %s" % path)

    candidate_counts = [len(grouped[image_id]) for image_id in annotation["image_ids"]]
    require(
        min(candidate_counts) == EXPECTED_CANDIDATES_PER_IMAGE
        and max(candidate_counts) == EXPECTED_CANDIDATES_PER_IMAGE,
        "Every image must contain exactly %d final candidates: %s" % (EXPECTED_CANDIDATES_PER_IMAGE, path),
    )
    tie_images = 0
    tied_score_groups = 0
    for image_id in annotation["image_ids"]:
        frequencies = Counter(row["score"] for row in grouped[image_id])
        groups = sum(count > 1 for count in frequencies.values())
        tie_images += int(groups > 0)
        tied_score_groups += groups
    return grouped, {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "image_count": len(grouped),
        "detection_count": len(detections),
        "candidates_per_image_min": min(candidate_counts),
        "candidates_per_image_max": max(candidate_counts),
        "exact_duplicate_record_count": exact_duplicates,
        "images_with_exact_score_ties": tie_images,
        "exact_score_tie_group_count": tied_score_groups,
        "out_of_bounds_box_count": out_of_bounds_boxes,
        "maximum_outside_excursion_px": maximum_outside_excursion,
        "degenerate_box_count": degenerate_boxes,
        "nonpositive_width_box_count": nonpositive_width_boxes,
        "nonpositive_height_box_count": nonpositive_height_boxes,
        "minimum_exported_width_px": minimum_width,
        "minimum_exported_height_px": minimum_height,
        "bbox_format": "COCO xywh",
        "bbox_bounds_check": "AUDITED__PRESERVED_UNCHANGED",
        "bbox_bounds_policy": "finite positive boxes are evaluated exactly as exported; no clipping or coordinate modification",
        "degenerate_bbox_policy": "retained in the score queue, assigned IoU zero, and counted as an other-invalid background action",
        "finite_score_and_bbox_check": "PASS",
        "formal_full_test_metric_replay": "PASS",
    }


def stable_rank(detections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(detections, key=lambda row: (-float(row["score"]), int(row["original_index"])))


def action_label(
    detection: Dict[str, Any],
    noncrowd_gt: List[Dict[str, Any]],
    matched: set,
    tau: float,
) -> Tuple[str, int]:
    overlaps = [(iou_xywh(detection["bbox"], gt["bbox"]), gt) for gt in noncrowd_gt]
    eligible = [
        (overlap, gt) for overlap, gt in overlaps
        if gt["category_id"] == detection["category_id"] and gt["id"] not in matched and overlap >= tau
    ]
    if eligible:
        overlap, gt = sorted(eligible, key=lambda item: (-item[0], int(item[1]["id"])))[0]
        del overlap
        return "success", int(gt["id"])
    duplicate = any(
        gt["category_id"] == detection["category_id"] and gt["id"] in matched and overlap >= tau
        for overlap, gt in overlaps
    )
    if duplicate:
        return "duplicate", -1
    category_error = any(
        gt["category_id"] != detection["category_id"] and overlap >= tau
        for overlap, gt in overlaps
    )
    if category_error:
        return "category_error", -1
    localization_insufficient = any(
        gt["category_id"] == detection["category_id"] and 0.0 < overlap < tau
        for overlap, gt in overlaps
    )
    if localization_insufficient:
        return "localization_insufficient", -1
    return "background", -1


def evaluate_image(
    detections: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
    tau: float,
    budgets: Sequence[int] = BUDGETS,
) -> Dict[int, Dict[str, int]]:
    noncrowd_gt = [row for row in gt_rows if not row["iscrowd"]]
    ranked = stable_rank(detections)
    max_budget = max(budgets)
    counts = Counter()
    matched = set()
    snapshots: Dict[int, Dict[str, int]] = {}
    for rank, detection in enumerate(ranked[:max_budget], start=1):
        label, annotation_id = action_label(detection, noncrowd_gt, matched, tau)
        counts[label] += 1
        if label == "success":
            matched.add(annotation_id)
        if rank in budgets:
            other_invalid = counts["localization_insufficient"] + counts["category_error"] + counts["background"]
            snapshots[rank] = {
                "noncrowd_gt_count": len(noncrowd_gt),
                "executed_actions": rank,
                "successful_actions": counts["success"],
                "duplicate_actions": counts["duplicate"],
                "other_invalid_actions": other_invalid,
                "localization_insufficient_actions": counts["localization_insufficient"],
                "category_error_actions": counts["category_error"],
                "background_actions": counts["background"],
            }
    require(set(snapshots) == set(budgets), "Insufficient ranked candidates for requested budgets")
    return snapshots


def safe_rate(numerator: int, denominator: int) -> float:
    require(denominator > 0, "Rate denominator must be positive")
    return float(numerator) / float(denominator)


def add_rates(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result["target_recall"] = safe_rate(result["successful_actions"], result["noncrowd_gt_count"])
    result["valid_action_rate"] = safe_rate(result["successful_actions"], result["executed_actions"])
    result["duplicate_action_rate"] = safe_rate(result["duplicate_actions"], result["executed_actions"])
    result["other_invalid_rate"] = safe_rate(result["other_invalid_actions"], result["executed_actions"])
    result["wasted_action_rate"] = safe_rate(
        result["duplicate_actions"] + result["other_invalid_actions"], result["executed_actions"]
    )
    require(
        result["successful_actions"] + result["duplicate_actions"] + result["other_invalid_actions"]
        == result["executed_actions"],
        "Action counts do not partition the executed queue",
    )
    require(
        result["localization_insufficient_actions"] + result["category_error_actions"] + result["background_actions"]
        == result["other_invalid_actions"],
        "Other-invalid decomposition is incomplete",
    )
    require(abs(result["valid_action_rate"] + result["wasted_action_rate"] - 1.0) <= 1.0e-12, "Rate complement failed")
    require(0.0 <= result["target_recall"] <= 1.0, "Target recall is outside [0, 1]")
    return result


def evaluate_method_seed(
    method: str,
    seed: int,
    tau: float,
    detections_by_image: Dict[int, List[Dict[str, Any]]],
    annotation: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    totals = {budget: Counter() for budget in BUDGETS}
    per_image = []
    for image_id in annotation["image_ids"]:
        snapshots = evaluate_image(
            detections_by_image[image_id], annotation["gt_by_image"][image_id], tau, BUDGETS
        )
        for budget in BUDGETS:
            snapshot = snapshots[budget]
            for field, value in snapshot.items():
                totals[budget][field] += int(value)
            per_image.append({
                "method": method,
                "seed": seed,
                "tau": tau,
                "K": budget,
                "image_id": image_id,
                **snapshot,
            })
    rows = []
    for budget in BUDGETS:
        row = add_rates({
            "method": method,
            "seed": seed,
            "tau": tau,
            "K": budget,
            **dict(totals[budget]),
        })
        require(row["noncrowd_gt_count"] == annotation["noncrowd_gt_count"], "GT denominator changed")
        require(row["executed_actions"] == EXPECTED_TEST_IMAGES * budget, "Action denominator changed")
        rows.append(row)
    return rows, per_image


def aggregate_three_seeds(per_seed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for tau in IOU_THRESHOLDS:
        for budget in BUDGETS:
            selected = [row for row in per_seed if row["tau"] == tau and row["K"] == budget]
            require(len(selected) == len(METHODS) * len(SEEDS), "Incomplete method/seed grid")
            for metric in RATE_METRICS:
                by_method = {
                    method: [
                        float(row[metric]) for row in selected if row["method"] == method
                    ] for method in METHODS
                }
                paired = []
                for seed in SEEDS:
                    b0 = next(row for row in selected if row["method"] == "B0" and row["seed"] == seed)
                    full = next(row for row in selected if row["method"] == "Full" and row["seed"] == seed)
                    paired.append(float(full[metric]) - float(b0[metric]))
                summary.append({
                    "tau": tau,
                    "K": budget,
                    "metric": metric,
                    "B0_mean": mean(by_method["B0"]),
                    "B0_sample_std": sample_std(by_method["B0"]),
                    "Full_mean": mean(by_method["Full"]),
                    "Full_sample_std": sample_std(by_method["Full"]),
                    "paired_delta_mean_Full_minus_B0": mean(paired),
                    "paired_delta_sample_std": sample_std(paired),
                    "paired_seed_deltas": paired,
                })
    return summary




def interpretation_markdown() -> str:
    return """# Interpretation boundary

This post-hoc evaluation treats each final scored detection as a proposed
sorting action under a fixed per-image action budget. It measures unique
non-crowd target coverage and how the limited proposal queue is divided among
valid, duplicate, and other-invalid detections.

It is not a physical robot experiment. The results do not measure grasp
success, realized material recovery, cycle time, collision avoidance,
graspability, or closed-loop sorting performance. The appropriate claim is
limited to the quality of detection-driven sorting proposals under the stated
offline matching rule.
"""


def run_evaluation(
    annotation_path: Path,
    prediction_root: Path,
    output_root: Path,
    protocol_sha256: str,
) -> Dict[str, Any]:
    annotation = load_annotation(annotation_path)
    predictions = {}
    audits = {}
    for method in METHODS:
        for seed in SEEDS:
            path = prediction_path(prediction_root, method, seed)
            grouped, audit = validate_prediction_payload(path, method, seed, annotation)
            predictions[(method, seed)] = grouped
            audits["%s_seed%d" % (method, seed)] = audit

    per_seed: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    for tau in IOU_THRESHOLDS:
        for method in METHODS:
            for seed in SEEDS:
                rows, image_rows = evaluate_method_seed(
                    method, seed, tau, predictions[(method, seed)], annotation
                )
                per_seed.extend(rows)
                per_image.extend(image_rows)
    summary = aggregate_three_seeds(per_seed)
    main_rows = [row for row in summary if row["tau"] == MAIN_IOU and row["K"] == MAIN_BUDGET]
    require(len(main_rows) == len(RATE_METRICS), "Main table is incomplete")

    results = output_root / "results"
    write_csv(
        results / "PER_SEED_METRICS.csv",
        per_seed,
        ["method", "seed", "tau", "K", "noncrowd_gt_count", "executed_actions", *COUNT_FIELDS, *RATE_METRICS],
    )
    write_csv(
        results / "PER_IMAGE_COUNTS.csv",
        per_image,
        ["method", "seed", "tau", "K", "image_id", "noncrowd_gt_count", "executed_actions", *COUNT_FIELDS],
    )
    write_csv(
        results / "THREE_SEED_SUMMARY.csv",
        summary,
        [
            "tau", "K", "metric", "B0_mean", "B0_sample_std", "Full_mean", "Full_sample_std",
            "paired_delta_mean_Full_minus_B0", "paired_delta_sample_std", "paired_seed_deltas",
        ],
    )
    write_csv(
        results / "MAIN_TABLE_K5_TAU075.csv",
        main_rows,
        [
            "tau", "K", "metric", "B0_mean", "B0_sample_std", "Full_mean", "Full_sample_std",
            "paired_delta_mean_Full_minus_B0", "paired_delta_sample_std", "paired_seed_deltas",
        ],
    )
    write_text(results / "INTERPRETATION.md", interpretation_markdown())

    report = {
        "status": "COMPLETE__OFFLINE_BUDGETED_SORTING_PROPOSAL_EVALUATION__STAGED_ONLY",
        "protocol_sha256": protocol_sha256,
        "dataset": "CDW1630 official Test",
        "annotation": {
            "path": str(annotation_path),
            "sha256": sha256(annotation_path),
            "image_count": len(annotation["image_ids"]),
            "noncrowd_gt_count": annotation["noncrowd_gt_count"],
            "crowd_gt_count": annotation["crowd_gt_count"],
            "empty_noncrowd_image_count": annotation["empty_noncrowd_image_count"],
        },
        "prediction_audit": audits,
        "methods": list(METHODS),
        "detector_seeds": list(SEEDS),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "action_budgets": list(BUDGETS),
        "main_setting": {"tau": MAIN_IOU, "K": MAIN_BUDGET},
        "per_seed_rows": per_seed,
        "three_seed_summary_rows": summary,
        "training_executed": False,
        "inference_executed": False,
        "gpu_used": False,
        "test_used_for_reporting": True,
        "test_used_for_model_checkpoint_threshold_parameter_or_protocol_selection": False,
        "release_write": False,
        "claim_boundary": "offline detection-driven sorting-proposal proxy; no physical robotic validation",
    }
    write_json(results / "ACTION_BUDGET_RESULTS.json", report)
    return report
