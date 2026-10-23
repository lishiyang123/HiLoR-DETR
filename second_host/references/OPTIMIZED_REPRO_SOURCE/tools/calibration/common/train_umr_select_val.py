#!/usr/bin/env python3
"""Fit UMR candidates on Train and select fusion parameters on Val only.

There is deliberately no Test input. All input paths are explicit CLI
arguments so this program is independent of historical diagnostic folders.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


NUM_CLASSES = 3
NUM_QUERIES = 300
TOPK = 300
IOU_LEVELS = (0.75, 0.80, 0.85, 0.90, 0.95)
SEEDS = (0, 1, 2)
POS_WEIGHTS = (1, 2, 4)
TAU_GRID = (0.0, 0.1, 0.2, 0.4)
BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SCORE_WEIGHTS = {"AP": 0.35, "AP75": 0.45, "APm": 0.10, "AR100": 0.10}
MAX_SAFE_DROP = 0.001
METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1.0e-9)


def compute_marginal_labels(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_iscrowd: torch.Tensor,
    boxes_absolute: torch.Tensor,
    qccr_score: torch.Tensor,
) -> torch.Tensor:
    num_queries, num_classes = qccr_score.shape
    if (num_queries, num_classes) != (NUM_QUERIES, NUM_CLASSES):
        raise RuntimeError(f"unexpected score shape: {tuple(qccr_score.shape)}")
    if gt_labels.numel() and (
        int(gt_labels.min()) < 0 or int(gt_labels.max()) >= num_classes
    ):
        raise RuntimeError(
            f"ground-truth category is outside contiguous IDs 0..{num_classes - 1}"
        )

    iou = box_iou_xyxy(boxes_absolute, gt_boxes)
    noncrowd = (gt_iscrowd == 0).bool()
    top100 = set(torch.topk(qccr_score.flatten(), k=100).indices.tolist())
    marginal = torch.zeros(num_queries, num_classes, dtype=torch.float32)
    for threshold in IOU_LEVELS:
        first_hit: dict[int, int] = {}
        threshold_target = torch.zeros_like(marginal)
        for gt_index in range(gt_boxes.shape[0]):
            if not bool(noncrowd[gt_index]):
                continue
            class_index = int(gt_labels[gt_index])
            valid = iou[:, gt_index] >= threshold
            if not bool(valid.any()):
                continue
            scores = qccr_score[:, class_index].clone()
            scores[~valid] = -float("inf")
            query_index = int(torch.argmax(scores))
            threshold_target[query_index, class_index] = 1.0
            first_hit[gt_index] = query_index * num_classes + class_index
        _covered = [index for index in first_hit if first_hit[index] in top100]
        marginal += threshold_target
    return marginal / float(len(IOU_LEVELS))


def qfl_loss(
    logits: torch.Tensor, targets: torch.Tensor, positive_weight: float
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = (probabilities - targets).abs().pow(2)
    positive = torch.full_like(targets, float(positive_weight))
    weights *= torch.where(targets > 0, positive, torch.ones_like(targets))
    return (weights * bce).mean()


def make_split(ids: list[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = sorted(set(ids))
    if len(unique_ids) < 2:
        raise RuntimeError("Train capture must contain at least two images")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(unique_ids))
    n_fit = min(max(int(round(len(unique_ids) * 0.9)), 1), len(unique_ids) - 1)
    fit_ids = {unique_ids[index] for index in permutation[:n_fit]}
    holdout_ids = {unique_ids[index] for index in permutation[n_fit:]}
    fit_idx = np.asarray([index for index, value in enumerate(ids) if value in fit_ids])
    holdout_idx = np.asarray(
        [index for index, value in enumerate(ids) if value in holdout_ids]
    )
    return fit_idx, holdout_idx


def train_one(
    head_type: type[torch.nn.Module],
    data: dict[str, torch.Tensor],
    image_ids: list[int],
    seed: int,
    positive_weight: int,
) -> tuple[torch.nn.Module, dict[str, list[float]], int, float, int, int]:
    fit_idx, holdout_idx = make_split(image_ids, seed)
    torch.manual_seed(seed)
    model = head_type(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_holdout = float("inf")
    best_epoch = -1
    best_state = None
    patience_count = 0
    history: dict[str, list[float]] = {"train": [], "holdout": []}

    for epoch in range(50):
        model.train()
        permutation = torch.randperm(len(fit_idx))
        train_total = 0.0
        train_batches = 0
        for start in range(0, len(permutation), 16):
            index = fit_idx[permutation[start : start + 16]]
            loss = qfl_loss(
                model(
                    data["hidden"][index],
                    data["classes"][index],
                    data["features"][index],
                ),
                data["targets"][index],
                positive_weight=float(positive_weight),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            train_batches += 1
        train_loss = train_total / max(train_batches, 1)
        history["train"].append(train_loss)

        model.eval()
        holdout_total = 0.0
        holdout_batches = 0
        with torch.no_grad():
            for start in range(0, len(holdout_idx), 16):
                index = holdout_idx[start : start + 16]
                loss = qfl_loss(
                    model(
                        data["hidden"][index],
                        data["classes"][index],
                        data["features"][index],
                    ),
                    data["targets"][index],
                    positive_weight=float(positive_weight),
                )
                holdout_total += float(loss)
                holdout_batches += 1
        holdout_loss = holdout_total / max(holdout_batches, 1)
        history["holdout"].append(holdout_loss)
        print(
            f"seed={seed} pw={positive_weight} epoch={epoch:02d}: "
            f"train={train_loss:.9f} holdout={holdout_loss:.9f}"
        )
        if holdout_loss < best_holdout - 1.0e-6:
            best_holdout = holdout_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 8:
            break

    if best_state is None:
        raise RuntimeError("UMR candidate training produced no best state")
    model.load_state_dict(best_state)
    return (
        model,
        history,
        best_epoch,
        best_holdout,
        int(len(fit_idx)),
        int(len(holdout_idx)),
    )


def xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    return [
        float(box[0]), float(box[1]),
        float(box[2] - box[0]), float(box[3] - box[1]),
    ]


def mean_valid(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[array > -1]
    return float(array.mean()) if array.size else float("nan")


def exact_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    precision = np.asarray(evaluation["precision"], dtype=np.float64)
    recall = np.asarray(evaluation["recall"], dtype=np.float64)
    params = evaluation["params"]
    ious = np.asarray(params.iouThrs)
    areas = list(params.areaRngLbl)
    max_dets = list(params.maxDets)
    category_ids = [int(value) for value in params.catIds]
    expected_category_ids = list(range(NUM_CLASSES))
    if category_ids != expected_category_ids or precision.shape[2] != NUM_CLASSES:
        raise RuntimeError(
            "Val evaluation category mismatch: "
            f"expected={expected_category_ids}, actual={category_ids}"
        )

    def threshold(value: float) -> int:
        found = np.where(np.isclose(ious, value))[0]
        if len(found) != 1:
            raise RuntimeError(f"missing IoU threshold {value}")
        return int(found[0])

    def ap(iou: float | None = None, area: str = "all", max_det: int = 100) -> float:
        area_index = areas.index(area)
        max_det_index = max_dets.index(max_det)
        selected = precision[:, :, :, area_index, max_det_index]
        if iou is not None:
            selected = precision[
                threshold(iou), :, :, area_index, max_det_index
            ]
        return mean_valid(selected)

    def ar(area: str = "all", max_det: int = 100) -> float:
        return mean_valid(recall[:, :, areas.index(area), max_dets.index(max_det)])

    return {
        "AP": ap(), "AP50": ap(0.50), "AP75": ap(0.75),
        "APs": ap(area="small"), "APm": ap(area="medium"),
        "APl": ap(area="large"), "AR1": ar(max_det=1),
        "AR10": ar(max_det=10), "AR100": ar(max_det=100),
        "ARs": ar(area="small"), "ARm": ar(area="medium"),
        "ARl": ar(area="large"), "AP80": ap(0.80),
        "AP85": ap(0.85), "AP90": ap(0.90), "AP95": ap(0.95),
    }


def run_coco(
    annotation: Path, detections: list[dict[str, Any]], image_ids: list[int]
) -> dict[str, float]:
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(
        coco_gt, iouType="bbox", print_function=lambda *_args, **_kwargs: None,
        separate_eval=True,
    )
    evaluator.cocoDt = coco_dt
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    return exact_metrics(evaluator.eval)


def score(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    return 100.0 * sum(
        weight * (metrics[name] - baseline[name])
        for name, weight in SCORE_WEIGHTS.items()
    )


def guards(
    row: dict[str, Any], baseline: dict[str, float], per_seed: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if row["beta"] <= 0.0:
        failed.append("beta_must_be_positive")
    if row["AP"] < baseline["AP"]:
        failed.append("mean_ap_not_lower")
    if row["AP75"] < baseline["AP75"]:
        failed.append("mean_ap75_not_lower")
    for name in ("AP50", "APm", "APl", "AR100"):
        if row[name] < baseline[name] - MAX_SAFE_DROP:
            failed.append(f"mean_{name.lower()}_drop_gt_0.1pp")
    positive = sum(
        item["AP"] >= baseline["AP"] and item["AP75"] >= baseline["AP75"]
        for item in per_seed
    )
    row["positive_seed_count"] = int(positive)
    if positive < 2:
        failed.append("fewer_than_two_positive_seeds")
    return not failed, failed


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite(row.get(key)) for key in columns})


def main() -> None:
    global NUM_CLASSES
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--qccr-artifact", type=Path, required=True)
    parser.add_argument("--val-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    NUM_CLASSES = int(args.num_classes)
    if NUM_CLASSES <= 0:
        raise RuntimeError("--num-classes must be positive")
    branch = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    train_capture_path = args.train_capture.resolve()
    val_capture_path = args.val_capture.resolve()
    qccr_path = args.qccr_artifact.resolve()
    annotation = args.val_annotation.resolve()
    run_dir = args.output_dir.resolve()
    for path in (
        branch, checkpoint, train_capture_path, val_capture_path,
        qccr_path, annotation,
    ):
        if not path.exists():
            raise RuntimeError(f"required input is missing: {path}")
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha:
        raise RuntimeError(
            "checkpoint SHA mismatch: "
            f"expected={args.expected_checkpoint_sha}, actual={checkpoint_sha}"
        )
    branch_qccr = branch / "engine/deim/qccr.py"
    branch_umr = branch / "engine/deim/umr.py"

    inputs = {
        "detector_checkpoint": checkpoint,
        "train_capture": train_capture_path,
        "val_capture": val_capture_path,
        "qccr_artifact": qccr_path,
        "val_annotation": annotation,
        "runtime_qccr": branch_qccr,
        "runtime_umr": branch_umr,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise RuntimeError("required retained input missing:\n" + "\n".join(missing))
    before = {name: sha256(path) for name, path in inputs.items()}

    sys.path.insert(0, str(branch))
    from engine.deim.qccr import apply_qccr, compute_competition_risk
    from engine.deim.umr import UMRHead, box_cxcywh_to_xyxy, build_image_features

    qccr_artifact = json.loads(qccr_path.read_text(encoding="utf-8"))
    if qccr_artifact.get("source_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("QCCR/checkpoint provenance mismatch")
    qccr = {
        key: float(value) for key, value in qccr_artifact["parameters"].items()
    }

    print("===== INPUT PREFLIGHT =====")
    print("Dataset         :", args.dataset_name)
    print("Checkpoint SHA  :", checkpoint_sha)
    print("Train capture   :", train_capture_path)
    print("Val capture     :", val_capture_path)
    print("QCCR artifact   :", qccr_path)
    print("Val annotation  :", annotation)
    print("Test files      : NOT ACCESSED")

    train_capture = torch.load(
        train_capture_path, map_location="cpu", weights_only=False
    )
    val_capture = torch.load(
        val_capture_path, map_location="cpu", weights_only=False
    )
    for split, capture in (("train", train_capture), ("val", val_capture)):
        if capture.get("split") != split or capture.get("method") not in {
            "PD", "frozen_native_detector"
        }:
            raise RuntimeError(f"invalid {split} capture provenance")
        if capture.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"{split} capture checkpoint mismatch")
        if int(capture.get("postprocessor_num_classes", -1)) != NUM_CLASSES:
            raise RuntimeError(f"{split} capture class-count mismatch")

    train_features: list[torch.Tensor] = []
    train_classes: list[torch.Tensor] = []
    train_hidden: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    train_image_ids: list[int] = []
    positive_count = 0
    candidate_count = 0
    for record_index, record in enumerate(train_capture["records"]):
        z_raw = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()
        height, width = [int(value) for value in record["orig_size"].tolist()]
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        boxes_absolute = box_cxcywh_to_xyxy(boxes) * scale
        risk = compute_competition_risk(z_raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        qccr_logit = apply_qccr(
            z_raw, risk, qccr["beta0"], qccr["beta_positive"],
            qccr["beta_negative"], qccr["duplicate_penalty"],
        )
        qccr_score = torch.sigmoid(qccr_logit)
        marginal = compute_marginal_labels(
            record["gt_boxes"], record["gt_labels"], record["gt_iscrowd"],
            boxes_absolute, qccr_score,
        )
        flat_idx, features, classes, hidden = build_image_features(
            z_raw, boxes, boxes_absolute, record["hL"].float(),
            qccr_logit, qccr_score, risk, K=TOPK,
        )
        query_idx = flat_idx // NUM_CLASSES
        class_idx = flat_idx % NUM_CLASSES
        if not torch.equal(classes, class_idx):
            raise RuntimeError("runtime UMR feature class mapping mismatch")
        selected_target = marginal[query_idx, class_idx]
        train_features.append(features)
        train_classes.append(classes)
        train_hidden.append(hidden)
        train_targets.append(selected_target)
        train_image_ids.append(int(record["image_id"]))
        positive_count += int((selected_target > 0).sum())
        candidate_count += int(selected_target.numel())
        if record_index % 100 == 0:
            print(
                f"Train feature/label build: {record_index}/"
                f"{len(train_capture['records'])}"
            )
    train_data = {
        "features": torch.stack(train_features),
        "classes": torch.stack(train_classes),
        "hidden": torch.stack(train_hidden),
        "targets": torch.stack(train_targets),
    }

    models_dir = run_dir / "models"
    reports_dir = run_dir / "training_reports"
    results_dir = run_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_paths: dict[tuple[int, int], Path] = {}
    model_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        for pw in POS_WEIGHTS:
            model, history, best_epoch, best_holdout, fit_count, holdout_count = train_one(
                UMRHead, train_data, train_image_ids, seed, pw
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            weight_path = models_dir / f"umr_seed{seed}_pw{pw}_best.pt"
            torch.save(model.state_dict(), weight_path)
            model_paths[(seed, pw)] = weight_path
            summary = {
                "seed": seed,
                "positive_weight": pw,
                "source_split": f"{args.dataset_name} Train",
                "source_checkpoint_sha256": checkpoint_sha,
                "qccr_artifact_sha256": before["qccr_artifact"],
                "weight_path": str(weight_path),
                "weight_sha256": sha256(weight_path),
                "num_classes": NUM_CLASSES,
                "head_parameters": parameter_count,
                "train_image_count": len(train_image_ids),
                "fit_image_count": fit_count,
                "holdout_image_count": holdout_count,
                "epochs_run": len(history["train"]),
                "best_epoch": best_epoch,
                "best_holdout_qfl": best_holdout,
                "m_star_positive_rate_m300": positive_count / max(candidate_count, 1),
                "history": history,
            }
            report_path = reports_dir / f"umr_seed{seed}_pw{pw}.json"
            report_path.write_text(
                json.dumps(finite(summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            model_summaries.append(summary)
            print(
                f"TRAIN CANDIDATE PASS: seed={seed} pw={pw} "
                f"best_epoch={best_epoch} holdout={best_holdout:.9f}"
            )

    val_prepared: list[dict[str, Any]] = []
    image_ids: list[int] = []
    for record in val_capture["records"]:
        z_raw = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()
        height, width = [int(value) for value in record["orig_size"].tolist()]
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        boxes_absolute = box_cxcywh_to_xyxy(boxes) * scale
        risk0 = compute_competition_risk(z_raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        qccr_logit0 = apply_qccr(
            z_raw, risk0, qccr["beta0"], qccr["beta_positive"],
            qccr["beta_negative"], qccr["duplicate_penalty"],
        )
        qccr_score0 = torch.sigmoid(qccr_logit0)
        flat_idx, features, classes, hidden = build_image_features(
            z_raw, boxes, boxes_absolute, record["hL"].float(),
            qccr_logit0, qccr_score0, risk0, K=TOPK,
        )
        val_prepared.append({
            "image_id": int(record["image_id"]),
            "z_raw": z_raw,
            "boxes": boxes,
            "boxes_absolute": boxes_absolute,
            "flat_idx": flat_idx,
            "features": features,
            "classes": classes,
            "hidden": hidden,
        })
        image_ids.append(int(record["image_id"]))

    def detections_for(
        predictions: list[torch.Tensor] | None, tau: float, beta: float
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for index, record in enumerate(val_prepared):
            z_raw = record["z_raw"]
            boxes = record["boxes"]
            if predictions is None or beta == 0.0:
                z_umr = z_raw
            else:
                bonus = torch.zeros(NUM_QUERIES * NUM_CLASSES, dtype=torch.float32)
                bonus[record["flat_idx"]] = torch.relu(predictions[index] - tau)
                z_umr = z_raw + beta * bonus.reshape(NUM_QUERIES, NUM_CLASSES)
            risk1 = compute_competition_risk(z_umr.unsqueeze(0), boxes.unsqueeze(0))[0]
            final_logit = apply_qccr(
                z_umr, risk1, qccr["beta0"], qccr["beta_positive"],
                qccr["beta_negative"], qccr["duplicate_penalty"],
            )
            top_scores, top_indices = torch.topk(
                torch.sigmoid(final_logit).flatten(), k=TOPK
            )
            labels = top_indices % NUM_CLASSES
            query_indices = top_indices // NUM_CLASSES
            selected_boxes = record["boxes_absolute"][query_indices]
            for label, candidate_score, box in zip(labels, top_scores, selected_boxes):
                detections.append({
                    "image_id": record["image_id"],
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(candidate_score),
                })
        return detections

    baseline = run_coco(annotation, detections_for(None, 0.0, 0.0), image_ids)
    print("===== QCCR VAL BASELINE =====")
    print(
        "AP={AP:.9f} AP50={AP50:.9f} AP75={AP75:.9f} "
        "APm={APm:.9f} AR100={AR100:.9f}".format(**baseline)
    )

    per_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for pw in POS_WEIGHTS:
            model = UMRHead(num_classes=NUM_CLASSES)
            state = torch.load(
                model_paths[(seed, pw)], map_location="cpu", weights_only=False
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            predictions: list[torch.Tensor] = []
            with torch.inference_mode():
                for record in val_prepared:
                    logits = model(
                        record["hidden"].unsqueeze(0),
                        record["classes"].unsqueeze(0),
                        record["features"].unsqueeze(0),
                    )
                    predictions.append(torch.sigmoid(logits).squeeze(0))
            for tau in TAU_GRID:
                for beta in BETA_GRID:
                    metrics = baseline if beta == 0.0 else run_coco(
                        annotation, detections_for(predictions, tau, beta), image_ids
                    )
                    row = {
                        "seed": seed, "pw": pw, "tau": tau, "beta": beta,
                        **{name: float(metrics[name]) for name in METRICS},
                    }
                    row["score"] = score(row, baseline)
                    per_seed_rows.append(row)
            print(f"VAL GRID PASS: seed={seed} pw={pw}")

    mean_rows: list[dict[str, Any]] = []
    for pw in POS_WEIGHTS:
        for tau in TAU_GRID:
            for beta in BETA_GRID:
                group = [
                    row for row in per_seed_rows
                    if row["pw"] == pw and row["tau"] == tau and row["beta"] == beta
                ]
                if len(group) != len(SEEDS):
                    raise RuntimeError("incomplete seed aggregation")
                row: dict[str, Any] = {"pw": pw, "tau": tau, "beta": beta}
                for name in METRICS:
                    values = np.asarray([item[name] for item in group], dtype=np.float64)
                    row[name] = float(np.nanmean(values))
                    row[f"delta_{name}_pp"] = 100.0 * (row[name] - baseline[name])
                row["score"] = score(row, baseline)
                allowed, failed = guards(row, baseline, group)
                row["guarded"] = bool(allowed)
                row["failed_guards"] = ";".join(failed)
                mean_rows.append(row)

    eligible = [row for row in mean_rows if row["guarded"]]
    eligible.sort(key=lambda row: (row["score"], row["AP"], row["AP75"]), reverse=True)
    selected = eligible[0] if eligible else None
    ranked = sorted(
        [row for row in mean_rows if row["beta"] > 0.0],
        key=lambda row: (row["score"], row["AP"], row["AP75"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["score_rank"] = rank
    mean_columns = [
        "score_rank", "guarded", "failed_guards", "positive_seed_count",
        "pw", "tau", "beta", "score",
    ] + list(METRICS) + [f"delta_{name}_pp" for name in METRICS]
    seed_columns = ["seed", "pw", "tau", "beta", "score"] + list(METRICS)
    write_csv(results_dir / "VAL_GRID_MEAN.csv", mean_rows, mean_columns)
    write_csv(results_dir / "VAL_GRID_PER_SEED.csv", per_seed_rows, seed_columns)

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        keys = (
            "pw", "tau", "beta", "score", "score_rank", "guarded",
            "failed_guards", "positive_seed_count",
        ) + METRICS + tuple(f"delta_{name}_pp" for name in METRICS)
        return {key: row.get(key) for key in keys if key in row}

    selected_model = None
    if selected is not None:
        selected_model = {
            "recommended_formal_seed": 0,
            "path": str(model_paths[(0, int(selected["pw"]))]),
            "sha256": sha256(model_paths[(0, int(selected["pw"]))]),
        }
    decision = {
        "protocol": f"{args.dataset_name} UMR Train candidates / Val-only selection",
        "test_accessed": False,
        "score_weights": SCORE_WEIGHTS,
        "max_safe_drop_metric_fraction": MAX_SAFE_DROP,
        "grids": {
            "seeds": list(SEEDS), "pw": list(POS_WEIGHTS),
            "tau": list(TAU_GRID), "beta": list(BETA_GRID),
        },
        "baseline_qccr_val70": baseline,
        "decision": "VAL_GO" if selected else "VAL_NO_GO",
        "selected": compact(selected),
        "selected_model": selected_model,
        "top5_by_val_score": [compact(row) for row in ranked[:5]],
        "eligible_candidate_count": len(eligible),
        "candidate_count_including_beta0": len(mean_rows),
        "detector_retrained": False,
        "qccr_refitted": False,
        "formal_source_modified": False,
        "official_test_accessed": False,
    }
    (results_dir / "VAL_SELECTION_DECISION.json").write_text(
        json.dumps(finite(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("one or more retained inputs changed during selection")
    provenance = {
        "data_policy": {
            "fit": [f"{args.dataset_name} Train frozen detector capture"],
            "select": [
                f"{args.dataset_name} Val frozen detector capture",
                f"{args.dataset_name} Val annotation",
            ],
            "forbidden": [f"{args.dataset_name} Test"],
            "test_accessed": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": before[name]}
            for name, path in sorted(inputs.items())
        },
        "inputs_unchanged": True,
        "candidate_training": finite(model_summaries),
    }
    (results_dir / "INPUT_PROVENANCE.json").write_text(
        json.dumps(finite(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== VAL-ONLY UMR DECISION =====")
    print("Decision:", decision["decision"])
    if selected:
        print(
            "Selected: pw={pw} tau={tau:.2f} beta={beta:.2f} "
            "score={score:+.6f} AP={AP:.9f} AP75={AP75:.9f}".format(**selected)
        )
    print("Inputs unchanged: PASS")
    print("Test accessed: NO")
    print("UMR TRAIN/VAL-ONLY PARAMETER SELECTION: PASS")


if __name__ == "__main__":
    main()
