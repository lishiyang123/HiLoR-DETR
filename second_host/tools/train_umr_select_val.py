#!/usr/bin/env python3
"""Train frozen UMR heads on Train and select fusion on Val only.

The grids and guards are inherited from the locked QCCR--UMR protocol.  This
program intentionally has no Test argument and no fallback split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from fit_common import (
        EXPECTED_CLASSES, EXPECTED_QUERIES, EXPECTED_TOPK, atomic_json,
        atomic_torch_save, cxcywh_to_xyxy, exact_metrics, install_source_root,
        load_capture, pairwise_iou, require_sha, run_coco, sha256_file,
        require_output_path, runtime_source_hashes, validate_annotation,
        xyxy_to_xywh, formal_fit_guard, runtime_source_path,
    )
except ImportError:  # imported as tools.train_umr_select_val in unit tests
    from tools.fit_common import (
    EXPECTED_CLASSES,
    EXPECTED_QUERIES,
    EXPECTED_TOPK,
    atomic_json,
    atomic_torch_save,
    cxcywh_to_xyxy,
    exact_metrics,
    install_source_root,
    load_capture,
    pairwise_iou,
    require_sha,
    run_coco,
    require_output_path,
    runtime_source_hashes,
    sha256_file,
    validate_annotation,
    xyxy_to_xywh,
    formal_fit_guard,
    runtime_source_path,
    )


IOU_LEVELS = (0.75, 0.80, 0.85, 0.90, 0.95)
SEEDS = (0, 1, 2)
POSITIVE_WEIGHTS = (1, 2, 4)
TAU_GRID = (0.0, 0.1, 0.2, 0.4)
BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SCORE_WEIGHTS = {"AP": 0.35, "AP75": 0.45, "APm": 0.10, "AR100": 0.10}
MAX_SAFE_DROP = 0.001
METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)
FEATURE_SCHEMA = "umr-25d-rtdetr-final-hidden-v1"


def compute_marginal_labels(
    gt_boxes, gt_labels, gt_iscrowd, boxes_absolute, qccr_score
):
    if tuple(qccr_score.shape) != (EXPECTED_QUERIES, EXPECTED_CLASSES):
        raise RuntimeError("unexpected QCCR score geometry")
    if gt_labels.numel() and (
        int(gt_labels.min()) < 0 or int(gt_labels.max()) >= EXPECTED_CLASSES
    ):
        raise RuntimeError("ground-truth category lies outside contiguous IDs 0..4")
    iou = pairwise_iou(boxes_absolute, gt_boxes)
    noncrowd = (gt_iscrowd == 0).bool()
    marginal = torch.zeros_like(qccr_score, dtype=torch.float32)
    for threshold in IOU_LEVELS:
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
            threshold_target[int(torch.argmax(scores)), class_index] = 1.0
        marginal += threshold_target
    return marginal / float(len(IOU_LEVELS))


def qfl_loss(logits, targets, positive_weight):
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = (probabilities - targets).abs().pow(2)
    weights *= torch.where(
        targets > 0,
        torch.full_like(targets, float(positive_weight)),
        torch.ones_like(targets),
    )
    return (weights * bce).mean()


def make_split(image_ids, seed):
    unique = sorted(set(image_ids))
    if len(unique) < 2:
        raise RuntimeError("Train capture must contain at least two images")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(unique))
    count = min(max(int(round(len(unique) * 0.9)), 1), len(unique) - 1)
    fit_ids = {unique[index] for index in permutation[:count]}
    holdout_ids = {unique[index] for index in permutation[count:]}
    fit = np.asarray([i for i, value in enumerate(image_ids) if value in fit_ids])
    holdout = np.asarray([i for i, value in enumerate(image_ids) if value in holdout_ids])
    return fit, holdout


def train_one(head_type, data, image_ids, seed, positive_weight, max_epochs=50):
    fit_indices, holdout_indices = make_split(image_ids, seed)
    torch.manual_seed(seed)
    model = head_type(num_classes=EXPECTED_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_holdout, best_epoch, best_state = float("inf"), -1, None
    patience = 0
    history = {"train": [], "holdout": []}
    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(fit_indices))
        train_losses = []
        for start in range(0, len(permutation), 16):
            index = fit_indices[permutation[start : start + 16]]
            loss = qfl_loss(
                model(data["hidden"][index], data["classes"][index], data["features"][index]),
                data["targets"][index],
                positive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        history["train"].append(float(np.mean(train_losses)))
        model.eval()
        holdout_losses = []
        with torch.no_grad():
            for start in range(0, len(holdout_indices), 16):
                index = holdout_indices[start : start + 16]
                holdout_losses.append(
                    float(
                        qfl_loss(
                            model(
                                data["hidden"][index],
                                data["classes"][index],
                                data["features"][index],
                            ),
                            data["targets"][index],
                            positive_weight,
                        )
                    )
                )
        holdout_loss = float(np.mean(holdout_losses))
        history["holdout"].append(holdout_loss)
        if holdout_loss < best_holdout - 1.0e-6:
            best_holdout = holdout_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 8:
            break
    if best_state is None:
        raise RuntimeError("UMR candidate training produced no best state")
    model.load_state_dict(best_state, strict=True)
    return (
        model,
        history,
        best_epoch,
        best_holdout,
        int(len(fit_indices)),
        int(len(holdout_indices)),
    )


def absolute_boxes(record):
    height, width = [int(value) for value in record["orig_size"].tolist()]
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return cxcywh_to_xyxy(record["pred_boxes"].float()) * scale


def build_train_data(
    capture, qccr, apply_qccr, compute_competition_risk, build_image_features
):
    features, classes, hidden, targets, image_ids = [], [], [], [], []
    positive_count = candidate_count = 0
    for record in capture["records"]:
        raw = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()
        boxes_abs = absolute_boxes(record)
        risk = compute_competition_risk(raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        calibrated = apply_qccr(
            raw, risk, qccr["beta0"], qccr["beta_positive"],
            qccr["beta_negative"], qccr["duplicate_penalty"],
        )
        calibrated_score = torch.sigmoid(calibrated)
        marginal = compute_marginal_labels(
            record["gt_boxes"].float(), record["gt_labels"].long(),
            record["gt_iscrowd"].long(), boxes_abs, calibrated_score,
        )
        flat_index, image_features, image_classes, image_hidden = build_image_features(
            raw,
            boxes,
            boxes_abs,
            record["hL"].float(),
            calibrated,
            calibrated_score,
            risk,
            EXPECTED_TOPK,
        )
        query_index = flat_index // EXPECTED_CLASSES
        class_index = flat_index % EXPECTED_CLASSES
        if not torch.equal(image_classes, class_index):
            raise RuntimeError("runtime UMR feature class mapping mismatch")
        selected_targets = marginal[query_index, class_index]
        features.append(image_features)
        classes.append(image_classes)
        hidden.append(image_hidden)
        targets.append(selected_targets)
        image_ids.append(int(record["image_id"]))
        positive_count += int((selected_targets > 0).sum())
        candidate_count += int(selected_targets.numel())
    return (
        {
            "features": torch.stack(features),
            "classes": torch.stack(classes),
            "hidden": torch.stack(hidden),
            "targets": torch.stack(targets),
        },
        image_ids,
        positive_count,
        candidate_count,
    )


def prepare_val(
    capture, qccr, apply_qccr, compute_competition_risk, build_image_features
):
    prepared = []
    for record in capture["records"]:
        raw = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()
        boxes_abs = absolute_boxes(record)
        risk = compute_competition_risk(raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        calibrated = apply_qccr(
            raw, risk, qccr["beta0"], qccr["beta_positive"],
            qccr["beta_negative"], qccr["duplicate_penalty"],
        )
        score = torch.sigmoid(calibrated)
        flat_index, features, classes, hidden = build_image_features(
            raw, boxes, boxes_abs, record["hL"].float(), calibrated, score, risk,
            EXPECTED_TOPK,
        )
        prepared.append(
            {
                "image_id": int(record["image_id"]),
                "raw": raw,
                "boxes": boxes,
                "boxes_absolute": boxes_abs,
                "flat_index": flat_index,
                "features": features,
                "classes": classes,
                "hidden": hidden,
            }
        )
    return prepared


def detections_for(
    prepared, predictions, tau, beta, qccr, apply_qccr, compute_competition_risk
):
    detections = []
    for index, record in enumerate(prepared):
        raw, boxes = record["raw"], record["boxes"]
        corrected = raw
        if predictions is not None and beta > 0.0:
            bonus = torch.zeros(
                EXPECTED_QUERIES * EXPECTED_CLASSES, dtype=torch.float32
            )
            bonus[record["flat_index"]] = torch.relu(predictions[index] - tau)
            corrected = raw + beta * bonus.reshape(EXPECTED_QUERIES, EXPECTED_CLASSES)
        risk = compute_competition_risk(corrected.unsqueeze(0), boxes.unsqueeze(0))[0]
        final_logit = apply_qccr(
            corrected, risk, qccr["beta0"], qccr["beta_positive"],
            qccr["beta_negative"], qccr["duplicate_penalty"],
        )
        scores, indices = torch.topk(torch.sigmoid(final_logit).flatten(), EXPECTED_TOPK)
        labels = indices % EXPECTED_CLASSES
        queries = indices // EXPECTED_CLASSES
        for label, candidate_score, box in zip(
            labels, scores, record["boxes_absolute"][queries]
        ):
            detections.append(
                {
                    "image_id": record["image_id"],
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(candidate_score),
                }
            )
    return detections


def weighted_score(metrics, baseline):
    return 100.0 * sum(
        weight * (metrics[name] - baseline[name])
        for name, weight in SCORE_WEIGHTS.items()
    )


def guards(row, baseline, per_seed):
    failed = []
    if row["beta"] <= 0:
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


def write_csv(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--expected-train-capture-sha", required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--qccr-artifact", type=Path, required=True)
    parser.add_argument("--expected-qccr-sha", required=True)
    parser.add_argument("--val-annotation", type=Path, required=True)
    parser.add_argument("--expected-val-annotation-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    formal_evidence = formal_fit_guard(
        tool_relative="tools/train_umr_select_val.py",
        adapter_root_path=args.adapter_root,
        source_root=args.source_root,
        require_umr=True,
    )

    output_dir = require_output_path(
        args.output_dir, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="UMR selection directory",
    )
    if output_dir.exists():
        raise RuntimeError(f"refusing to reuse UMR selection directory: {output_dir}")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = require_sha(
        checkpoint, args.expected_checkpoint_sha, "detector checkpoint"
    )
    train_path, val_path = args.train_capture.resolve(), args.val_capture.resolve()
    train_capture = load_capture(
        train_path,
        expected_sha=args.expected_train_capture_sha,
        split="train",
        expected_checkpoint_sha=checkpoint_sha,
        adapter_root_path=args.adapter_root,
        detector_seed=args.detector_seed,
    )
    val_capture = load_capture(
        val_path,
        expected_sha=args.expected_val_capture_sha,
        split="val",
        expected_checkpoint_sha=checkpoint_sha,
        adapter_root_path=args.adapter_root,
        detector_seed=args.detector_seed,
    )
    if train_capture["data_split_lock_sha256"] != val_capture["data_split_lock_sha256"]:
        raise RuntimeError("Train and Val captures use different data locks")
    annotation = args.val_annotation.resolve()
    validate_annotation(annotation, args.expected_val_annotation_sha, val_capture)
    qccr_path = require_output_path(
        args.qccr_artifact, namespace="inference",
        detector_seed=args.detector_seed, adapter_root_path=args.adapter_root,
        label="frozen QCCR input",
    )
    qccr_sha = require_sha(qccr_path, args.expected_qccr_sha, "QCCR artifact")
    qccr_artifact = json.loads(qccr_path.read_text(encoding="utf-8"))
    if (
        qccr_artifact.get("method") != "QCCR-v1"
        or qccr_artifact.get("dataset") != "CDW1630"
        or qccr_artifact.get("source_checkpoint_sha256") != checkpoint_sha
        or qccr_artifact.get("test_used_for_selection") is not False
        or qccr_artifact.get("selection_decision") != "OOF_VAL_GO"
        or qccr_artifact.get("detector_seed") != args.detector_seed
        or qccr_artifact.get("implementation_lock_sha256")
        != formal_evidence["implementation_lock_sha256"]
    ):
        raise RuntimeError("invalid or mismatched frozen QCCR artifact")
    if qccr_artifact.get("source_capture_sha256") != args.expected_val_capture_sha:
        raise RuntimeError("QCCR artifact was not fitted from this exact Val capture")
    if qccr_artifact.get("source_data_split_lock_sha256") != val_capture.get(
        "data_split_lock_sha256"
    ):
        raise RuntimeError("QCCR artifact uses another data-split lock")
    qccr = {key: float(value) for key, value in qccr_artifact["parameters"].items()}

    source_root = args.source_root.resolve()
    runtime_qccr = runtime_source_path(source_root, "hilor_qccr.py")
    runtime_umr = runtime_source_path(source_root, "hilor_umr.py")
    inputs = {
        "detector_checkpoint": checkpoint,
        "train_capture": train_path,
        "val_capture": val_path,
        "qccr_artifact": qccr_path,
        "val_annotation": annotation,
        "runtime_qccr": runtime_qccr,
        "runtime_umr": runtime_umr,
    }
    for path in inputs.values():
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")
    before = {name: sha256_file(path) for name, path in inputs.items()}
    runtime_hashes = runtime_source_hashes(source_root, require_umr=True)
    qccr_runtime = qccr_artifact.get("runtime_source_sha256")
    if not isinstance(qccr_runtime, dict) or qccr_runtime.get(
        "hilor_qccr.py"
    ) != runtime_hashes["hilor_qccr.py"]:
        raise RuntimeError("QCCR artifact was fitted with another QCCR runtime")
    apply_qccr, compute_competition_risk, UMRHead, build_image_features = (
        install_source_root(source_root, require_umr=True)
    )
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "training_reports"
    results_dir = output_dir / "results"
    models_dir.mkdir()
    reports_dir.mkdir()
    results_dir.mkdir()

    train_data, train_image_ids, positive_count, candidate_count = build_train_data(
        train_capture, qccr, apply_qccr, compute_competition_risk, build_image_features
    )
    model_paths = {}
    model_summaries = []
    for seed in SEEDS:
        for positive_weight in POSITIVE_WEIGHTS:
            model, history, best_epoch, best_holdout, fit_count, holdout_count = train_one(
                UMRHead, train_data, train_image_ids, seed, positive_weight
            )
            path = models_dir / f"umr_seed{seed}_pw{positive_weight}_best.pt"
            atomic_torch_save(path, model.state_dict())
            model_paths[(seed, positive_weight)] = path
            summary = {
                "schema": "rtdetr-hilor-umr-training-report-v1",
                "test_accessed": False,
                "seed": seed,
                "detector_seed": args.detector_seed,
                "positive_weight": positive_weight,
                "source_split": "CDW1630 Train",
                "source_checkpoint_sha256": checkpoint_sha,
                "train_capture_sha256": args.expected_train_capture_sha,
                "val_capture_sha256": args.expected_val_capture_sha,
                "source_data_split_lock_sha256": train_capture[
                    "data_split_lock_sha256"
                ],
                "qccr_artifact_sha256": qccr_sha,
                "qccr_source_capture_sha256": qccr_artifact[
                    "source_capture_sha256"
                ],
                "runtime_source_sha256": runtime_hashes,
                "weight_path": str(path),
                "weight_sha256": sha256_file(path),
                "feature_schema": FEATURE_SCHEMA,
                "head_parameters": sum(value.numel() for value in model.parameters()),
                "train_image_count": len(train_image_ids),
                "fit_image_count": fit_count,
                "holdout_image_count": holdout_count,
                "epochs_run": len(history["train"]),
                "best_epoch": best_epoch,
                "best_holdout_qfl": best_holdout,
                "positive_target_rate_m300": positive_count / max(candidate_count, 1),
                "history": history,
            }
            atomic_json(
                reports_dir / f"umr_seed{seed}_pw{positive_weight}.json", summary
            )
            model_summaries.append(summary)

    prepared = prepare_val(
        val_capture, qccr, apply_qccr, compute_competition_risk, build_image_features
    )
    image_ids = [record["image_id"] for record in prepared]
    baseline = exact_metrics(
        run_coco(
            annotation,
            detections_for(
                prepared, None, 0.0, 0.0, qccr, apply_qccr, compute_competition_risk
            ),
            image_ids,
        )
    )
    per_seed_rows = []
    for seed in SEEDS:
        for positive_weight in POSITIVE_WEIGHTS:
            model = UMRHead(num_classes=EXPECTED_CLASSES)
            model.load_state_dict(
                torch.load(
                    model_paths[(seed, positive_weight)],
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            model.eval()
            predictions = []
            with torch.inference_mode():
                for record in prepared:
                    predictions.append(
                        torch.sigmoid(
                            model(record["hidden"], record["classes"], record["features"])
                        )
                    )
            for tau in TAU_GRID:
                for beta in BETA_GRID:
                    metrics = baseline
                    if beta > 0:
                        metrics = exact_metrics(
                            run_coco(
                                annotation,
                                detections_for(
                                    prepared, predictions, tau, beta, qccr,
                                    apply_qccr, compute_competition_risk,
                                ),
                                image_ids,
                            )
                        )
                    row = {
                        "seed": seed, "pw": positive_weight,
                        "tau": tau, "beta": beta, **metrics,
                    }
                    row["score"] = weighted_score(row, baseline)
                    per_seed_rows.append(row)

    mean_rows = []
    for positive_weight in POSITIVE_WEIGHTS:
        for tau in TAU_GRID:
            for beta in BETA_GRID:
                group = [
                    row for row in per_seed_rows
                    if row["pw"] == positive_weight and row["tau"] == tau
                    and row["beta"] == beta
                ]
                if len(group) != len(SEEDS):
                    raise RuntimeError("incomplete UMR seed aggregation")
                row = {"pw": positive_weight, "tau": tau, "beta": beta}
                for name in METRICS:
                    row[name] = float(np.mean([item[name] for item in group]))
                    row[f"delta_{name}_pp"] = 100.0 * (row[name] - baseline[name])
                row["score"] = weighted_score(row, baseline)
                allowed, failed = guards(row, baseline, group)
                row["guarded"] = bool(allowed)
                row["failed_guards"] = ";".join(failed)
                mean_rows.append(row)
    eligible = sorted(
        [row for row in mean_rows if row["guarded"]],
        key=lambda row: (row["score"], row["AP"], row["AP75"]),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    ranked = sorted(
        [row for row in mean_rows if row["beta"] > 0],
        key=lambda row: (row["score"], row["AP"], row["AP75"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        row["score_rank"] = rank
    selected_model = None
    if selected is not None:
        selected_path = model_paths[(0, int(selected["pw"]))]
        selected_model = {
            "recommended_formal_seed": 0,
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
        }
    decision = {
        "schema": "rtdetr-hilor-umr-val-selection-v1",
        "protocol": "CDW1630 Train candidates / Val-only selection",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": args.detector_seed,
        "test_accessed": False,
        "feature_schema": FEATURE_SCHEMA,
        "source_checkpoint_sha256": checkpoint_sha,
        "train_capture_sha256": args.expected_train_capture_sha,
        "val_capture_sha256": args.expected_val_capture_sha,
        "source_data_split_lock_sha256": train_capture["data_split_lock_sha256"],
        "qccr_artifact_sha256": qccr_sha,
        "qccr_source_capture_sha256": qccr_artifact["source_capture_sha256"],
        "runtime_source_sha256": runtime_hashes,
        "score_weights": SCORE_WEIGHTS,
        "max_safe_drop_metric_fraction": MAX_SAFE_DROP,
        "grids": {
            "seeds": list(SEEDS), "positive_weights": list(POSITIVE_WEIGHTS),
            "tau": list(TAU_GRID), "beta": list(BETA_GRID),
        },
        "baseline_qccr_val": baseline,
        "decision": "VAL_GO" if selected is not None else "VAL_NO_GO",
        "selected": selected,
        "selected_model": selected_model,
        "top5_by_val_score": ranked[:5],
        "eligible_candidate_count": len(eligible),
        "candidate_count_including_beta0": len(mean_rows),
        "detector_retrained": False,
        "qccr_refitted": False,
        "official_test_accessed": False,
        "implementation_lock_sha256": formal_evidence[
            "implementation_lock_sha256"
        ],
    }
    atomic_json(results_dir / "VAL_SELECTION_DECISION.json", decision)
    mean_columns = [
        "score_rank", "guarded", "failed_guards", "positive_seed_count",
        "pw", "tau", "beta", "score", *METRICS,
        *[f"delta_{name}_pp" for name in METRICS],
    ]
    write_csv(results_dir / "VAL_GRID_MEAN.csv", mean_rows, mean_columns)
    write_csv(
        results_dir / "VAL_GRID_PER_SEED.csv",
        per_seed_rows,
        ["seed", "pw", "tau", "beta", "score", *METRICS],
    )
    after = {name: sha256_file(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("one or more retained inputs changed during UMR selection")
    input_provenance = {
        "schema": "rtdetr-hilor-umr-input-provenance-v1",
        "test_accessed": False,
        "data_policy": {
            "fit": ["CDW1630 Train frozen detector capture"],
            "select": ["CDW1630 Val frozen detector capture", "CDW1630 Val annotation"],
            "forbidden": ["CDW1630 Test"],
        },
        "inputs": {
            name: {"path": str(path), "sha256": before[name]}
            for name, path in sorted(inputs.items())
        },
        "inputs_unchanged": True,
        "candidate_training": model_summaries,
    }
    atomic_json(
        results_dir / "INPUT_PROVENANCE.json",
        input_provenance,
    )
    print("RT-DETR UMR TRAIN/VAL-ONLY SELECTION: PASS")
    print("Decision:", decision["decision"])
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
