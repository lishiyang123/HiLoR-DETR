"""CCVR value fitting, frozen-candidate replay and aggregation on CPU."""
from __future__ import annotations
import copy
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


NUM_CLASSES = 5


NUM_QUERIES = 300


TOPK = 300


IOU_LEVELS = (0.75, 0.80, 0.85, 0.90, 0.95)


METRICS = (
    "AP",
    "AP50",
    "AP75",
    "AP80",
    "AP85",
    "AP90",
    "AP95",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


ROWS = (
    "qccr_only",
    "current_train_capture_umr",
    "val_oof_umr",
    "train_matched_umr",
)


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
    require(
        (num_queries, num_classes) == (NUM_QUERIES, NUM_CLASSES),
        "Unexpected QCCR score shape: %r" % (tuple(qccr_score.shape),),
    )
    if gt_labels.numel():
        require(
            int(gt_labels.min()) >= 0 and int(gt_labels.max()) < NUM_CLASSES,
            "Ground-truth category is outside contiguous IDs 0..4",
        )
    iou = box_iou_xyxy(boxes_absolute, gt_boxes)
    noncrowd = (gt_iscrowd == 0).bool()
    marginal = torch.zeros(num_queries, num_classes, dtype=torch.float32)
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
            query_index = int(torch.argmax(scores))
            threshold_target[query_index, class_index] = 1.0
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


def inner_split(image_ids: Sequence[int], seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_ids = sorted(set(int(value) for value in image_ids))
    require(len(unique_ids) >= 2, "UMR fitting needs at least two images")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(unique_ids))
    n_fit = min(max(int(round(len(unique_ids) * 0.9)), 1), len(unique_ids) - 1)
    fit_ids = {unique_ids[index] for index in permutation[:n_fit]}
    holdout_ids = {unique_ids[index] for index in permutation[n_fit:]}
    fit_idx = torch.tensor(
        [index for index, value in enumerate(image_ids) if int(value) in fit_ids],
        dtype=torch.long,
    )
    holdout_idx = torch.tensor(
        [index for index, value in enumerate(image_ids) if int(value) in holdout_ids],
        dtype=torch.long,
    )
    require(len(fit_idx) > 0 and len(holdout_idx) > 0, "Invalid inner split")
    return fit_idx, holdout_idx


def train_one(
    head_type: Any,
    data: Dict[str, torch.Tensor],
    image_ids: Sequence[int],
    seed: int = 0,
    positive_weight: int = 4,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    fit_idx, holdout_idx = inner_split(image_ids, seed)
    torch.manual_seed(seed)
    model = head_type(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_holdout = float("inf")
    best_epoch = -1
    best_state = None
    patience_count = 0
    history = {"train": [], "holdout": []}
    for epoch in range(50):
        model.train()
        permutation = fit_idx[torch.randperm(len(fit_idx))]
        train_total = 0.0
        train_batches = 0
        for start in range(0, len(permutation), 16):
            index = permutation[start : start + 16]
            loss = qfl_loss(
                model(data["hidden"][index], data["classes"][index], data["features"][index]),
                data["targets"][index],
                positive_weight=float(positive_weight),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            train_batches += 1
        train_loss = train_total / max(train_batches, 1)
        model.eval()
        holdout_total = 0.0
        holdout_batches = 0
        with torch.inference_mode():
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
        history["train"].append(train_loss)
        history["holdout"].append(holdout_loss)
        if holdout_loss < best_holdout - 1.0e-6:
            best_holdout = holdout_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 8:
            break
    require(best_state is not None, "UMR training produced no best state")
    model.load_state_dict(best_state)
    report = {
        "head_seed": int(seed),
        "positive_weight": int(positive_weight),
        "fit_image_count": int(len(fit_idx)),
        "inner_holdout_image_count": int(len(holdout_idx)),
        "epochs_run": int(len(history["train"])),
        "best_epoch": int(best_epoch),
        "best_holdout_qfl": float(best_holdout),
        "history": history,
    }
    return model, report


def deterministic_subset_ids(
    all_image_ids: Iterable[int], count: int, fold: int, selection_seed: int
) -> List[int]:
    unique_ids = sorted(set(int(value) for value in all_image_ids))
    require(count <= len(unique_ids), "Matched Train subset is larger than Train")
    scored = []
    for image_id in unique_ids:
        key = "%d:%d:%d" % (selection_seed, fold, image_id)
        score = hashlib.sha256(key.encode("ascii")).hexdigest()
        scored.append((score, image_id))
    scored.sort()
    return sorted(image_id for _score, image_id in scored[:count])


def xyxy_to_xywh(box: torch.Tensor) -> List[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ]


def mean_valid(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[array > -1]
    return float(array.mean()) if array.size else float("nan")


def exact_metrics(evaluation: Dict[str, Any]) -> Dict[str, float]:
    precision = np.asarray(evaluation["precision"], dtype=np.float64)
    recall = np.asarray(evaluation["recall"], dtype=np.float64)
    params = evaluation["params"]
    ious = np.asarray(params.iouThrs)
    areas = list(params.areaRngLbl)
    max_dets = list(params.maxDets)
    category_ids = [int(value) for value in params.catIds]
    require(category_ids == list(range(NUM_CLASSES)), "Val category IDs are not 0..4")

    def threshold(value: float) -> int:
        found = np.where(np.isclose(ious, value))[0]
        require(len(found) == 1, "Missing IoU threshold %.2f" % value)
        return int(found[0])

    def ap(iou: Optional[float] = None, area: str = "all", max_det: int = 100) -> float:
        selected = precision[:, :, :, areas.index(area), max_dets.index(max_det)]
        if iou is not None:
            selected = precision[threshold(iou), :, :, areas.index(area), max_dets.index(max_det)]
        return mean_valid(selected)

    def ar(area: str = "all", max_det: int = 100) -> float:
        return mean_valid(recall[:, :, areas.index(area), max_dets.index(max_det)])

    return {
        "AP": ap(),
        "AP50": ap(0.50),
        "AP75": ap(0.75),
        "AP80": ap(0.80),
        "AP85": ap(0.85),
        "AP90": ap(0.90),
        "AP95": ap(0.95),
        "APs": ap(area="small"),
        "APm": ap(area="medium"),
        "APl": ap(area="large"),
        "AR1": ar(max_det=1),
        "AR10": ar(max_det=10),
        "AR100": ar(max_det=100),
        "ARs": ar(area="small"),
        "ARm": ar(area="medium"),
        "ARl": ar(area="large"),
    }


def run_coco(
    annotation: Path, detections: List[Dict[str, Any]], image_ids: Sequence[int]
) -> Dict[str, float]:
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(
        coco_gt,
        iouType="bbox",
        print_function=lambda *_args, **_kwargs: None,
        separate_eval=True,
    )
    evaluator.cocoDt = coco_dt
    evaluator.params.imgIds = sorted(int(value) for value in image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    return exact_metrics(evaluator.eval)


def load_capture(path: Path, split: str, detector_sha: str) -> Dict[str, Any]:
    capture = torch.load(path, map_location="cpu", weights_only=False)
    require(capture.get("split") == split, "%s capture split mismatch" % split)
    require(
        capture.get("method") in {"PD", "frozen_native_detector"},
        "%s capture method is not frozen native detector" % split,
    )
    require(capture.get("checkpoint_sha256") == detector_sha, "%s capture/checkpoint mismatch" % split)
    require(int(capture.get("postprocessor_num_classes", -1)) == NUM_CLASSES, "Class count mismatch")
    require(int(capture.get("postprocessor_num_top_queries", -1)) == TOPK, "Top-k mismatch")
    return capture


def fold_parameters(fits: Dict[str, Any], candidate_key: str) -> Dict[int, Dict[str, float]]:
    entries = fits["candidates"][candidate_key]["folds"]
    result = {
        int(entry["fold"]): {key: float(value) for key, value in entry["parameters"].items()}
        for entry in entries
    }
    require(set(result) == {0, 1, 2, 3, 4}, "Incomplete QCCR fold coefficients")
    return result


def prepare_record(
    record: Dict[str, Any],
    parameters: Dict[str, float],
    apply_qccr: Any,
    compute_competition_risk: Any,
    box_cxcywh_to_xyxy: Any,
    build_image_features: Any,
    with_target: bool,
) -> Dict[str, Any]:
    z_raw = record["pred_logits"].float()
    boxes = record["pred_boxes"].float()
    height, width = [int(value) for value in record["orig_size"].tolist()]
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    boxes_absolute = box_cxcywh_to_xyxy(boxes) * scale
    risk0 = compute_competition_risk(z_raw.unsqueeze(0), boxes.unsqueeze(0))[0]
    qccr_logit0 = apply_qccr(
        z_raw,
        risk0,
        parameters["beta0"],
        parameters["beta_positive"],
        parameters["beta_negative"],
        parameters["duplicate_penalty"],
    )
    qccr_score0 = torch.sigmoid(qccr_logit0)
    flat_idx, features, classes, hidden = build_image_features(
        z_raw,
        boxes,
        boxes_absolute,
        record["hL"].float(),
        qccr_logit0,
        qccr_score0,
        risk0,
        K=TOPK,
    )
    query_idx = flat_idx // NUM_CLASSES
    class_idx = flat_idx % NUM_CLASSES
    require(torch.equal(classes, class_idx), "UMR class mapping mismatch")
    prepared = {
        "image_id": int(record["image_id"]),
        "z_raw": z_raw,
        "boxes": boxes,
        "boxes_absolute": boxes_absolute,
        "flat_idx": flat_idx,
        "features": features,
        "classes": classes,
        "hidden": hidden,
    }
    if with_target:
        marginal = compute_marginal_labels(
            record["gt_boxes"],
            record["gt_labels"],
            record["gt_iscrowd"],
            boxes_absolute,
            qccr_score0,
        )
        prepared["targets"] = marginal[query_idx, class_idx]
    return prepared


def stack_training_data(records: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    require(len(records) > 1, "Training record set is too small")
    return {
        "features": torch.stack([record["features"] for record in records]),
        "classes": torch.stack([record["classes"] for record in records]),
        "hidden": torch.stack([record["hidden"] for record in records]),
        "targets": torch.stack([record["targets"] for record in records]),
    }


def predict(model: torch.nn.Module, record: Dict[str, Any]) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        logits = model(
            record["hidden"].unsqueeze(0),
            record["classes"].unsqueeze(0),
            record["features"].unsqueeze(0),
        )
    return torch.sigmoid(logits).squeeze(0)


def detections_for_record(
    record: Dict[str, Any],
    prediction: Optional[torch.Tensor],
    parameters: Dict[str, float],
    apply_qccr: Any,
    compute_competition_risk: Any,
    tau: float = 0.0,
    beta: float = 2.0,
) -> List[Dict[str, Any]]:
    z_raw = record["z_raw"]
    boxes = record["boxes"]
    if prediction is None:
        z_umr = z_raw
    else:
        bonus = torch.zeros(NUM_QUERIES * NUM_CLASSES, dtype=torch.float32)
        bonus[record["flat_idx"]] = torch.relu(prediction - tau)
        z_umr = z_raw + beta * bonus.reshape(NUM_QUERIES, NUM_CLASSES)
    risk1 = compute_competition_risk(z_umr.unsqueeze(0), boxes.unsqueeze(0))[0]
    final_logit = apply_qccr(
        z_umr,
        risk1,
        parameters["beta0"],
        parameters["beta_positive"],
        parameters["beta_negative"],
        parameters["duplicate_penalty"],
    )
    top_scores, top_indices = torch.topk(torch.sigmoid(final_logit).flatten(), k=TOPK)
    labels = top_indices % NUM_CLASSES
    query_indices = top_indices // NUM_CLASSES
    selected_boxes = record["boxes_absolute"][query_indices]
    return [
        {
            "image_id": record["image_id"],
            "category_id": int(label),
            "bbox": xyxy_to_xywh(box),
            "score": float(score),
        }
        for label, score, box in zip(labels, top_scores, selected_boxes)
    ]


def run_seed(
    seed: int,
    seed_lock: Dict[str, Any],
    lock: Dict[str, Any],
    output_dir: Path,
    protocol: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    detector_sha = seed_lock["detector_checkpoint"]["sha256"]
    train_capture = load_capture(Path(seed_lock["train_capture"]["path"]), "train", detector_sha)
    val_capture = load_capture(Path(seed_lock["val_capture"]["path"]), "val", detector_sha)
    fits = read_json(Path(seed_lock["qccr_fits"]["path"]))
    fold_by_image = {int(key): int(value) for key, value in fits["fold_by_image"].items()}
    require(
        {str(key): value for key, value in sorted(fold_by_image.items())}
        == lock["shared_fold_by_image"],
        "Seed fold map differs from shared lock",
    )
    parameters_by_fold = fold_parameters(fits, protocol["qccr"]["candidate_key"])
    val_records = {int(record["image_id"]): record for record in val_capture["records"]}
    train_records = {int(record["image_id"]): record for record in train_capture["records"]}
    val_image_count = int(protocol["split_counts"]["val"]["images"])
    train_image_count = int(protocol["split_counts"]["train"]["images"])
    require(len(val_records) == val_image_count and set(val_records) == set(fold_by_image), "Val capture/fold mismatch")
    require(len(train_records) == train_image_count, "Train capture/annotation count mismatch")

    CandidateValueHead = runtime["CandidateValueHead"]
    current_head = CandidateValueHead(num_classes=NUM_CLASSES)
    current_state = torch.load(
        Path(seed_lock["formal_umr_head"]["path"]), map_location="cpu", weights_only=False
    )
    current_head.load_state_dict(current_state, strict=True)
    current_head.eval()

    detections = {row: [] for row in ROWS}
    fold_reports = []
    seed_dir = output_dir / ("seed%d" % seed)
    heads_dir = seed_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    selection_seed = int(protocol["train_matched_control"]["selection_seed"])
    for fold in range(5):
        qccr = parameters_by_fold[fold]
        target_ids = sorted(image_id for image_id, value in fold_by_image.items() if value == fold)
        outer_fit_ids = sorted(image_id for image_id, value in fold_by_image.items() if value != fold)
        require(set(target_ids).isdisjoint(outer_fit_ids), "Outer fold leakage")
        require(len(target_ids) + len(outer_fit_ids) == val_image_count, "Outer fold coverage mismatch")
        print("seed%d fold%d: fitting Val-OOF head" % (seed, fold), flush=True)

        val_fit_prepared = [
            prepare_record(
                val_records[image_id], qccr, runtime["apply_qccr"],
                runtime["compute_competition_risk"], runtime["box_cxcywh_to_xyxy"],
                runtime["build_image_features"], True,
            )
            for image_id in outer_fit_ids
        ]
        val_model, val_training = train_one(
            CandidateValueHead,
            stack_training_data(val_fit_prepared),
            [record["image_id"] for record in val_fit_prepared],
            seed=0,
            positive_weight=4,
        )
        val_head_path = heads_dir / ("fold%d_val_oof_umr.pt" % fold)
        torch.save(val_model.state_dict(), val_head_path)

        matched_ids = deterministic_subset_ids(
            train_records.keys(), len(outer_fit_ids), fold, selection_seed
        )
        matched_prepared = [
            prepare_record(
                train_records[image_id], qccr, runtime["apply_qccr"],
                runtime["compute_competition_risk"], runtime["box_cxcywh_to_xyxy"],
                runtime["build_image_features"], True,
            )
            for image_id in matched_ids
        ]
        print("seed%d fold%d: fitting matched-Train head" % (seed, fold), flush=True)
        matched_model, matched_training = train_one(
            CandidateValueHead,
            stack_training_data(matched_prepared),
            [record["image_id"] for record in matched_prepared],
            seed=0,
            positive_weight=4,
        )
        matched_head_path = heads_dir / ("fold%d_train_matched_umr.pt" % fold)
        torch.save(matched_model.state_dict(), matched_head_path)

        for image_id in target_ids:
            prepared = prepare_record(
                val_records[image_id], qccr, runtime["apply_qccr"],
                runtime["compute_competition_risk"], runtime["box_cxcywh_to_xyxy"],
                runtime["build_image_features"], False,
            )
            current_prediction = predict(current_head, prepared)
            val_prediction = predict(val_model, prepared)
            matched_prediction = predict(matched_model, prepared)
            predictions = {
                "qccr_only": None,
                "current_train_capture_umr": current_prediction,
                "val_oof_umr": val_prediction,
                "train_matched_umr": matched_prediction,
            }
            for row, prediction in predictions.items():
                detections[row].extend(
                    detections_for_record(
                        prepared,
                        prediction,
                        qccr,
                        runtime["apply_qccr"],
                        runtime["compute_competition_risk"],
                        tau=0.0,
                        beta=2.0,
                    )
                )
        fold_report = {
            "fold": fold,
            "target_image_count": len(target_ids),
            "outer_val_fit_image_count": len(outer_fit_ids),
            "matched_train_image_count": len(matched_ids),
            "target_image_ids_sha256": hashlib.sha256(
                json.dumps(target_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "outer_val_fit_image_ids_sha256": hashlib.sha256(
                json.dumps(outer_fit_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "matched_train_image_ids_sha256": hashlib.sha256(
                json.dumps(matched_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "qccr_parameters": qccr,
            "val_oof_head": {
                "path": str(val_head_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": sha256(val_head_path),
                "training": val_training,
            },
            "train_matched_head": {
                "path": str(matched_head_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": sha256(matched_head_path),
                "training": matched_training,
            },
            "target_fold_used_for_fitting": False,
        }
        fold_reports.append(fold_report)
        print(
            "seed%d fold%d: target=%d outer-fit=%d matched-train=%d PASS"
            % (seed, fold, len(target_ids), len(outer_fit_ids), len(matched_ids)),
            flush=True,
        )
        del val_fit_prepared, matched_prepared, val_model, matched_model
        gc.collect()

    metrics = {}
    for row in ROWS:
        require(len(detections[row]) == val_image_count * TOPK, "%s detection count mismatch" % row)
        metrics[row] = run_coco(
            Path(lock["val_annotation"]["path"]), detections[row], sorted(val_records)
        )
        with gzip.open(seed_dir / (row + "_predictions.json.gz"), "wt", encoding="utf-8") as stream:
            json.dump([{key: d[key] for key in ("image_id", "category_id", "bbox", "score")}
                       for d in detections[row]], stream, separators=(",", ":"), allow_nan=False)
    baseline = metrics["qccr_only"]
    deltas = {
        row: {metric: 100.0 * (metrics[row][metric] - baseline[metric]) for metric in METRICS}
        for row in ROWS
    }
    payload = {
        "format_version": 1,
        "dataset": "CDW1630",
        "detector_seed": seed,
        "rows": list(ROWS),
        "metrics": metrics,
        "delta_vs_qccr_only_pp": deltas,
        "fold_reports": fold_reports,
        "val_image_count": val_image_count,
        "detections_per_row": val_image_count * TOPK,
        "oof_aggregation": "held-out detections merged before one COCO evaluation",
        "qccr_lambda_reg": 0.001,
        "umr_tuple": {"positive_weight": 4, "tau": 0.0, "beta": 2.0, "topk": 300, "head_seed": 0},
        "detector_retrained": False,
        "test_accessed": False,
        "no_result_based_selection": True,
    }
    write_json_atomic(seed_dir / "OOF_METRICS.json", finite(payload))
    with (seed_dir / "OOF_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["row"] + list(METRICS) + ["delta_%s_pp" % metric for metric in METRICS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in ROWS:
            record = {"row": row}
            record.update(metrics[row])
            record.update({"delta_%s_pp" % key: value for key, value in deltas[row].items()})
            writer.writerow(record)
    del train_capture, val_capture, detections, train_records, val_records
    gc.collect()
    return payload


def aggregate(seed_payloads: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    rows_json = []
    rows_csv = []
    for row in ROWS:
        record = {"row": row, "metrics": {}, "delta_vs_qccr_only_pp": {}}
        csv_record = {"row": row}
        for metric in METRICS:
            values = np.asarray(
                [payload["metrics"][row][metric] for payload in seed_payloads], dtype=np.float64
            )
            deltas = np.asarray(
                [payload["delta_vs_qccr_only_pp"][row][metric] for payload in seed_payloads],
                dtype=np.float64,
            )
            record["metrics"][metric] = {
                "per_seed": [float(value) for value in values],
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
            record["delta_vs_qccr_only_pp"][metric] = {
                "per_seed": [float(value) for value in deltas],
                "mean": float(np.mean(deltas)),
                "sample_std": float(np.std(deltas, ddof=1)),
            }
            csv_record["%s_mean" % metric] = float(np.mean(values))
            csv_record["%s_sample_std" % metric] = float(np.std(values, ddof=1))
            csv_record["delta_%s_pp_mean" % metric] = float(np.mean(deltas))
            csv_record["delta_%s_pp_sample_std" % metric] = float(np.std(deltas, ddof=1))
        rows_json.append(record)
        rows_csv.append(csv_record)
    summary = {
        "format_version": 1,
        "dataset": "CDW1630",
        "detector_seeds": [0, 1, 2],
        "fold_count": 5,
        "rows": list(ROWS),
        "results": rows_json,
        "interpretation_scope": "Val-only robustness diagnostic; no formal Test result replacement",
        "detector_retrained": False,
        "test_accessed": False,
        "no_result_based_selection": True,
    }
    write_json_atomic(output_dir / "THREE_SEED_SUMMARY.json", finite(summary))
    columns = ["row"]
    for metric in METRICS:
        columns.extend(
            [
                "%s_mean" % metric,
                "%s_sample_std" % metric,
                "delta_%s_pp_mean" % metric,
                "delta_%s_pp_sample_std" % metric,
            ]
        )
    with (output_dir / "THREE_SEED_SUMMARY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_csv)
