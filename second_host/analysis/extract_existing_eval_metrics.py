#!/usr/bin/env python3
"""Extract fixed-IoU and three-seed metrics from existing COCO evidence only.

No model, checkpoint, image tensor, or GPU is loaded.  Native RT-DETR is
re-accumulated from its already frozen prediction JSON; all other rows are read
from saved COCO ``eval`` payloads.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


STANDARD_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)
STRICT_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
SEEDS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_torch_eval(path: Path) -> Mapping[str, Any]:
    payload = torch.load(require_file(path), map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "precision" in payload and "recall" in payload:
        return payload
    if isinstance(payload, Mapping):
        for key in ("eval", "bbox", "coco_eval_bbox"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and "precision" in nested and "recall" in nested:
                return nested
    raise RuntimeError(f"unsupported COCO eval payload: {path}")


def eval_params(payload: Mapping[str, Any]):
    params = payload.get("params")
    if params is None:
        raise RuntimeError("COCO eval payload has no params")
    return params


def mean_valid(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def exact_index(values, target: float, label: str) -> int:
    values = np.asarray(values, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise RuntimeError(f"{label}={target} has {matches.size} matches")
    return int(matches[0])


def label_index(values, target: str, label: str) -> int:
    values = list(values)
    if values.count(target) != 1:
        raise RuntimeError(f"{label}={target!r} is not unique")
    return values.index(target)


def metric_bundle(payload: Mapping[str, Any]) -> dict[str, float]:
    params = eval_params(payload)
    precision = np.asarray(payload["precision"], dtype=np.float64)
    recall = np.asarray(payload["recall"], dtype=np.float64)
    if precision.ndim != 5 or recall.ndim != 4:
        raise RuntimeError("unexpected COCO precision/recall dimensions")
    all_area = label_index(params.areaRngLbl, "all", "area")
    small = label_index(params.areaRngLbl, "small", "area")
    medium = label_index(params.areaRngLbl, "medium", "area")
    large = label_index(params.areaRngLbl, "large", "area")
    md1 = exact_index(params.maxDets, 1, "maxDets")
    md10 = exact_index(params.maxDets, 10, "maxDets")
    md100 = exact_index(params.maxDets, 100, "maxDets")
    t50 = exact_index(params.iouThrs, 0.50, "IoU")
    t75 = exact_index(params.iouThrs, 0.75, "IoU")
    result = {
        "AP": mean_valid(precision[:, :, :, all_area, md100]),
        "AP50": mean_valid(precision[t50, :, :, all_area, md100]),
        "AP75": mean_valid(precision[t75, :, :, all_area, md100]),
        "APs": mean_valid(precision[:, :, :, small, md100]),
        "APm": mean_valid(precision[:, :, :, medium, md100]),
        "APl": mean_valid(precision[:, :, :, large, md100]),
        "AR1": mean_valid(recall[:, :, all_area, md1]),
        "AR10": mean_valid(recall[:, :, all_area, md10]),
        "AR100": mean_valid(recall[:, :, all_area, md100]),
        "ARs": mean_valid(recall[:, :, small, md100]),
        "ARm": mean_valid(recall[:, :, medium, md100]),
        "ARl": mean_valid(recall[:, :, large, md100]),
    }
    for threshold in STRICT_THRESHOLDS:
        index = exact_index(params.iouThrs, threshold, "IoU")
        result[f"AP{int(round(threshold * 100))}"] = mean_valid(
            precision[index, :, :, all_area, md100]
        )
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite metric found in COCO eval payload")
    return result


def evaluate_prediction_json(annotation: Path, predictions: Path) -> Mapping[str, Any]:
    coco_gt = COCO(str(require_file(annotation)))
    with require_file(predictions).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise RuntimeError(f"prediction JSON root is not a list: {predictions}")
    coco_dt = coco_gt.loadRes(records)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator.eval


def aggregate(per_seed: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    names = list(STANDARD_NAMES) + [f"AP{int(t * 100)}" for t in STRICT_THRESHOLDS]
    result: dict[str, Any] = {}
    for name in names:
        values = [float(per_seed[str(seed)][name]) * 100.0 for seed in SEEDS]
        result[name] = {
            "unit": "percentage points",
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
        }
    return result


def source_record(path: Path) -> dict[str, Any]:
    path = require_file(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    project = Path("/root/autodl-tmp/BTC_DEIM_project").resolve()
    adapter = project / "second_host"
    final_release = project / "HiLoR-DETR/main_host"
    sota = project / "benchmarks/02_RT_DETR_R18"
    annotation = require_file(
        project / "main_host/datasets/CDW1630/annotations/instances_test2017.json"
    )
    output_dir = adapter / "analysis/strict_iou_and_main_gccs"

    rtdetr_rows: dict[str, Any] = {}
    for method in ("native", "gccs", "full"):
        per_seed: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        for seed in SEEDS:
            if method == "native":
                path = require_file(
                    sota / f"results/test_reeval_v1/seed{seed}/predictions.json"
                )
                payload = evaluate_prediction_json(annotation, path)
            elif method == "gccs":
                path = require_file(adapter / f"evaluations_gccs/seed{seed}/coco_bbox_eval.pth")
                payload = load_torch_eval(path)
            else:
                path = require_file(adapter / f"evaluations/seed{seed}/coco_bbox_eval.pth")
                payload = load_torch_eval(path)
            per_seed[str(seed)] = metric_bundle(payload)
            sources[str(seed)] = source_record(path)
        rtdetr_rows[method] = {
            "per_seed_fraction": per_seed,
            "three_seed_percentage_points": aggregate(per_seed),
            "sources": sources,
        }

    main_gccs_per_seed: dict[str, Any] = {}
    main_gccs_sources: dict[str, Any] = {}
    for seed in SEEDS:
        path = require_file(
            final_release / f"results/CDW1630/Ours/seed{seed}/native_test/eval.pth"
        )
        main_gccs_per_seed[str(seed)] = metric_bundle(load_torch_eval(path))
        main_gccs_sources[str(seed)] = source_record(path)

    common = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operation": "read-only extraction from frozen predictions/eval tensors; no inference or training",
        "test_used_for_selection_fitting_or_tuning": False,
        "annotation": source_record(annotation),
        "seeds": list(SEEDS),
        "sample_sd_denominator": "n-1",
    }
    write_new_json(
        output_dir / "RTDETR_THREE_STAGE_ALL_AND_STRICT_IOU.json",
        {
            "schema": "cdw1630-rtdetr-three-stage-all-and-strict-iou-v1",
            **common,
            "methods": rtdetr_rows,
        },
    )
    write_new_json(
        output_dir / "MAIN_HOST_GCCS_THREE_SEED_ALL_METRICS.json",
        {
            "schema": "cdw1630-main-host-gccs-three-seed-all-metrics-v1",
            **common,
            "method": "DEIM/D-FINE-M + GCCS native detector output",
            "per_seed_fraction": main_gccs_per_seed,
            "three_seed_percentage_points": aggregate(main_gccs_per_seed),
            "sources": main_gccs_sources,
        },
    )
    print("EXISTING EVAL METRIC EXTRACTION: PASS")
    print(f"Output: {output_dir}")
    print("GPU used: NO")
    print("Inference rerun: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
