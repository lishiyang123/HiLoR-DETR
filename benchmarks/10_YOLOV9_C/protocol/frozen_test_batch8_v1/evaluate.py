#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.metadata, io
from pathlib import Path
from typing import Any
from common import atomic_json, read_json, require, sha256

METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")

def mean_valid(value: Any) -> float | None:
    import numpy as np
    selected = value[value > -1]
    return float(np.mean(selected)) if selected.size else None

def tensor_sha256(value: Any) -> str:
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(tuple(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()

def compute(annotation: Path, predictions_path: Path) -> dict[str, Any]:
    import numpy as np
    from faster_coco_eval import COCO, COCOeval_faster
    version = importlib.metadata.version("faster-coco-eval")
    require(version == "1.7.2", f"Unexpected faster-coco-eval: {version}")
    predictions = read_json(predictions_path)
    require(isinstance(predictions, list) and predictions, "Predictions must be a nonempty list")
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(annotation)); detections = truth.loadRes(predictions)
        evaluator = COCOeval_faster(truth, iouType="bbox", print_function=lambda *_a, **_k: None, separate_eval=True)
        evaluator.cocoDt = detections; evaluator.params.imgIds = sorted(truth.getImgIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    values = [float(value) for value in evaluator.stats]
    require(len(values) == 12, "Unexpected metric vector")
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    iou_values = [float(value) for value in evaluator.params.iouThrs]
    high_quality = {}
    for threshold in (0.80, 0.85, 0.90, 0.95):
        index = min(range(len(iou_values)), key=lambda item: abs(iou_values[item] - threshold))
        require(abs(iou_values[index] - threshold) < 1e-9, f"Missing IoU threshold: {threshold}")
        high_quality[f"AP{int(round(100 * threshold))}"] = mean_valid(precision[index, :, :, 0, 2])
    categories = {int(row["id"]): str(row["name"]) for row in truth.dataset["categories"]}
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        per_class[categories[int(category_id)]] = mean_valid(precision[:, :, index, 0, 2])
    return {"schema": "cdw1630-yolov9c-batch8-faster-coco-result-v1", "status": "PASS",
            "implementation": "faster_coco_eval.COCOeval_faster", "faster_coco_eval": version,
            "iou_type": "bbox", "maxDets": [1, 10, 100], "annotation_sha256": sha256(annotation),
            "predictions_sha256": sha256(predictions_path), "images": len(evaluator.params.imgIds),
            "detections": len(predictions), "metrics": dict(zip(METRICS, values)),
            "high_quality": high_quality, "per_class_AP": per_class,
            "precision_sha256": tensor_sha256(precision), "recall_sha256": tensor_sha256(recall)}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    observed = compute(args.annotation.resolve(strict=True), args.predictions.resolve(strict=True))
    if args.verify:
        require(read_json(args.output.resolve(strict=True)) == observed, "Exact metric replay differs")
        print("YOLOv9-C FASTER-COCO-EVAL 1.7.2 REPLAY: PASS")
    else:
        require(not args.output.exists(), f"Refusing existing metric: {args.output}")
        atomic_json(args.output, observed)
        print(" ".join(f"{key}={observed['metrics'][key]:.9f}" for key in METRICS))

if __name__ == "__main__": main()
