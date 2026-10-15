#!/usr/bin/env python3
"""Frozen YOLOv9-C Test inference using the official dual-head path."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from common import atomic_json, coco_row, normalize_names, require, sha256


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_checkpoint_metadata(torch: Any, checkpoint: Path) -> dict[str, Any]:
    try:
        row = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        row = torch.load(checkpoint, map_location="cpu")
    require(isinstance(row, dict), "Selected checkpoint is not an official YOLOv9 checkpoint")
    options = row.get("opt", {})
    if not isinstance(options, dict):
        options = vars(options)
    return {
        "epoch_zero_based": int(row.get("epoch", -1)),
        "train_seed": int(options.get("seed", -1)),
        "train_epochs": int(options.get("epochs", -1)),
        "train_batch_size": int(options.get("batch_size", -1)),
        "imgsz": int(options.get("imgsz", -1)),
        "resume": bool(options.get("resume", False)),
        "has_ema": row.get("ema") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--best-epoch-zero-based", type=int, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.source = args.source.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.annotation = args.annotation.resolve(strict=True)
    args.images = args.images.resolve(strict=True)
    require(args.images.is_dir(), "Frozen Test image path is not a directory")
    require(sha256(args.checkpoint) == args.checkpoint_sha256, "Selected checkpoint SHA256 changed")

    sys.path.insert(0, str(args.source))
    os.chdir(args.source)
    import cv2
    import numpy as np
    import torch
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inference = protocol["inference"]
    require(int(inference["batch"]) == 8 and inference["precision"] == "FP32" and not inference["half"],
            "Frozen FP32/batch-8 inference contract changed")
    require(args.source_commit == "5b10b5099005a45332ebe7669f8480d5fc791a65",
            "Official YOLOv9 source commit changed")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "CUDA is required for formal YOLOv9 Test inference")

    checkpoint_metadata = load_checkpoint_metadata(torch, args.checkpoint)
    expected_metadata = {
        "epoch_zero_based": args.best_epoch_zero_based,
        "train_seed": args.seed,
        "train_epochs": int(protocol["epochs"]),
        "train_batch_size": 8,
        "imgsz": 640,
        "resume": False,
        "has_ema": True,
    }
    require(checkpoint_metadata == expected_metadata,
            f"Selected-checkpoint metadata changed: {checkpoint_metadata}")

    device = select_device(str(inference["device"]), batch_size=int(inference["batch"]))
    model = attempt_load(str(args.checkpoint), device=device, inplace=True, fuse=True).float().eval()
    names = normalize_names(getattr(model, "names", None))
    require(names == protocol["class_names"], f"Checkpoint class mapping changed: {names!r}")
    stride = int(model.stride.max())
    require(int(inference["imgsz"]) % stride == 0, "Frozen image size is not stride-aligned")

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    declared: dict[str, dict[str, Any]] = {}
    for row in annotation["images"]:
        name = Path(str(row["file_name"])).name
        require(name not in declared, f"Duplicate Test basename in annotation: {name}")
        declared[name] = row
    paths = sorted((path for path in args.images.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=lambda path: path.name)
    require(len(paths) == int(protocol["test_image_count"]), "Frozen Test image count changed")
    require({path.name for path in paths} == set(declared), "Frozen Test image/annotation mapping changed")

    prepared: list[tuple[Path, Any, tuple[int, int], Any, Any]] = []
    for path in paths:
        original = cv2.imread(str(path))
        require(original is not None, f"Cannot read Test image: {path}")
        native_shape = original.shape[:2]
        image, ratio, pad = letterbox(
            original, new_shape=int(inference["imgsz"]), auto=bool(inference["letterbox_auto"]),
            scaleup=bool(inference["scaleup"]), stride=stride,
        )
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        prepared.append((path, image, native_shape, ratio, pad))

    predictions: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    observed_batches: list[int] = []
    batch = int(inference["batch"])
    with torch.inference_mode():
        for offset in range(0, len(prepared), batch):
            selected = prepared[offset:offset + batch]
            observed_batches.append(len(selected))
            images = torch.from_numpy(np.stack([row[1] for row in selected])).to(device, non_blocking=True)
            images = images.float().div_(255.0)
            # Exact official val_dual.py inference branch for a dual-head YOLOv9 detector.
            outputs = model(images, augment=False)[0][1]
            detections = non_max_suppression(
                outputs, conf_thres=float(inference["conf"]), iou_thres=float(inference["iou"]),
                labels=(), multi_label=bool(inference["multi_label"]),
                agnostic=bool(inference["agnostic_nms"]), max_det=int(inference["max_det"]),
            )
            require(len(detections) == len(selected), "YOLOv9 batch output length changed")
            for index, detection in enumerate(detections):
                path, image, native_shape, ratio, pad = selected[index]
                annotation_row = declared[path.name]
                count = 0
                if len(detection):
                    detection = detection.clone()
                    scale_boxes(images[index].shape[1:], detection[:, :4], native_shape,
                                ratio_pad=(ratio, pad))
                    for x1, y1, x2, y2, confidence, category in detection.detach().cpu().tolist():
                        predictions.append(coco_row(
                            int(annotation_row["id"]), int(category), [x1, y1, x2, y2],
                            float(confidence), len(names),
                        ))
                        count += 1
                seen.append({
                    "image_id": int(annotation_row["id"]), "file_name": path.name,
                    "width": int(annotation_row["width"]), "height": int(annotation_row["height"]),
                    "detections": count,
                })

    require(predictions, "YOLOv9 Test inference produced no detections")
    require(observed_batches[:-1] == [8] * (len(observed_batches) - 1), "Non-final batch is not eight")
    require(observed_batches[-1] == len(paths) % 8 == 5, "Final Test batch changed")
    require(len(seen) == len(paths) and len({row["image_id"] for row in seen}) == len(paths),
            "Frozen Test coverage changed")
    predictions.sort(key=lambda row: (row["image_id"], -row["score"], row["category_id"], row["bbox"]))
    seen.sort(key=lambda row: row["image_id"])

    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "predictions.coco.json", predictions)
    atomic_json(args.output / "images_seen.json", seen)
    atomic_json(args.output / "runtime.json", {
        "schema": "cdw1630-sota-yolov9-c-batch8-test-runtime-v1", "status": "PASS",
        "model": protocol["model"], "model_key": protocol["model_key"], "seed": args.seed,
        "source": str(args.source), "source_commit": args.source_commit,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "annotation": str(args.annotation), "annotation_sha256": sha256(args.annotation),
        "images_directory": str(args.images), "images": len(seen), "detections": len(predictions),
        "classes": names, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "native_inference": protocol["native_inference"], "inference": inference,
        "loader": {"class": "frozen-directory-batch-loader", "requested_batch": batch,
                   "effective_batch": max(observed_batches), "batch_sizes": observed_batches,
                   "workers": 0, "source_is_directory": True},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
    })
    print(f"YOLOv9-C seed{args.seed}: images={len(seen)} detections={len(predictions)} "
          f"effective_batch={max(observed_batches)} final_batch={observed_batches[-1]}")


if __name__ == "__main__":
    main()
