#!/usr/bin/env python3
"""Frozen official-checkpoint YOLO Test inference with a verified directory-backed batch-8 loader."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from common import atomic_json, coco_row, digest, normalize_names, require, sha256


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    methods = {row["key"]: row for row in protocol["models"]}
    require(args.model_key in methods, f"Unknown model key: {args.model_key}")
    method = methods[args.model_key]
    inference = protocol["inference"]
    weight = args.weight.resolve(strict=True)
    annotation_path = args.annotation.resolve(strict=True)
    images = args.images.resolve(strict=True)
    require(images.is_dir(), "Image source must be a directory")
    require(digest(weight, "md5") == method["weight_md5"], "Official checkpoint MD5 changed")
    require(inference["source_type"] == "directory_path_string" and int(inference["batch"]) == 8,
            "Frozen directory/batch contract changed")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(annotation["categories"], key=lambda row: int(row["id"]))
    expected_names = [str(row["name"]) for row in categories]
    require([int(row["id"]) for row in categories] == protocol["dataset"]["category_ids"],
            "Test category ids changed")
    annotation_images = {Path(str(row["file_name"])).name: row for row in annotation["images"]}
    paths = sorted((path for path in images.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=lambda path: path.name)
    require(len(paths) == protocol["dataset"]["images"] == len(annotation_images),
            "Test image count changed")
    require(len(annotation["annotations"]) == protocol["dataset"]["annotations"],
            "Test annotation count changed")
    require(all(path.name in annotation_images for path in paths), "Image/annotation mapping changed")

    seed = int(inference["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "CUDA is required for formal Test inference")

    model = YOLO(str(weight), task="detect")
    names = normalize_names(model.names)
    aliases = protocol["class_mapping"]["aliases"]
    require([int(row["id"]) for row in aliases] == list(range(len(expected_names))),
            "Frozen checkpoint class ids are not contiguous")
    require([str(row["canonical"]) for row in aliases] == expected_names,
            "Frozen checkpoint aliases do not match canonical Test categories")
    require(len(names) == len(aliases), f"Checkpoint class count changed: {names!r}")
    for index, name in enumerate(names):
        require(name in aliases[index]["allowed_checkpoint_names"],
                f"Checkpoint class mapping changed at id={index}: {name!r} not in "
                f"{aliases[index]['allowed_checkpoint_names']!r}")

    rows: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    # A directory string is required. A Python list can select an in-memory loader and ignore batch.
    results = model.predict(
        source=str(images), stream=True, imgsz=int(inference["imgsz"]), batch=int(inference["batch"]),
        conf=float(inference["conf"]), iou=float(inference["iou"]), max_det=int(inference["max_det"]),
        device=int(inference["device"]), half=bool(inference["half"]), augment=bool(inference["augment"]),
        agnostic_nms=bool(inference["agnostic_nms"]), rect=bool(inference["rect"]),
        verbose=False, save=False, plots=False,
    )
    observed_names: set[str] = set()
    outside_canvas = 0
    for result in results:
        name = Path(result.path).name
        require(name not in observed_names and name in annotation_images,
                f"Unexpected or duplicate image result: {name}")
        observed_names.add(name)
        image = annotation_images[name]
        count = 0
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            require(len(xyxy) == len(scores) == len(classes), "Detection tensor lengths differ")
            for coordinates, score, category in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                outside_canvas += int(x1 < 0.0 or y1 < 0.0 or x2 > float(image["width"]) or
                                      y2 > float(image["height"]))
                rows.append(coco_row(int(image["id"]), int(category), coordinates, float(score), len(names)))
                count += 1
        seen.append({"image_id": int(image["id"]), "file_name": name,
                     "width": int(image["width"]), "height": int(image["height"]), "detections": count})
    require(observed_names == {path.name for path in paths}, "Inference image coverage mismatch")
    require(rows, "Inference produced no detections")

    predictor = getattr(model, "predictor", None)
    dataset = getattr(predictor, "dataset", None)
    require(dataset is not None, "Predictor dataset unavailable for batch verification")
    loader_class = type(dataset).__name__
    effective_batch = int(getattr(dataset, "bs", -1))
    require(loader_class in protocol["inference"]["allowed_loader_classes"] and
            loader_class != "LoadPilAndNumpy", f"Unexpected inference loader: {loader_class}")
    require(effective_batch == int(inference["batch"]) == 8,
            f"Requested/effective batch mismatch: {inference['batch']} != {effective_batch}")

    seen.sort(key=lambda row: row["image_id"])
    rows.sort(key=lambda row: (row["image_id"], -row["score"], row["category_id"], row["bbox"]))
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "predictions.coco.json", rows)
    atomic_json(args.output / "images_seen.json", seen)
    atomic_json(args.output / "runtime.json", {
        "schema": "cdw1630-official-yolo-batch8-inference-v1", "status": "PASS", "mode": "FORMAL",
        "model_key": method["key"], "model": method["name"], "scope": "official_single_checkpoint",
        "weight": str(weight), "weight_md5": digest(weight, "md5"), "weight_sha256": sha256(weight),
        "annotation": str(annotation_path), "annotation_sha256": sha256(annotation_path),
        "images_directory": str(images), "images": len(seen), "detections": len(rows),
        "outside_canvas": outside_canvas, "classes": names,
        "canonical_classes": expected_names,
        "class_mapping_policy": protocol["class_mapping"]["policy"],
        "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "inference": inference,
        "loader": {"source_argument_type": "str", "source_is_directory": True,
                   "class": loader_class, "requested_batch": int(inference["batch"]),
                   "effective_batch": effective_batch},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "ultralytics": getattr(ultralytics, "__version__", "UNKNOWN"),
                    "gpu": torch.cuda.get_device_name(int(inference["device"]))},
        "training_executed": False, "test_selection_or_fitting": False,
    })
    print(f"{method['name']} FORMAL: images={len(seen)} detections={len(rows)} "
          f"loader={loader_class} effective_batch={effective_batch}")


if __name__ == "__main__":
    main()
