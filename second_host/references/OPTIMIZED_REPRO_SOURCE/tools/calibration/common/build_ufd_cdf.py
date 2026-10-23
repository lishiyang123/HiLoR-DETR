#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def max_ioa(boxes: list[list[float]]) -> list[float]:
    result = []
    for i, (x, y, w, h) in enumerate(boxes):
        area = w * h
        maximum = 0.0
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j:
                continue
            iw = max(0.0, min(x + w, x2 + w2) - max(x, x2))
            ih = max(0.0, min(y + h, y2 + h2) - max(y, y2))
            maximum = max(maximum, (iw * ih) / area)
        result.append(maximum)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--base-size", type=float, default=640.0)
    args = parser.parse_args()
    train_path = args.annotation.resolve()
    output = args.output.resolve()
    if not train_path.is_file():
        raise RuntimeError(f"training annotation is missing: {train_path}")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite CDF: {output}")
    data = json.loads(train_path.read_text(encoding="utf-8"))
    if args.expected_images is not None and len(data["images"]) != args.expected_images:
        raise RuntimeError(
            "training image count mismatch: "
            f"expected={args.expected_images}, actual={len(data['images'])}"
        )
    image_size = {
        int(item["id"]): (float(item["width"]), float(item["height"]))
        for item in data["images"]
    }
    by_image = defaultdict(list)
    for item in data["annotations"]:
        by_image[int(item["image_id"])].append([float(v) for v in item["bbox"]])
    area_values = []
    aspect_values = []
    overlap_values = []
    for image_id in sorted(by_image):
        width, height = image_size[image_id]
        boxes = by_image[image_id]
        for _, _, box_width, box_height in boxes:
            area_values.append(
                (box_width * box_height)
                / (width * height)
                * (args.base_size ** 2)
            )
            aspect_values.append(
                abs(
                    math.log(
                        (box_width / width)
                        / (box_height / height)
                    )
                )
            )
        overlap_values.extend(max_ioa(boxes))
    if not (len(area_values) == len(aspect_values) == len(overlap_values)):
        raise RuntimeError("inconsistent Train CDF population")
    if len(area_values) != len(data["annotations"]):
        raise RuntimeError("Train CDF population does not match Train annotations")
    payload = {
        "protocol": {
            "name": "DU-FDR UFD fixed empirical difficulty CDF",
            "dataset": args.dataset_name,
            "base_size": args.base_size,
            "population": len(area_values),
            "definitions": {
                "area": "normalized w*h * 640^2",
                "aspect": "abs(log((w/W)/(h/H))) using Train image dimensions",
                "overlap": "per-object max IoA = intersection/current-object-area",
            },
            "usage": "built deterministically from the supplied Train annotation only",
        },
        "summary": {
            "area_640": summary(area_values),
            "aspect_difficulty": summary(aspect_values),
            "max_ioa": summary(overlap_values),
        },
        "area_640_sorted": sorted(area_values),
        "aspect_difficulty_sorted": sorted(aspect_values),
        "max_ioa_sorted": sorted(overlap_values),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = {
        "format_version": 1,
        "source_train_annotation": str(train_path),
        "source_sha256": sha256(train_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "population": len(area_values),
        "source_images": len(data["images"]),
        "val_used": False,
        "test_used": False,
        "summary": payload["summary"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("===== UFD TRAIN-ONLY CDF =====")
    print("Dataset      :", args.dataset_name)
    print("Source images:", len(data["images"]))
    print("Population   :", len(area_values))
    for name, values in payload["summary"].items():
        print(
            f"{name:22s}: min={values['min']:.6f} q25={values['q25']:.6f} "
            f"q50={values['q50']:.6f} q75={values['q75']:.6f} max={values['max']:.6f}"
        )
    print("CDF SHA256:", report["output_sha256"])
    print("Val/Test used: NO")
    print("UFD TRAIN-ONLY CDF BUILD: PASS")


if __name__ == "__main__":
    main()
