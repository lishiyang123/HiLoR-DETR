#!/usr/bin/env python3
"""Capture official D-FINE predictions without changing its source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def convert_result(image_id: int, result: dict, width: int, height: int) -> tuple[list[dict], int]:
    boxes = result["boxes"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    if not (len(boxes) == len(labels) == len(scores)):
        fail("Official postprocessor returned unequal boxes/labels/scores")
    rows = []
    outside_count = 0
    for box, label, score in zip(boxes, labels, scores):
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            fail("Non-finite or malformed D-FINE box")
        x1, y1, x2, y2 = (float(x) for x in box)
        label = int(label)
        score = float(score)
        if label not in range(5) or not math.isfinite(score) or not 0 <= score <= 1:
            fail("Invalid D-FINE label or score")
        if x2 <= x1 or y2 <= y1:
            fail("Nonpositive D-FINE box")
        # The pinned official D-FINE evaluation path passes native
        # postprocessor boxes directly to COCO evaluation without clipping.
        # Preserve those semantics; only count coordinates outside the canvas.
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            outside_count += 1
        rows.append({"image_id": image_id, "category_id": label,
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return rows, outside_count


def run(args: argparse.Namespace) -> None:
    import torch

    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(source))
    from src.core import YAMLConfig  # noqa: E402

    if args.predictions.exists() or args.images_seen.exists():
        fail("Prediction output already exists; refusing overwrite")
    if not torch.cuda.is_available():
        fail("CUDA GPU required for frozen D-FINE-M inference")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cfg = YAMLConfig(str(args.config.resolve(strict=True)))
    if cfg.yaml_cfg.get("remap_mscoco_category") is not False or cfg.yaml_cfg.get("num_classes") != 5:
        fail("Formal D-FINE class mapping is not 0..4")
    val_cfg = cfg.yaml_cfg["val_dataloader"]
    dataset_cfg = val_cfg["dataset"]
    if Path(dataset_cfg["ann_file"]).resolve() != args.annotation.resolve(strict=True):
        fail("Loaded config annotation does not match requested annotation")
    if Path(dataset_cfg["img_folder"]).resolve() != args.images.resolve(strict=True):
        fail("Loaded config image directory does not match requested images")

    device = torch.device("cuda:0")
    model = cfg.model.to(device)
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        fail("Selected checkpoint is not a dictionary")
    ema = checkpoint.get("ema")
    weights = ema.get("module") if isinstance(ema, dict) else None
    weight_source = "ema.module" if isinstance(weights, dict) else "model"
    if not isinstance(weights, dict):
        weights = checkpoint.get("model")
    if not isinstance(weights, dict):
        fail("Selected checkpoint lacks EMA and model weights")
    model.load_state_dict(weights, strict=True)
    model.eval()
    postprocessor = cfg.postprocessor.to(device)
    postprocessor.eval()
    loader = cfg.val_dataloader

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    image_by_id = {int(row["id"]): row for row in annotation["images"]}
    if len(image_by_id) != len(annotation["images"]):
        fail("Duplicate image IDs in annotation")
    expected_count = min(len(image_by_id), args.limit) if args.limit else len(image_by_id)
    rows: list[dict] = []
    seen: list[int] = []
    outside_count = 0
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            outputs = model(samples)
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postprocessor(outputs, sizes)
            if len(results) != len(targets):
                fail("Official postprocessor batch length mismatch")
            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())
                if image_id not in image_by_id or image_id in seen:
                    fail("Unexpected or duplicate predicted image ID")
                meta = image_by_id[image_id]
                converted, outside = convert_result(
                    image_id, result, int(meta["width"]), int(meta["height"])
                )
                rows.extend(converted)
                outside_count += outside
                seen.append(image_id)
                if len(seen) == expected_count:
                    break
            if len(seen) == expected_count:
                break
    if len(seen) != expected_count:
        fail(f"Incomplete inference: {len(seen)}/{expected_count} images")
    if not args.limit and set(seen) != set(image_by_id):
        fail("Formal inference did not cover exactly all annotated images")
    if not rows:
        fail("No detections produced")
    save_new(args.predictions, rows)
    save_new(args.images_seen, {"image_ids": seen, "image_count": len(seen),
                                "detections": len(rows), "checkpoint_sha256": sha256(args.checkpoint),
                                "config_sha256": sha256(args.config), "annotation_sha256": sha256(args.annotation),
                                "weight_source": weight_source, "parameters": sum(p.numel() for p in model.parameters()),
                                "precision": "FP32", "test_time_augmentation": False,
                                "box_handling": "official postprocessor coordinates preserved without clipping",
                                "outside_canvas_detections": outside_count,
                                "predictions_sha256": sha256(args.predictions)})
    print(f"D-FINE-M {args.tag}: images={len(seen)} detections={len(rows)} "
          f"outside_canvas={outside_count} parameters={sum(p.numel() for p in model.parameters())} "
          f"weights={weight_source}")
    print(f"Predictions: {args.predictions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images-seen", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
