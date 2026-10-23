#!/usr/bin/env python3
"""Capture frozen detector outputs from Train and Val only.

The detector is loaded once, strictly, from the locked Ours checkpoint. The
script performs inference only and also saves the native baseline COCOeval
object for each split. Capture records use the schema consumed by the QCCR and
UMR calibration tools. No Test path is accepted by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        dim=-1,
    )


def load_annotation(path: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for item in payload["annotations"]:
        annotations.setdefault(int(item["image_id"]), []).append(item)
    return images, annotations


def ground_truth(
    image_id: int,
    images: dict[int, dict],
    annotations: dict[int, list[dict]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    image = images[image_id]
    width = int(image["width"])
    height = int(image["height"])
    boxes = []
    labels = []
    areas = []
    crowds = []
    for item in annotations.get(image_id, []):
        x, y, w, h = [float(value) for value in item["bbox"]]
        x1 = min(max(x, 0.0), float(width))
        y1 = min(max(y, 0.0), float(height))
        x2 = min(max(x + w, 0.0), float(width))
        y2 = min(max(y + h, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(int(item["category_id"]))
        areas.append(float(item.get("area", (x2 - x1) * (y2 - y1))))
        crowds.append(int(item.get("iscrowd", 0)))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(areas, dtype=torch.float32),
        torch.tensor(crowds, dtype=torch.int64),
        height,
        width,
    )


def manual_native_post(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    sizes_wh: torch.Tensor,
    num_classes: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes_xyxy = cxcywh_to_xyxy(boxes)
    boxes_xyxy = boxes_xyxy * sizes_wh.repeat(1, 2).unsqueeze(1)
    all_scores = torch.sigmoid(logits).flatten(1)
    scores, flat_indices = torch.topk(all_scores, k=topk, dim=-1)
    labels = flat_indices % num_classes
    query_indices = flat_indices // num_classes
    selected_boxes = boxes_xyxy.gather(
        1, query_indices.unsqueeze(-1).expand(-1, -1, 4)
    )
    return query_indices, labels, scores, selected_boxes


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in payload and isinstance(payload["ema"], dict):
        state = payload["ema"].get("module")
        if isinstance(state, dict):
            return state
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has neither ema.module nor model state")
    return state


def capture_split(
    *,
    config_path: Path,
    split: str,
    cfg_model,
    postprocessor,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_sha: str,
    output_dir: Path,
    capture_prefix: str,
) -> dict:
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    loader = cfg.val_dataloader
    evaluator = cfg.evaluator
    evaluator.cleanup()

    dataset_cfg = cfg.yaml_cfg["val_dataloader"]["dataset"]
    annotation_path = Path(dataset_cfg["ann_file"]).resolve()
    images, annotations = load_annotation(annotation_path)

    num_classes = int(postprocessor.num_classes)
    topk = int(postprocessor.num_top_queries)
    records = []
    query_counts = set()
    total_gt = 0
    max_score_error = 0.0
    max_box_error = 0.0

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(loader):
            samples = samples.to(device)
            outputs = cfg_model(samples)
            if "umr_decoder_features" not in outputs:
                raise RuntimeError("model did not expose the required final-layer hL")

            batch_gt = []
            sizes_wh = []
            for target in targets:
                image_id = int(target["image_id"].item())
                gt = ground_truth(image_id, images, annotations)
                batch_gt.append((image_id, *gt))
                sizes_wh.append((gt[5], gt[4]))

            sizes_wh_tensor = torch.tensor(
                sizes_wh, dtype=torch.float32, device=device
            )
            native_results = postprocessor(outputs, sizes_wh_tensor)
            query_idx, labels, scores, selected_boxes = manual_native_post(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                sizes_wh_tensor,
                num_classes,
                topk,
            )

            result_by_image = {}
            for local_index, gt in enumerate(batch_gt):
                image_id, gt_boxes, gt_labels, gt_area, gt_crowd, height, width = gt
                native = native_results[local_index]
                if not torch.equal(native["labels"], labels[local_index]):
                    raise RuntimeError(
                        f"{split} image {image_id}: native/manual labels differ"
                    )
                score_error = float(
                    (native["scores"] - scores[local_index]).abs().max().item()
                )
                box_error = float(
                    (native["boxes"] - selected_boxes[local_index]).abs().max().item()
                )
                max_score_error = max(max_score_error, score_error)
                max_box_error = max(max_box_error, box_error)
                if score_error > 1.0e-6 or box_error > 1.0e-6:
                    raise RuntimeError(
                        f"{split} image {image_id}: native replay mismatch "
                        f"score={score_error}, box={box_error}"
                    )

                logits_cpu = outputs["pred_logits"][local_index].detach().cpu().float()
                boxes_cpu = outputs["pred_boxes"][local_index].detach().cpu().float()
                hl_cpu = (
                    outputs["umr_decoder_features"][local_index]
                    .detach()
                    .cpu()
                    .float()
                )
                query_counts.add(int(logits_cpu.shape[0]))
                total_gt += int((gt_crowd == 0).sum().item())
                records.append(
                    {
                        "image_id": image_id,
                        "pred_logits": logits_cpu,
                        "pred_boxes": boxes_cpu,
                        "hL": hl_cpu,
                        # Reviewed audit tools expect [height, width].
                        "orig_size": torch.tensor([height, width], dtype=torch.int64),
                        "target_boxes": gt_boxes,
                        "target_labels": gt_labels,
                        "target_area": gt_area,
                        "target_iscrowd": gt_crowd,
                        "gt_boxes": gt_boxes,
                        "gt_labels": gt_labels,
                        "gt_area": gt_area,
                        "gt_iscrowd": gt_crowd,
                        "post_query_idx": query_idx[local_index].detach().cpu().long(),
                        "post_labels": labels[local_index].detach().cpu().long(),
                        "post_scores": scores[local_index].detach().cpu().float(),
                        "post_boxes": selected_boxes[local_index].detach().cpu().float(),
                    }
                )
                result_by_image[image_id] = {
                    key: value.detach().cpu()
                    for key, value in native.items()
                }

            evaluator.update(result_by_image)
            if batch_index % 20 == 0:
                print(f"capture {split}: batch {batch_index}/{len(loader)}")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    if len(query_counts) != 1:
        raise RuntimeError(f"non-constant query count: {sorted(query_counts)}")
    capture = {
        "format_version": 2,
        "method": "frozen_native_detector",
        "split": split,
        "use_focal_loss": bool(postprocessor.use_focal_loss),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "postprocessor_num_classes": num_classes,
        "postprocessor_num_top_queries": topk,
        "annotation": str(annotation_path),
        "records": records,
        "summary": {
            "image_count": len(records),
            "gt_count": total_gt,
            "query_counts": sorted(query_counts),
            "native_manual_score_max_error": max_score_error,
            "native_manual_box_max_error": max_box_error,
        },
    }
    capture_path = output_dir / f"{capture_prefix}_{split}_capture.pt"
    torch.save(capture, capture_path)
    eval_path = output_dir / f"{capture_prefix}_{split}_baseline_eval.pth"
    torch.save(evaluator.coco_eval["bbox"].eval, eval_path)
    print(f"capture saved: {capture_path}")
    print(f"baseline eval saved: {eval_path}")
    return capture["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--val-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--capture-prefix", required=True)
    args = parser.parse_args()

    branch_root = args.branch_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(branch_root))

    from engine.core import YAMLConfig

    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != args.expected_checkpoint_sha:
        raise RuntimeError(
            "Detector checkpoint SHA mismatch: "
            f"expected={args.expected_checkpoint_sha}, actual={checkpoint_sha}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the frozen detector capture")
    device = torch.device("cuda:0")

    cfg = YAMLConfig(str(args.train_config.resolve()))
    model = cfg.model.to(device)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint_path), strict=True)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"strict load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    if int(postprocessor.num_classes) != args.num_classes:
        raise RuntimeError(
            "capture class-count mismatch: "
            f"expected={args.num_classes}, actual={postprocessor.num_classes}"
        )
    if bool(postprocessor.qccr_enabled) or bool(postprocessor.umr_enabled):
        raise RuntimeError("capture must use native inference with QCCR/UMR disabled")

    summaries = {}
    for split, config_path in (
        ("train", args.train_config.resolve()),
        ("val", args.val_config.resolve()),
    ):
        summaries[split] = capture_split(
            config_path=config_path,
            split=split,
            cfg_model=model,
            postprocessor=postprocessor,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            output_dir=output_dir,
            capture_prefix=args.capture_prefix,
        )

    summary = {
        "checkpoint_sha256": checkpoint_sha,
        "strict_load_missing": missing,
        "strict_load_unexpected": unexpected,
        "splits": summaries,
    }
    (output_dir / "CAPTURE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("TRAIN/VAL FROZEN DETECTOR CAPTURE: PASS")


if __name__ == "__main__":
    main()
