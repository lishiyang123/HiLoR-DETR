#!/usr/bin/env python
"""Build frozen multi-IoU QCCR calibration targets from a Val capture.

The target for candidate (query i, class c) is the fraction of COCO IoU
thresholds at which it is a TP under the original frozen PD ranking:

    y_ic = mean_tau TP_ic(tau), tau = 0.50, ..., 0.95

Only the original PostProcessor Top-300 candidates participate in target
construction and later parameter fitting. All 300 x C candidates retain their
logit and competition risk so inference can still recalibrate before Top-300.

This script never modifies model source, checkpoints, captures, or outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.deim.competition_mapping import compute_competition_risk


IOU_THRESHOLDS = tuple(
    round(0.50 + 0.05 * index, 2)
    for index in range(10)
)


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


def pairwise_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    )

    left_top = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )
    right_bottom = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    size = (right_bottom - left_top).clamp_min(0)
    intersection = size[..., 0] * size[..., 1]

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp_min(1.0e-9)


def build_record_targets(record, num_classes, num_top_queries):
    logits = record["pred_logits"].float()
    boxes_cxcywh = record["pred_boxes"].float()

    num_queries = logits.shape[0]

    if logits.shape != (num_queries, num_classes):
        raise RuntimeError(
            f"unexpected logits shape: {tuple(logits.shape)}"
        )

    if boxes_cxcywh.shape != (num_queries, 4):
        raise RuntimeError(
            f"unexpected boxes shape: {tuple(boxes_cxcywh.shape)}"
        )

    risk = compute_competition_risk(
        logits.unsqueeze(0),
        boxes_cxcywh.unsqueeze(0),
    )[0].float()

    height, width = [
        int(value)
        for value in record["orig_size"].tolist()
    ]

    scale = torch.tensor(
        [width, height, width, height],
        dtype=torch.float32,
    )

    predicted_boxes = (
        cxcywh_to_xyxy(boxes_cxcywh) * scale
    )

    target_boxes = record["target_boxes"].float()
    target_labels = record["target_labels"].long()
    target_iscrowd = record["target_iscrowd"].long()

    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise RuntimeError(
            f"unexpected target box shape: {tuple(target_boxes.shape)}"
        )

    top_queries = record["post_query_idx"].long()
    top_labels = record["post_labels"].long()

    if top_queries.numel() != num_top_queries:
        raise RuntimeError(
            f"expected {num_top_queries} post candidates, "
            f"got {top_queries.numel()}"
        )

    top_mask = torch.zeros(
        num_queries,
        num_classes,
        dtype=torch.bool,
    )
    top_mask[top_queries, top_labels] = True

    if int(top_mask.sum()) != num_top_queries:
        raise RuntimeError(
            "duplicate (query,class) entry in original Top-300"
        )

    targets = torch.zeros(
        num_queries,
        num_classes,
        dtype=torch.float32,
    )

    # post_* is already sorted by the frozen original score.
    # COCO maxDets=100 is enforced per image-category.
    for class_index in range(num_classes):
        valid_gt = torch.nonzero(
            (target_labels == class_index)
            & (target_iscrowd == 0),
            as_tuple=False,
        ).flatten()

        if valid_gt.numel() == 0:
            continue

        class_positions = torch.nonzero(
            top_labels == class_index,
            as_tuple=False,
        ).flatten()

        if class_positions.numel() == 0:
            continue

        class_positions = class_positions[:100]
        class_queries = top_queries[class_positions]

        class_iou = pairwise_iou(
            predicted_boxes[class_queries],
            target_boxes[valid_gt],
        )

        for threshold in IOU_THRESHOLDS:
            matched = torch.zeros(
                valid_gt.numel(),
                dtype=torch.bool,
            )

            for local_query_index, query_index in enumerate(
                class_queries.tolist()
            ):
                available = ~matched

                if not bool(available.any()):
                    break

                candidate_iou = class_iou[
                    local_query_index
                ].clone()
                candidate_iou[~available] = -1.0

                best_iou, best_gt = candidate_iou.max(dim=0)

                if float(best_iou) >= threshold:
                    targets[
                        query_index,
                        class_index,
                    ] += 1.0 / len(IOU_THRESHOLDS)
                    matched[int(best_gt)] = True

    return {
        "logits": logits.reshape(-1),
        "competition_risk": risk.reshape(-1),
        "targets": targets.reshape(-1),
        "fit_mask": top_mask.reshape(-1),
        "query_index": torch.arange(
            num_queries,
            dtype=torch.int64,
        ).repeat_interleave(num_classes),
        "class_index": torch.arange(
            num_classes,
            dtype=torch.int64,
        ).repeat(num_queries),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    capture_path = Path(args.capture).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()

    capture = torch.load(
        capture_path,
        map_location="cpu",
        weights_only=False,
    )

    if capture.get("split") != "val":
        raise RuntimeError(
            "QCCR fitting targets must be built from Val, "
            f"got split={capture.get('split')!r}"
        )

    if capture.get("method") not in {"PD", "frozen_native_detector"}:
        raise RuntimeError(
            "expected a frozen native detector capture, "
            f"got method={capture.get('method')!r}"
        )

    if not bool(capture.get("use_focal_loss")):
        raise RuntimeError("QCCR-v1 requires focal/sigmoid logits")

    num_classes = int(
        capture["postprocessor_num_classes"]
    )
    num_top_queries = int(
        capture["postprocessor_num_top_queries"]
    )

    if num_classes <= 0 or num_top_queries != 300:
        raise RuntimeError(
            "invalid capture geometry: "
            f"num_classes={num_classes}, num_top_queries={num_top_queries}"
        )

    parts = {
        "logits": [],
        "competition_risk": [],
        "targets": [],
        "fit_mask": [],
        "image_id": [],
        "query_index": [],
        "class_index": [],
    }

    for record in capture["records"]:
        built = build_record_targets(
            record,
            num_classes=num_classes,
            num_top_queries=num_top_queries,
        )

        count = built["logits"].numel()

        for key in (
            "logits",
            "competition_risk",
            "targets",
            "fit_mask",
            "query_index",
            "class_index",
        ):
            parts[key].append(built[key])

        parts["image_id"].append(
            torch.full(
                (count,),
                int(record["image_id"]),
                dtype=torch.int64,
            )
        )

    tensors = {
        key: torch.cat(value, dim=0)
        for key, value in parts.items()
    }

    expected_total = (
        len(capture["records"])
        * capture["summary"]["query_counts"][0]
        * num_classes
    )

    if tensors["logits"].numel() != expected_total:
        raise RuntimeError(
            "candidate count mismatch: "
            f"{tensors['logits'].numel()} vs {expected_total}"
        )

    fit_mask = tensors["fit_mask"]
    fit_targets = tensors["targets"][fit_mask]
    fit_risk = tensors["competition_risk"][fit_mask]

    if int(fit_mask.sum()) != (
        len(capture["records"]) * num_top_queries
    ):
        raise RuntimeError(
            "Top-300 fit-mask cardinality mismatch"
        )

    target_min_before_clamp = float(
        tensors["targets"].min()
    )
    target_max_before_clamp = float(
        tensors["targets"].max()
    )

    tolerance = 1.0e-6

    if (
        target_min_before_clamp < -tolerance
        or target_max_before_clamp > 1.0 + tolerance
    ):
        raise RuntimeError(
            "targets exceed numerical tolerance: "
            f"min={target_min_before_clamp}, "
            f"max={target_max_before_clamp}"
        )

    # Ten float32 additions of 0.1 may produce 1.000000119.
    # Clamp only after validating that the excess is numerical.
    tensors["targets"] = tensors["targets"].clamp(
        min=0.0,
        max=1.0,
    )
    fit_targets = tensors["targets"][fit_mask]

    payload = {
        "format_version": 1,
        "method": "QCCR-v1",
        "source_capture": str(capture_path),
        "source_checkpoint": capture["checkpoint"],
        "source_checkpoint_sha256": capture.get("checkpoint_sha256"),
        "source_split": capture["split"],
        "source_method": capture["method"],
        "iou_thresholds": list(IOU_THRESHOLDS),
        "num_classes": num_classes,
        "num_queries": int(
            capture["summary"]["query_counts"][0]
        ),
        "num_top_queries": num_top_queries,
        "tensors": tensors,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(payload, output_path)

    summary = {
        "image_count": len(capture["records"]),
        "gt_count": int(capture["summary"]["gt_count"]),
        "all_candidate_count": int(
            tensors["logits"].numel()
        ),
        "fit_candidate_count": int(fit_mask.sum()),
        "fit_target_mass": float(fit_targets.sum()),
        "fit_nonzero_target_count": int(
            (fit_targets > 0).sum()
        ),
        "fit_nonzero_target_ratio": float(
            (fit_targets > 0).float().mean()
        ),
        "fit_full_target_count": int(
            (fit_targets == 1).sum()
        ),
        "fit_risk_mean": float(fit_risk.mean()),
        "fit_risk_positive_ratio": float(
            (fit_risk > 0).float().mean()
        ),
        "fit_risk_p50": float(
            torch.quantile(fit_risk, 0.50)
        ),
        "fit_risk_p90": float(
            torch.quantile(fit_risk, 0.90)
        ),
        "target_path": str(output_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("QCCR VAL TARGET BUILD: PASS")


if __name__ == "__main__":
    main()
