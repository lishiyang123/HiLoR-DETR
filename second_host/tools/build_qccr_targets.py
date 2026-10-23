#!/usr/bin/env python3
"""Build frozen multi-IoU QCCR targets from the bound Val capture only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from fit_common import (
        EXPECTED_CLASSES, EXPECTED_QUERIES, EXPECTED_TOPK, atomic_json,
        atomic_torch_save, cxcywh_to_xyxy, install_source_root, load_capture,
        pairwise_iou, require_output_path, runtime_source_hashes, sha256_file,
        formal_fit_guard,
    )
except ImportError:  # imported as tools.build_qccr_targets in unit tests
    from tools.fit_common import (
    EXPECTED_CLASSES,
    EXPECTED_QUERIES,
    EXPECTED_TOPK,
    atomic_json,
    atomic_torch_save,
    cxcywh_to_xyxy,
    install_source_root,
    load_capture,
    pairwise_iou,
    require_output_path,
    runtime_source_hashes,
    sha256_file,
    formal_fit_guard,
    )


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
TARGET_SCHEMA = "rtdetr-hilor-qccr-multiiou-targets-v1"


def build_record_targets(record, compute_competition_risk):
    logits = record["pred_logits"].float()
    boxes_cxcywh = record["pred_boxes"].float()
    risk = compute_competition_risk(
        logits.unsqueeze(0), boxes_cxcywh.unsqueeze(0)
    )[0].float()
    height, width = [int(value) for value in record["orig_size"].tolist()]
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    predicted_boxes = cxcywh_to_xyxy(boxes_cxcywh) * scale
    target_boxes = record["target_boxes"].float()
    target_labels = record["target_labels"].long()
    target_iscrowd = record["target_iscrowd"].long()

    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise RuntimeError("invalid target box geometry")
    if target_labels.numel() and (
        int(target_labels.min()) < 0 or int(target_labels.max()) >= EXPECTED_CLASSES
    ):
        raise RuntimeError("target category lies outside contiguous IDs 0..4")

    top_queries = record["post_query_idx"].long()
    top_labels = record["post_labels"].long()
    top_mask = torch.zeros(EXPECTED_QUERIES, EXPECTED_CLASSES, dtype=torch.bool)
    top_mask[top_queries, top_labels] = True
    if int(top_mask.sum()) != EXPECTED_TOPK:
        raise RuntimeError("native Top-K contains duplicate query-class candidates")

    targets = torch.zeros(EXPECTED_QUERIES, EXPECTED_CLASSES, dtype=torch.float32)
    # Reproduce COCO's score-order matching and maxDets=100 per category.
    for class_index in range(EXPECTED_CLASSES):
        valid_gt = torch.nonzero(
            (target_labels == class_index) & (target_iscrowd == 0), as_tuple=False
        ).flatten()
        class_positions = torch.nonzero(top_labels == class_index, as_tuple=False).flatten()
        if valid_gt.numel() == 0 or class_positions.numel() == 0:
            continue
        class_queries = top_queries[class_positions[:100]]
        class_iou = pairwise_iou(predicted_boxes[class_queries], target_boxes[valid_gt])
        for threshold in IOU_THRESHOLDS:
            matched = torch.zeros(valid_gt.numel(), dtype=torch.bool)
            for local_index, query_index in enumerate(class_queries.tolist()):
                if not bool((~matched).any()):
                    break
                candidate_iou = class_iou[local_index].clone()
                candidate_iou[matched] = -1.0
                best_iou, best_gt = candidate_iou.max(dim=0)
                if float(best_iou) >= threshold:
                    targets[query_index, class_index] += 1.0 / len(IOU_THRESHOLDS)
                    matched[int(best_gt)] = True

    return {
        "logits": logits.reshape(-1),
        "competition_risk": risk.reshape(-1),
        "targets": targets.reshape(-1),
        "fit_mask": top_mask.reshape(-1),
        "query_index": torch.arange(EXPECTED_QUERIES).repeat_interleave(
            EXPECTED_CLASSES
        ),
        "class_index": torch.arange(EXPECTED_CLASSES).repeat(EXPECTED_QUERIES),
    }


def build_payload(
    capture, capture_path, capture_sha, source_root, *,
    implementation_lock_sha256, detector_seed=None,
):
    if (
        not isinstance(implementation_lock_sha256, str)
        or len(implementation_lock_sha256) != 64
        or any(character not in "0123456789abcdef" for character in implementation_lock_sha256)
    ):
        raise RuntimeError("QCCR targets require a valid implementation-lock SHA-256")
    runtime_hashes = runtime_source_hashes(source_root, require_umr=False)
    _, compute_competition_risk, _, _ = install_source_root(
        source_root, require_umr=False
    )
    parts = {
        key: []
        for key in (
            "logits", "competition_risk", "targets", "fit_mask", "image_id",
            "query_index", "class_index",
        )
    }
    for record in capture["records"]:
        built = build_record_targets(record, compute_competition_risk)
        count = built["logits"].numel()
        for key, value in built.items():
            parts[key].append(value)
        parts["image_id"].append(
            torch.full((count,), int(record["image_id"]), dtype=torch.int64)
        )
    tensors = {key: torch.cat(values, dim=0) for key, values in parts.items()}
    expected = len(capture["records"]) * EXPECTED_QUERIES * EXPECTED_CLASSES
    if tensors["logits"].numel() != expected:
        raise RuntimeError("QCCR target candidate cardinality mismatch")
    fit_mask = tensors["fit_mask"].bool()
    if int(fit_mask.sum()) != len(capture["records"]) * EXPECTED_TOPK:
        raise RuntimeError("QCCR target fit-mask cardinality mismatch")
    target_min = float(tensors["targets"].min())
    target_max = float(tensors["targets"].max())
    if target_min < -1.0e-6 or target_max > 1.0 + 1.0e-6:
        raise RuntimeError("QCCR targets lie outside [0,1]")
    tensors["targets"] = tensors["targets"].clamp(0.0, 1.0)
    payload = {
        "schema": TARGET_SCHEMA,
        "format_version": 1,
        "method": "QCCR-v1",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": detector_seed,
        "source_capture": str(capture_path),
        "source_capture_sha256": capture_sha,
        "source_checkpoint": capture["checkpoint"],
        "source_checkpoint_sha256": capture["checkpoint_sha256"],
        "source_data_split_lock_sha256": capture["data_split_lock_sha256"],
        "runtime_source_sha256": runtime_hashes,
        "source_split": "val",
        "test_accessed": False,
        "iou_thresholds": list(IOU_THRESHOLDS),
        "num_classes": EXPECTED_CLASSES,
        "num_queries": EXPECTED_QUERIES,
        "num_top_queries": EXPECTED_TOPK,
        "tensors": tensors,
    }
    payload["implementation_lock_sha256"] = implementation_lock_sha256
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    formal_evidence = formal_fit_guard(
        tool_relative="tools/build_qccr_targets.py",
        adapter_root_path=args.adapter_root,
        source_root=args.source_root,
        check_runtime=False,
    )

    capture_path = args.val_capture.resolve()
    capture = load_capture(
        capture_path,
        expected_sha=args.expected_val_capture_sha,
        split="val",
        expected_checkpoint_sha=args.expected_checkpoint_sha,
        adapter_root_path=args.adapter_root,
        detector_seed=args.detector_seed,
    )
    payload = build_payload(
        capture,
        capture_path,
        args.expected_val_capture_sha,
        args.source_root.resolve(),
        detector_seed=args.detector_seed,
        implementation_lock_sha256=formal_evidence[
            "implementation_lock_sha256"
        ],
    )
    fit_mask = payload["tensors"]["fit_mask"].bool()
    fit_targets = payload["tensors"]["targets"][fit_mask]
    fit_risk = payload["tensors"]["competition_risk"][fit_mask]
    summary = {
        "schema": "rtdetr-hilor-qccr-target-summary-v1",
        "test_accessed": False,
        "source_capture_sha256": args.expected_val_capture_sha,
        "source_checkpoint_sha256": args.expected_checkpoint_sha,
        "image_count": len(capture["records"]),
        "gt_count": sum(int(record["target_labels"].numel()) for record in capture["records"]),
        "all_candidate_count": int(payload["tensors"]["logits"].numel()),
        "fit_candidate_count": int(fit_mask.sum()),
        "fit_target_mass": float(fit_targets.sum()),
        "fit_nonzero_target_count": int((fit_targets > 0).sum()),
        "fit_full_target_count": int((fit_targets == 1).sum()),
        "fit_risk_mean": float(fit_risk.mean()),
        "fit_risk_positive_ratio": float((fit_risk > 0).float().mean()),
    }
    output = require_output_path(
        args.output, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="QCCR target output",
    )
    summary_path = require_output_path(
        args.summary, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="QCCR target summary",
    )
    summary["source_data_split_lock_sha256"] = capture["data_split_lock_sha256"]
    summary["runtime_source_sha256"] = payload["runtime_source_sha256"]
    summary["detector_seed"] = args.detector_seed
    atomic_torch_save(output, payload)
    summary["target_path"] = str(output)
    summary["target_sha256"] = sha256_file(output)
    atomic_json(summary_path, summary)
    print("RT-DETR QCCR VAL TARGET BUILD: PASS")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
