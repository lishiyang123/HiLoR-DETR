#!/usr/bin/env python3
"""Select a QCCR regularizer using image-level OOF COCO metrics on Val."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

try:
    from fit_common import (
        EXPECTED_CLASSES, EXPECTED_TOPK, atomic_json, cxcywh_to_xyxy,
        exact_metrics, install_source_root, load_capture, require_sha, run_coco,
        require_output_path, runtime_source_hashes, validate_annotation,
        xyxy_to_xywh, formal_fit_guard,
    )
    from fit_qccr import FIT_SCHEMA
except ImportError:  # imported as tools.evaluate_qccr_oof_val in unit tests
    from tools.fit_common import (
    EXPECTED_CLASSES,
    EXPECTED_TOPK,
    atomic_json,
    cxcywh_to_xyxy,
    exact_metrics,
    install_source_root,
    load_capture,
    require_sha,
    run_coco,
    require_output_path,
    runtime_source_hashes,
    validate_annotation,
    xyxy_to_xywh,
    formal_fit_guard,
    )
    from tools.fit_qccr import FIT_SCHEMA


SELECTION_SCHEMA = "rtdetr-hilor-qccr-oof-val-selection-v1"
STANDARD_METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)


def baseline_detections(capture):
    detections = []
    for record in capture["records"]:
        for label, score, box in zip(
            record["post_labels"], record["post_scores"], record["post_boxes"]
        ):
            detections.append(
                {
                    "image_id": int(record["image_id"]),
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                }
            )
    return detections


def prepare_records(capture, compute_competition_risk):
    prepared = []
    for record in capture["records"]:
        logits = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()
        risk = compute_competition_risk(logits.unsqueeze(0), boxes.unsqueeze(0))[0]
        height, width = [int(value) for value in record["orig_size"].tolist()]
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        prepared.append(
            {
                "image_id": int(record["image_id"]),
                "logits": logits,
                "risk": risk,
                "boxes_absolute": cxcywh_to_xyxy(boxes) * scale,
            }
        )
    return prepared


def oof_detections(prepared, fit_result, candidate_key, apply_qccr):
    candidate = fit_result["candidates"][candidate_key]
    fold_parameters = {
        int(entry["fold"]): entry["parameters"] for entry in candidate["folds"]
    }
    fold_by_image = {
        int(image_id): int(fold)
        for image_id, fold in fit_result["fold_by_image"].items()
    }
    detections = []
    for record in prepared:
        image_id = record["image_id"]
        if image_id not in fold_by_image:
            raise RuntimeError("OOF fit does not cover every Val image")
        parameters = fold_parameters[fold_by_image[image_id]]
        calibrated = apply_qccr(
            record["logits"],
            record["risk"],
            parameters["beta0"],
            parameters["beta_positive"],
            parameters["beta_negative"],
            parameters["duplicate_penalty"],
        )
        scores, indices = torch.topk(
            torch.sigmoid(calibrated).flatten(), EXPECTED_TOPK, dim=0
        )
        labels = indices % EXPECTED_CLASSES
        queries = indices // EXPECTED_CLASSES
        for label, score, box in zip(labels, scores, record["boxes_absolute"][queries]):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                }
            )
    return detections


def decide(metrics_by_candidate, baseline, fit_result):
    decisions = {}
    for key, metrics in metrics_by_candidate.items():
        deltas = {
            name: metrics[name] - baseline[name]
            for name in ("AP", "AP50", "AP75", "APm", "APl", "AR100", "AP90", "AP95")
        }
        guards = {
            "ap_gain_at_least_0.1pp": deltas["AP"] >= 0.001,
            "ap75_drop_no_more_than_0.05pp": deltas["AP75"] >= -0.0005,
            "apm_not_lower": deltas["APm"] >= 0.0,
            "ar100_drop_no_more_than_0.1pp": deltas["AR100"] >= -0.001,
            "ap90_drop_no_more_than_0.05pp": deltas["AP90"] >= -0.0005,
        }
        decisions[key] = {
            "metrics": metrics,
            "deltas_vs_native": deltas,
            "guards": guards,
            "qualified": bool(all(guards.values())),
        }
    qualified = [key for key, item in decisions.items() if item["qualified"]]
    selected_key = None
    if qualified:
        selected_key = max(
            qualified,
            key=lambda key: (
                decisions[key]["metrics"]["AP"],
                decisions[key]["metrics"]["AP75"],
                decisions[key]["metrics"]["APm"],
            ),
        )
    if selected_key is None:
        return decisions, None, None, None, "OOF_VAL_NO_GO"
    candidate = fit_result["candidates"][selected_key]
    return (
        decisions,
        selected_key,
        candidate["lambda_reg"],
        candidate["full_val_fit"]["parameters"],
        "OOF_VAL_GO",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--expected-val-capture-sha", required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--expected-baseline-eval-sha", required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--expected-fits-sha", required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--val-annotation", type=Path, required=True)
    parser.add_argument("--expected-val-annotation-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    formal_evidence = formal_fit_guard(
        tool_relative="tools/evaluate_qccr_oof_val.py",
        adapter_root_path=args.adapter_root,
        source_root=args.source_root,
    )

    output_dir = require_output_path(
        args.output_dir, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="QCCR OOF selection directory",
    )
    if output_dir.exists():
        raise RuntimeError(f"refusing to reuse OOF selection directory: {output_dir}")
    capture_path = args.val_capture.resolve()
    capture = load_capture(
        capture_path,
        expected_sha=args.expected_val_capture_sha,
        split="val",
        expected_checkpoint_sha=args.checkpoint_sha,
        adapter_root_path=args.adapter_root,
        detector_seed=args.detector_seed,
    )
    annotation = args.val_annotation.resolve()
    validate_annotation(annotation, args.expected_val_annotation_sha, capture)
    baseline_path = args.baseline_eval.resolve()
    require_sha(baseline_path, args.expected_baseline_eval_sha, "native Val evaluation")
    fit_path = require_output_path(
        args.fits, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root, label="QCCR fit input",
    )
    fit_sha = require_sha(fit_path, args.expected_fits_sha, "QCCR fits")
    fit_result = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit_result.get("implementation_lock_sha256") != formal_evidence[
        "implementation_lock_sha256"
    ]:
        raise RuntimeError("QCCR fits belong to another implementation lock")
    if fit_result.get("schema") != FIT_SCHEMA or fit_result.get("test_accessed") is not False:
        raise RuntimeError("invalid QCCR fit artifact")
    if fit_result.get("source_capture_sha256") != args.expected_val_capture_sha:
        raise RuntimeError("QCCR fits belong to another Val capture")
    if fit_result.get("source_data_split_lock_sha256") != capture.get(
        "data_split_lock_sha256"
    ):
        raise RuntimeError("QCCR fits use another data-split lock")
    if fit_result.get("source_checkpoint_sha256") != args.checkpoint_sha:
        raise RuntimeError("QCCR fits belong to another detector")
    if fit_result.get("detector_seed") != args.detector_seed:
        raise RuntimeError("QCCR fits belong to another detector seed")
    if fit_result.get("fold_count") != 5:
        raise RuntimeError("formal QCCR selection requires five image-level folds")
    if fit_result.get("seed") != 0:
        raise RuntimeError("formal QCCR selection requires the locked fold seed 0")

    source_root = args.source_root.resolve()
    current_runtime_hashes = runtime_source_hashes(source_root, require_umr=False)
    if fit_result.get("runtime_source_sha256") != current_runtime_hashes:
        raise RuntimeError("QCCR fits were produced by another runtime source")
    apply_qccr, compute_competition_risk, _, _ = install_source_root(
        source_root, require_umr=False
    )
    image_ids = [int(record["image_id"]) for record in capture["records"]]
    frozen_baseline = exact_metrics(
        torch.load(baseline_path, map_location="cpu", weights_only=False)
    )
    replay_baseline = exact_metrics(
        run_coco(annotation, baseline_detections(capture), image_ids)
    )
    errors = {
        metric: abs(replay_baseline[metric] - frozen_baseline[metric])
        for metric in STANDARD_METRICS
    }
    maximum_error = max(errors.values())
    if maximum_error > 1.0e-6:
        raise RuntimeError(f"native Val COCO replay mismatch: {maximum_error}")

    prepared = prepare_records(capture, compute_competition_risk)
    metrics_by_candidate = {}
    for key in fit_result["candidates"]:
        metrics_by_candidate[key] = exact_metrics(
            run_coco(
                annotation,
                oof_detections(prepared, fit_result, key, apply_qccr),
                image_ids,
            )
        )
    decisions, selected_key, selected_lambda, selected_parameters, decision = decide(
        metrics_by_candidate, replay_baseline, fit_result
    )
    summary = {
        "schema": SELECTION_SCHEMA,
        "format_version": 1,
        "protocol": "image-level-five-fold-OOF-Val",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": args.detector_seed,
        "implementation_lock_sha256": formal_evidence[
            "implementation_lock_sha256"
        ],
        "test_accessed": False,
        "capture": str(capture_path),
        "capture_sha256": args.expected_val_capture_sha,
        "source_capture_sha256": args.expected_val_capture_sha,
        "source_data_split_lock_sha256": capture["data_split_lock_sha256"],
        "runtime_source_sha256": current_runtime_hashes,
        "fits": str(fit_path),
        "fits_sha256": fit_sha,
        "checkpoint_sha256": args.checkpoint_sha,
        "annotation": str(annotation),
        "annotation_sha256": args.expected_val_annotation_sha,
        "baseline_replay_max_error": maximum_error,
        "baseline_metrics": replay_baseline,
        "candidate_decisions": decisions,
        "decision": decision,
        "selected_candidate_key": selected_key,
        "selected_lambda_reg": selected_lambda,
        "selected_full_val_parameters": selected_parameters,
    }
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "OOF_VAL_SELECTION.json", summary)
    with (output_dir / "OOF_VAL_METRICS.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", *STANDARD_METRICS, "AP80", "AP85", "AP90", "AP95"])
        writer.writerow(
            ["native", *[replay_baseline[name] for name in STANDARD_METRICS],
             *[replay_baseline[name] for name in ("AP80", "AP85", "AP90", "AP95")]]
        )
        for key, metrics in metrics_by_candidate.items():
            writer.writerow(
                [key, *[metrics[name] for name in STANDARD_METRICS],
                 *[metrics[name] for name in ("AP80", "AP85", "AP90", "AP95")]]
            )
    print("RT-DETR QCCR OOF VAL COCO SELECTION: PASS")
    print("Decision:", decision)
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
