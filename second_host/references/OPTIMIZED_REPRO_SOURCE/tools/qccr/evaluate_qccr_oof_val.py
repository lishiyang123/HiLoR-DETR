#!/usr/bin/env python
"""Evaluate QCCR candidates using image-level OOF predictions on Val only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchvision


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.deim.qccr import (
    apply_qccr,
    compute_competition_risk,
)
from metrics import extract_exact_metrics


STANDARD_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def xyxy_to_xywh(box):
    return [
        float(box[0]),
        float(box[1]),
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ]


def run_coco(annotation, detections, image_ids):
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections)

    evaluator = COCOeval_faster(
        coco_gt,
        iouType="bbox",
        print_function=print,
        separate_eval=True,
    )
    evaluator.cocoDt = coco_dt
    evaluator.params.imgIds = sorted(image_ids)

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator.eval


def baseline_detections(capture):
    detections = []

    for record in capture["records"]:
        image_id = int(record["image_id"])
        labels = record["post_labels"]
        scores = record["post_scores"]
        boxes = record["post_boxes"]

        for label, score, box in zip(
            labels,
            scores,
            boxes,
        ):
            detections.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": xyxy_to_xywh(box),
                "score": float(score),
            })

    return detections


def prepare_records(capture):
    prepared = []

    for record in capture["records"]:
        logits = record["pred_logits"].float()
        boxes = record["pred_boxes"].float()

        risk = compute_competition_risk(
            logits.unsqueeze(0),
            boxes.unsqueeze(0),
        )[0]

        height, width = [
            int(value)
            for value in record["orig_size"].tolist()
        ]

        scale = torch.tensor(
            [width, height, width, height],
            dtype=torch.float32,
        )

        boxes_absolute = torchvision.ops.box_convert(
            boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ) * scale

        prepared.append({
            "image_id": int(record["image_id"]),
            "logits": logits,
            "boxes": boxes,
            "risk": risk,
            "boxes_absolute": boxes_absolute,
        })

    return prepared


def qccr_oof_detections(
    prepared,
    fit_result,
    candidate_key,
    num_classes,
    num_top_queries,
):
    candidate = fit_result["candidates"][candidate_key]

    fold_parameters = {
        int(entry["fold"]): entry["parameters"]
        for entry in candidate["folds"]
    }

    fold_by_image = {
        int(image_id): int(fold)
        for image_id, fold
        in fit_result["fold_by_image"].items()
    }

    detections = []

    for record in prepared:
        image_id = record["image_id"]
        fold = fold_by_image[image_id]
        parameters = fold_parameters[fold]

        calibrated = apply_qccr(
            logits=record["logits"],
            competition_risk=record["risk"],
            beta0=parameters["beta0"],
            beta_positive=parameters[
                "beta_positive"
            ],
            beta_negative=parameters[
                "beta_negative"
            ],
            duplicate_penalty=parameters[
                "duplicate_penalty"
            ],
        )

        all_scores = torch.sigmoid(calibrated).flatten()

        top_scores, top_indices = torch.topk(
            all_scores,
            k=num_top_queries,
            dim=0,
        )

        labels = top_indices % num_classes
        query_indices = top_indices // num_classes
        selected_boxes = record[
            "boxes_absolute"
        ][query_indices]

        for label, score, box in zip(
            labels,
            top_scores,
            selected_boxes,
        ):
            detections.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": xyxy_to_xywh(box),
                "score": float(score),
            })

    return detections


def save_tables(output_dir, metrics_by_method):
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_path = output_dir / "oof_standard12.csv"
    with standard_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        methods = list(metrics_by_method.keys())
        writer.writerow(["metric", *methods])

        for metric in STANDARD_METRICS:
            writer.writerow([
                metric,
                *[
                    metrics_by_method[method]["overall"][
                        metric
                    ]
                    for method in methods
                ],
            ])

    per_iou_path = output_dir / "oof_per_iou.csv"
    with per_iou_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        methods = list(metrics_by_method.keys())
        writer.writerow(["metric", *methods])

        for threshold in (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ):
            ap_key = f"AP{threshold:.2f}"
            ar_key = f"AR{threshold:.2f}"

            writer.writerow([
                ap_key,
                *[
                    metrics_by_method[method][
                        "per_iou_ap"
                    ][ap_key]
                    for method in methods
                ],
            ])
            writer.writerow([
                ar_key,
                *[
                    metrics_by_method[method][
                        "per_iou_ar"
                    ][ar_key]
                    for method in methods
                ],
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--fits", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    capture_path = Path(args.capture).resolve()
    baseline_eval_path = Path(
        args.baseline_eval
    ).resolve()
    fit_path = Path(args.fits).resolve()
    annotation_path = Path(args.annotation).resolve()
    output_dir = Path(args.output_dir).resolve()

    capture = torch.load(
        capture_path,
        map_location="cpu",
        weights_only=False,
    )
    fit_result = json.loads(
        fit_path.read_text(encoding="utf-8")
    )

    if capture.get("split") != "val":
        raise RuntimeError("OOF selection must use Val")

    if fit_result.get("fold_count") != 5:
        raise RuntimeError("expected five-fold fit")

    num_classes = int(
        capture["postprocessor_num_classes"]
    )
    num_top_queries = int(
        capture["postprocessor_num_top_queries"]
    )
    image_ids = [
        int(record["image_id"])
        for record in capture["records"]
    ]

    baseline_eval = torch.load(
        baseline_eval_path,
        map_location="cpu",
        weights_only=False,
    )
    frozen_baseline = extract_exact_metrics(
        baseline_eval
    )

    baseline_replay = extract_exact_metrics(
        run_coco(
            annotation=annotation_path,
            detections=baseline_detections(capture),
            image_ids=image_ids,
        )
    )

    reproduction_errors = {
        metric: abs(
            baseline_replay["overall"][metric]
            - frozen_baseline["overall"][metric]
        )
        for metric in STANDARD_METRICS
    }
    max_reproduction_error = max(
        reproduction_errors.values()
    )

    if max_reproduction_error > 1.0e-6:
        raise RuntimeError(
            "baseline COCO replay mismatch: "
            f"{max_reproduction_error}"
        )

    prepared = prepare_records(capture)

    metrics_by_method = {
        "PD_baseline": baseline_replay,
    }

    for candidate_key in fit_result["candidates"]:
        detections = qccr_oof_detections(
            prepared=prepared,
            fit_result=fit_result,
            candidate_key=candidate_key,
            num_classes=num_classes,
            num_top_queries=num_top_queries,
        )

        metrics_by_method[
            f"QCCR_lambda_{candidate_key}"
        ] = extract_exact_metrics(
            run_coco(
                annotation=annotation_path,
                detections=detections,
                image_ids=image_ids,
            )
        )

    save_tables(
        output_dir=output_dir,
        metrics_by_method=metrics_by_method,
    )

    baseline = metrics_by_method[
        "PD_baseline"
    ]["overall"]

    decisions = {}

    for candidate_key in fit_result["candidates"]:
        method = f"QCCR_lambda_{candidate_key}"
        result = metrics_by_method[method]["overall"]

        deltas = {
            key: result[key] - baseline[key]
            for key in (
                "AP",
                "AP50",
                "AP75",
                "APm",
                "APl",
                "AR100",
                "AP90",
                "AP95",
            )
        }

        guards = {
            "ap_gain_at_least_0.1pp":
                deltas["AP"] >= 0.001,
            "ap75_drop_no_more_than_0.05pp":
                deltas["AP75"] >= -0.0005,
            "apm_not_lower":
                deltas["APm"] >= 0.0,
            "ar100_drop_no_more_than_0.1pp":
                deltas["AR100"] >= -0.001,
            "ap90_drop_no_more_than_0.05pp":
                deltas["AP90"] >= -0.0005,
        }

        decisions[candidate_key] = {
            "method": method,
            "metrics": result,
            "deltas_vs_pd": deltas,
            "guards": guards,
            "qualified": bool(all(guards.values())),
        }

    qualified = [
        key
        for key, decision in decisions.items()
        if decision["qualified"]
    ]

    if qualified:
        selected_key = max(
            qualified,
            key=lambda key: (
                decisions[key]["metrics"]["AP"],
                decisions[key]["metrics"]["AP75"],
                decisions[key]["metrics"]["APm"],
            ),
        )
        decision_name = "OOF_VAL_GO"
        selected_lambda = fit_result[
            "candidates"
        ][selected_key]["lambda_reg"]
        selected_parameters = fit_result[
            "candidates"
        ][selected_key][
            "full_val_fit"
        ]["parameters"]
    else:
        selected_key = None
        selected_lambda = None
        selected_parameters = None
        decision_name = "OOF_VAL_NO_GO"

    summary = {
        "format_version": 1,
        "protocol": "image-level-five-fold-OOF-Val",
        "capture": str(capture_path),
        "fits": str(fit_path),
        "annotation": str(annotation_path),
        "baseline_replay_max_error":
            max_reproduction_error,
        "baseline_metrics": baseline,
        "candidate_decisions": decisions,
        "decision": decision_name,
        "selected_candidate_key": selected_key,
        "selected_lambda_reg": selected_lambda,
        "selected_full_val_parameters":
            selected_parameters,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "oof_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== OOF VAL SUMMARY =====")
    print(
        "baseline replay max error:",
        max_reproduction_error,
    )
    print(
        "PD baseline:",
        "AP=%.6f AP75=%.6f APm=%.6f "
        "AR100=%.6f AP90=%.6f AP95=%.6f"
        % (
            baseline["AP"],
            baseline["AP75"],
            baseline["APm"],
            baseline["AR100"],
            baseline["AP90"],
            baseline["AP95"],
        )
    )

    for key, item in decisions.items():
        metric = item["metrics"]
        delta = item["deltas_vs_pd"]

        print(
            "lambda=%s AP=%.6f (%+.6f) "
            "AP75=%.6f (%+.6f) "
            "APm=%.6f (%+.6f) "
            "AR100=%.6f (%+.6f) "
            "qualified=%s"
            % (
                key,
                metric["AP"],
                delta["AP"],
                metric["AP75"],
                delta["AP75"],
                metric["APm"],
                delta["APm"],
                metric["AR100"],
                delta["AR100"],
                item["qualified"],
            )
        )

    print("decision:", decision_name)
    print("selected lambda:", selected_lambda)
    print("summary:", summary_path)
    print("QCCR OOF VAL COCO EVALUATION: PASS")


if __name__ == "__main__":
    main()
