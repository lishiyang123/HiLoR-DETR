#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch

from benchmark_runtime import (benchmark, enforce_thread_protocol, load_test_index,
                               prepare_image, profile_flops)
from common import count_parameters, require, sha256


OFFICIAL_REPARAMETERIZATION = (
    "https://github.com/WongKinYiu/yolov9/blob/"
    "5b10b5099005a45332ebe7669f8480d5fc791a65/tools/reparameterization.ipynb"
)


def normalize_names(value):
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
    return list(value)


def load_checkpoint(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = payload.get("ema") or payload.get("model")
    require(model is not None, "YOLOv9 checkpoint has no model/EMA")
    return model.float().eval()


def source_key_for_yolov9_c(target_key: str) -> str | None:
    match = re.match(r"^model\.(\d+)\.(.+)$", target_key)
    require(match is not None, f"Unexpected GELAN-C state key: {target_key}")
    index = int(match.group(1))
    suffix = match.group(2)
    if index < 22:
        return f"model.{index + 1}.{suffix}"
    require(index == 22, f"Unexpected GELAN-C layer index: {target_key}")
    if suffix.startswith("cv2."):
        return "model.38.cv4." + suffix[len("cv2."):]
    if suffix.startswith("cv3."):
        return "model.38.cv5." + suffix[len("cv3."):]
    if suffix.startswith("dfl."):
        return "model.38.dfl2." + suffix[len("dfl."):]
    return None


def official_reparameterize(model_class, source: Path, dual_model):
    config = source / "models/detect/gelan-c.yaml"
    require(config.is_file(), f"Missing official GELAN-C deployment config: {config}")
    nc = int(dual_model.nc)
    converted = model_class(str(config), ch=3, nc=nc, anchors=3).float().eval()
    converted.names = dual_model.names
    converted.nc = nc
    source_state = dual_model.state_dict()
    target_state = converted.state_dict()
    mapped = []
    mapped_targets = set()
    constructor_buffers = []
    for target_key, target_value in target_state.items():
        source_key = source_key_for_yolov9_c(target_key)
        if source_key is None:
            constructor_buffers.append(target_key)
            continue
        require(source_key in source_state,
                f"Official YOLOv9-C mapping source is missing: {source_key}")
        source_value = source_state[source_key]
        require(tuple(source_value.shape) == tuple(target_value.shape),
                f"Official YOLOv9-C mapping shape mismatch: {target_key} <- {source_key}")
        target_state[target_key] = source_value.detach().clone()
        mapped.append({"target": target_key, "source": source_key})
        mapped_targets.add(target_key)
    converted.load_state_dict(target_state, strict=True)
    trainable = {name for name, _ in converted.named_parameters()}
    require(not (trainable & set(constructor_buffers)),
            "Official conversion left trainable parameters at constructor values")
    require(trainable <= mapped_targets,
            "Official conversion did not map every GELAN-C trainable tensor")
    return converted, {
        "official_reference": OFFICIAL_REPARAMETERIZATION,
        "deployment_config": str(config.resolve()),
        "deployment_config_sha256": sha256(config),
        "mapped_tensors": len(mapped),
        "constructor_only_buffers": constructor_buffers,
        "mapping": "GELAN-C layers 0:21 <- YOLOv9-C layers 1:22; "
                   "single DDetect cv2/cv3/dfl <- DualDDetect cv4/cv5/dfl2 at layer 38",
    }


def dual_prediction(output):
    require(isinstance(output, (tuple, list)) and len(output) >= 1,
            "Unexpected YOLOv9-C dual-head output")
    first = output[0]
    require(isinstance(first, (tuple, list)) and len(first) >= 2,
            "Unexpected YOLOv9-C dual prediction branches")
    return first[1]


def single_prediction(output):
    require(isinstance(output, (tuple, list)) and len(output) >= 1,
            "Unexpected converted GELAN-C output")
    prediction = output[0]
    require(torch.is_tensor(prediction), "Converted GELAN-C prediction is not a tensor")
    return prediction


def compare_tensor(reference: torch.Tensor, candidate: torch.Tensor,
                   *, atol: float, rtol: float, label: str,
                   enforce: bool = True) -> dict:
    require(reference.shape == candidate.shape, f"{label} output shape changed")
    delta = (reference - candidate).abs()
    maximum = float(delta.max().item())
    mean = float(delta.mean().item())
    passed = bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol))
    if enforce:
        require(passed, f"{label} numerical equivalence failed: max_abs={maximum:.9g}")
    return {"max_abs": maximum, "mean_abs": mean, "atol": atol, "rtol": rtol,
            "passed": passed, "gate_enforced": enforce}


def native_nms(non_max_suppression, prediction: torch.Tensor):
    return non_max_suppression(
        prediction.float().clone(), conf_thres=0.001, iou_thres=0.7,
        classes=None, agnostic=False, multi_label=True, labels=(), max_det=300,
    )


def compare_detections(reference, candidate, *, atol: float, rtol: float, label: str) -> dict:
    require(len(reference) == len(candidate), f"{label} batch output count changed")
    maximum = 0.0
    detections = 0
    for left, right in zip(reference, candidate):
        require(left.shape == right.shape, f"{label} NMS output shape/order changed")
        detections += int(left.shape[0])
        if left.numel():
            maximum = max(maximum, float((left - right).abs().max().item()))
            require(torch.allclose(left, right, atol=atol, rtol=rtol),
                    f"{label} NMS output changed: max_abs={maximum:.9g}")
    return {"detections": detections, "max_abs": maximum, "atol": atol,
            "rtol": rtol, "passed": True}


def aligned_box_iou(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(reference[:, :2], candidate[:, :2])
    bottom_right = torch.minimum(reference[:, 2:4], candidate[:, 2:4])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    reference_area = ((reference[:, 2] - reference[:, 0]).clamp(min=0) *
                      (reference[:, 3] - reference[:, 1]).clamp(min=0))
    candidate_area = ((candidate[:, 2] - candidate[:, 0]).clamp(min=0) *
                      (candidate[:, 3] - candidate[:, 1]).clamp(min=0))
    return intersection / (reference_area + candidate_area - intersection).clamp(min=1.0e-12)


def compare_fused_detections(reference, candidate) -> dict:
    """Record per-image fusion diagnostics without tuning on individual boxes."""
    require(len(reference) == len(candidate), "Fusion batch output count changed")
    detections = 0
    candidate_detections = 0
    maximum_box = 0.0
    maximum_confidence = 0.0
    minimum_iou = 1.0
    count_and_order_exact = True
    classes_exact = True
    for left, right in zip(reference, candidate):
        detections += int(left.shape[0])
        candidate_detections += int(right.shape[0])
        if left.shape != right.shape:
            count_and_order_exact = False
            classes_exact = False
            continue
        if not left.numel():
            continue
        row_classes_exact = bool(torch.equal(left[:, 5], right[:, 5]))
        classes_exact = classes_exact and row_classes_exact
        box_delta = float((left[:, :4] - right[:, :4]).abs().max().item())
        confidence_delta = float((left[:, 4] - right[:, 4]).abs().max().item())
        iou = aligned_box_iou(left[:, :4], right[:, :4])
        row_minimum_iou = float(iou.min().item())
        maximum_box = max(maximum_box, box_delta)
        maximum_confidence = max(maximum_confidence, confidence_delta)
        minimum_iou = min(minimum_iou, row_minimum_iou)
    return {
        "reference_detections": detections,
        "candidate_detections": candidate_detections,
        "count_and_order_exact": count_and_order_exact,
        "classes_exact": classes_exact,
        "max_box_abs_px": maximum_box,
        "max_confidence_abs": maximum_confidence,
        "min_aligned_iou": minimum_iou,
        "role": "diagnostic only; complete Val COCO metrics are the fusion gate",
    }


def detections_to_coco(detections, image_id: int, category_ids: list[int]) -> list[dict]:
    rows = []
    for detection in detections:
        for value in detection.detach().cpu().float().tolist():
            x1, y1, x2, y2, confidence, class_index = value
            index = int(class_index)
            require(0 <= index < len(category_ids), "YOLOv9 class index is out of range")
            rows.append({
                "image_id": int(image_id),
                "category_id": int(category_ids[index]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(confidence),
            })
    return rows


def evaluate_coco(annotation: Path, predictions: list[dict], image_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    require(predictions, "Fusion equivalence produced no Val detections")
    ground_truth = COCO(str(annotation))
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()

    # COCOeval.summarize() hard-codes maxDets=100 for its first AP row. With
    # the released max_det=300 inference protocol that produces a misleading
    # -1. Summarize the accumulated precision/recall tensors directly at the
    # declared maxDets values instead.
    precision = evaluator.eval["precision"]  # T x R x K x A x M
    recall = evaluator.eval["recall"]        # T x K x A x M

    def valid_mean(values) -> float:
        valid = values[values > -1]
        require(valid.size > 0, "COCO equivalence metric has no valid entries")
        return float(valid.mean())

    metrics = {
        "AP": valid_mean(precision[:, :, :, 0, 2]),
        "AP50": valid_mean(precision[0, :, :, 0, 2]),
        "AP75": valid_mean(precision[5, :, :, 0, 2]),
        "AP_small": valid_mean(precision[:, :, :, 1, 2]),
        "AP_medium": valid_mean(precision[:, :, :, 2, 2]),
        "AP_large": valid_mean(precision[:, :, :, 3, 2]),
        "AR1": valid_mean(recall[:, :, 0, 0]),
        "AR10": valid_mean(recall[:, :, 0, 1]),
        "AR300": valid_mean(recall[:, :, 0, 2]),
        "AR_small": valid_mean(recall[:, :, 1, 2]),
        "AR_medium": valid_mean(recall[:, :, 2, 2]),
        "AR_large": valid_mean(recall[:, :, 3, 2]),
    }
    print("YOLOv9 fusion candidate Val COCO metrics: " +
          json.dumps(metrics, sort_keys=True))
    return metrics


def verify_deployment_equivalence(dual_model, converted, non_max_suppression,
                                  annotation: Path, image_root: Path):
    require(torch.cuda.is_available(), "CUDA is unavailable for deployment equivalence")
    rows = load_test_index(annotation, image_root, expected_count=249)
    annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
    categories = sorted(annotation_payload["categories"], key=lambda row: int(row["id"]))
    category_ids = [int(row["id"]) for row in categories]
    category_names = [str(row["name"]) for row in categories]
    require(normalize_names(converted.names) == category_names,
            "YOLOv9 class names do not match CDW1630 Val categories")
    dual_model = dual_model.cuda().float().eval()
    converted = converted.cuda().float().eval()
    reparameterization_rows = []
    cached_inputs = []
    cached_converted = []
    with torch.inference_mode():
        for row in rows:
            image, _ = prepare_image(row)
            image = image.cuda()
            dual_raw = dual_prediction(dual_model(image, augment=False))
            converted_raw = single_prediction(converted(image, augment=False))
            raw = compare_tensor(dual_raw, converted_raw, atol=1.0e-5, rtol=1.0e-5,
                                 label="official reparameterization")
            nms = compare_detections(
                native_nms(non_max_suppression, dual_raw),
                native_nms(non_max_suppression, converted_raw),
                atol=1.0e-5, rtol=1.0e-5, label="official reparameterization",
            )
            reparameterization_rows.append({"image_id": row["image_id"],
                                            "raw": raw, "postprocess": nms})
            cached_inputs.append(image.cpu())
            cached_converted.append(converted_raw.cpu())
    dual_model = dual_model.cpu()
    del dual_model
    converted = converted.cpu().float().eval()
    gc.collect()
    torch.cuda.empty_cache()

    batchnorm_before = sum(1 for module in converted.modules()
                           if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    fusion_candidate = copy.deepcopy(converted)
    fuse = getattr(fusion_candidate, "fuse", None)
    require(callable(fuse), "Converted GELAN-C model has no official fuse method")
    deployed = fuse()
    if deployed is not None:
        fusion_candidate = deployed
    fusion_candidate = fusion_candidate.cuda().float().eval()
    fusion_rows = []
    unfused_coco = []
    fused_coco = []
    with torch.inference_mode():
        for row, cpu_image, cpu_reference in zip(rows, cached_inputs, cached_converted):
            fused_raw = single_prediction(
                fusion_candidate(cpu_image.cuda(), augment=False))
            reference = cpu_reference.cuda()
            raw = compare_tensor(reference, fused_raw, atol=5.0e-4, rtol=1.0e-4,
                                 label="Conv-BN fusion", enforce=False)
            raw["role"] = "diagnostic only; complete Val COCO metrics are the fusion gate"
            reference_nms = native_nms(non_max_suppression, reference)
            fused_nms = native_nms(non_max_suppression, fused_raw)
            nms = compare_fused_detections(reference_nms, fused_nms)
            unfused_coco.extend(detections_to_coco(
                reference_nms, row["image_id"], category_ids))
            fused_coco.extend(detections_to_coco(
                fused_nms, row["image_id"], category_ids))
            fusion_rows.append({"image_id": row["image_id"],
                                "raw": raw, "postprocess": nms})
    image_ids = [row["image_id"] for row in rows]
    unfused_metrics = evaluate_coco(annotation, unfused_coco, image_ids)
    fused_metrics = evaluate_coco(annotation, fused_coco, image_ids)
    metric_deltas = {
        name: fused_metrics[name] - unfused_metrics[name]
        for name in unfused_metrics
    }
    maximum_metric_delta = max(abs(value) for value in metric_deltas.values())
    fusion_passed = maximum_metric_delta <= 1.0e-4
    fusion_candidate = fusion_candidate.cpu().float().eval()
    fusion_candidate_parameters = count_parameters(fusion_candidate)
    fusion_candidate_batchnorm_after = sum(
        1 for module in fusion_candidate.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    require(fusion_candidate_batchnorm_after == 0,
            "YOLOv9 Conv-BN fusion candidate is structurally incomplete: "
            f"{batchnorm_before} -> {fusion_candidate_batchnorm_after}")
    if fusion_passed:
        selected = fusion_candidate
        selected_deployment = "fused_single_head"
        fallback_applied = False
    else:
        selected = converted
        selected_deployment = "unfused_single_head"
        fallback_applied = True
        del fusion_candidate
    selected = selected.cpu().float().eval()
    selected_batchnorm_after = sum(
        1 for module in selected.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    torch.cuda.empty_cache()
    return selected, {
        "status": "PASS",
        "split": "CDW1630 Val",
        "validation_images": len(rows),
        "test_images_used": 0,
        "image_ids": [row["image_id"] for row in rows],
        "dual_to_single_head": reparameterization_rows,
        "unfused_to_fused": fusion_rows,
        "fusion_coco_metric_equivalence": {
            "evaluator": "pycocotools COCOeval",
            "max_detections": 300,
            "unfused": unfused_metrics,
            "fused": fused_metrics,
            "delta_fused_minus_unfused": metric_deltas,
            "max_abs_metric_delta": maximum_metric_delta,
            "acceptance_threshold": 1.0e-4,
            "passed": fusion_passed,
        },
        "fusion_candidate_parameters": fusion_candidate_parameters,
        "fusion_candidate_batchnorm_layers_after": fusion_candidate_batchnorm_after,
        "selected_deployment": selected_deployment,
        "fallback_applied": fallback_applied,
        "selected_batchnorm_layers_after": selected_batchnorm_after,
        "decision_rule": "use fused single-head only when maximum absolute Val COCO "
                         "metric delta <= 1e-4; otherwise use unfused single-head",
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="YOLOv9-C")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--class-names-json", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--equivalence-annotation", type=Path, required=True)
    parser.add_argument("--equivalence-image-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("SMOKE", "FORMAL"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    enforce_thread_protocol()

    source = args.source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    annotation = args.annotation.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    equivalence_annotation = args.equivalence_annotation.resolve(strict=True)
    equivalence_image_root = args.equivalence_image_root.resolve(strict=True)
    require(sha256(checkpoint) == args.checkpoint_sha256, "Checkpoint SHA256 changed")
    sys.path.insert(0, str(source))
    os.chdir(source)
    from models.yolo import Model
    from utils.general import non_max_suppression, scale_boxes

    dual_model = load_checkpoint(checkpoint)
    require(normalize_names(dual_model.names) == json.loads(args.class_names_json),
            "YOLOv9 class mapping changed")
    training_graph_parameters = count_parameters(dual_model)
    training_graph_profile = profile_flops(dual_model)

    converted, conversion = official_reparameterize(Model, source, dual_model)
    converted_unfused_parameters = count_parameters(converted)
    converted_unfused_profile = profile_flops(converted)
    converted, equivalence = verify_deployment_equivalence(
        dual_model, converted, non_max_suppression,
        equivalence_annotation, equivalence_image_root
    )
    del dual_model
    postdeploy = count_parameters(converted)
    batchnorm_after = sum(1 for module in converted.modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    fusion_applied = equivalence["selected_deployment"] == "fused_single_head"

    def forward_fn(image):
        return single_prediction(converted(image, augment=False))

    def postprocess_fn(prediction, _original_sizes, original_hw):
        detections = native_nms(non_max_suppression, prediction)
        for detection in detections:
            if len(detection):
                detection[:, :4] = scale_boxes((640, 640), detection[:, :4], original_hw)
        return detections

    benchmark(
        name=args.name, detector=converted, postprocessor=None,
        forward_fn=forward_fn, postprocess_fn=postprocess_fn,
        count_outputs_fn=lambda result: sum(len(row) for row in result),
        annotation=annotation, image_root=image_root,
        protocol=args.protocol.resolve(strict=True), checkpoint=checkpoint,
        checkpoint_role="seed0 Val-selected checkpoint",
        source_record={"path": str(source), "commit": args.source_commit},
        config_record={
            "training_graph": "models/detect/yolov9-c.yaml",
            "deployed_graph": conversion["deployment_config"],
            "deployed_graph_sha256": conversion["deployment_config_sha256"],
        },
        predeploy_parameters=converted_unfused_parameters,
        postdeploy_parameters=postdeploy, postprocessor_parameters=0, fitted_scalars=0,
        deployment={
            "method": "official YOLOv9-C reparameterization to GELAN-C; "
                      "Conv-BN fusion only if the frozen Val equivalence gate passes",
            "official_reparameterization": conversion,
            "reparameterization_applied": True,
            "deployment_equivalence": equivalence,
            "training_dual_graph_parameters": training_graph_parameters,
            "training_dual_graph_profile": training_graph_profile,
            "converted_single_head_unfused_parameters": converted_unfused_parameters,
            "converted_single_head_unfused_profile": converted_unfused_profile,
            "batchnorm_layers_after": batchnorm_after,
            "fusion_applied": fusion_applied,
            "selected_deployment": equivalence["selected_deployment"],
            "fusion_fallback_applied": equivalence["fallback_applied"],
            "native_postprocess": {
                "type": "official YOLOv9 single-head NMS",
                "prediction_branch": "converted GELAN-C detector(image)[0]",
                "confidence_threshold": 0.001, "iou_threshold": 0.7,
                "max_detections": 300, "multi_label": True,
            },
        },
        output=args.output, mode=args.mode, warmup=args.warmup, repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
