#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch

from benchmark_runtime import benchmark, enforce_thread_protocol, profile_flops
from common import count_parameters, floating_tree_to_fp32, require, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-class", choices=("YOLO", "YOLOv10"), required=True)
    parser.add_argument("--native-postprocess", choices=("nms", "v10"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("SMOKE", "FORMAL"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    enforce_thread_protocol()

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ops
    if args.model_class == "YOLOv10":
        from ultralytics import YOLOv10
        wrapper_class = YOLOv10
    else:
        wrapper_class = YOLO
    try:
        from ultralytics.utils.nms import non_max_suppression
        nms_import = "ultralytics.utils.nms.non_max_suppression"
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
        nms_import = "ultralytics.utils.ops.non_max_suppression"

    checkpoint = args.checkpoint.resolve(strict=True)
    require(sha256(checkpoint) == args.checkpoint_sha256, "Checkpoint SHA256 changed")
    source_record = {
        "ultralytics_version": getattr(ultralytics, "__version__", "UNKNOWN"),
        "imported_file": str(Path(ultralytics.__file__).resolve()),
    }
    if args.source is not None:
        source = args.source.resolve(strict=True)
        imported = Path(ultralytics.__file__).resolve()
        require(source in imported.parents, f"Imported Ultralytics is outside bound source: {imported}")
        source_record["path"] = str(source)
        if args.source_commit:
            source_record["declared_commit"] = args.source_commit
            if (source / ".git").exists():
                commit = subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip()
                require(commit == args.source_commit, "Official-source commit changed")
                source_record["verified_git_commit"] = commit

    wrapper = wrapper_class(str(checkpoint))
    detector = wrapper.model.float().eval()
    predeploy = count_parameters(detector)
    predeploy_flops = profile_flops(detector)
    batchnorm_before = sum(1 for module in detector.modules()
                           if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    fuse = getattr(detector, "fuse", None)
    require(callable(fuse), "Ultralytics detector has no official fuse method")
    try:
        deployed = fuse(verbose=False)
    except TypeError:
        deployed = fuse()
    if deployed is not None:
        detector = deployed
    detector = detector.float().eval()
    postdeploy = count_parameters(detector)
    batchnorm_after = sum(1 for module in detector.modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    require(batchnorm_after == 0,
            f"Conv-BN fusion incomplete: before={batchnorm_before} after={batchnorm_after}")

    def forward_fn(image):
        return detector(image)

    def v10_postprocess(raw, _original_sizes, original_hw):
        prediction = raw["one2one"] if isinstance(raw, dict) else raw
        prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        prediction = floating_tree_to_fp32(prediction)
        if prediction.shape[-1] == 6:
            detections = prediction
        else:
            prediction = prediction.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(prediction, 300, prediction.shape[-1] - 4)
            boxes = ops.xywh2xyxy(boxes)
            detections = torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), dim=-1)
        rows = []
        for detection in detections:
            detection = detection[detection[:, 4] > 0.001]
            if len(detection):
                detection[:, :4] = ops.scale_boxes((640, 640), detection[:, :4], original_hw)
            rows.append(detection)
        return rows

    def nms_postprocess(raw, _original_sizes, original_hw):
        prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
        prediction = floating_tree_to_fp32(prediction)
        detections = non_max_suppression(
            prediction,
            conf_thres=0.001,
            iou_thres=0.7,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=5,
        )
        for detection in detections:
            if len(detection):
                detection[:, :4] = ops.scale_boxes((640, 640), detection[:, :4], original_hw)
        return detections

    postprocess_fn = v10_postprocess if args.native_postprocess == "v10" else nms_postprocess
    native = ({
        "type": "YOLOv10 end-to-end top-k",
        "confidence_threshold": 0.001,
        "max_detections": 300,
    } if args.native_postprocess == "v10" else {
        "type": "Ultralytics NMS",
        "confidence_threshold": 0.001,
        "iou_threshold": 0.7,
        "max_detections": 300,
        "multi_label": False,
        "implementation": nms_import,
    })
    deployment = {
        "method": "official detector.fuse()",
        "batchnorm_layers_before": batchnorm_before,
        "batchnorm_layers_after": batchnorm_after,
        "fusion_applied": True,
        "already_fused_at_load": batchnorm_before == 0,
        "predeploy_profile": predeploy_flops,
        "native_postprocess": native,
    }
    benchmark(
        name=args.name,
        detector=detector,
        postprocessor=None,
        forward_fn=forward_fn,
        postprocess_fn=postprocess_fn,
        count_outputs_fn=lambda result: sum(len(row) for row in result),
        annotation=args.annotation.resolve(strict=True),
        image_root=args.image_root.resolve(strict=True),
        protocol=args.protocol.resolve(strict=True),
        checkpoint=checkpoint,
        checkpoint_role="seed0 Val-selected checkpoint",
        source_record=source_record,
        config_record=None,
        predeploy_parameters=predeploy,
        postdeploy_parameters=postdeploy,
        postprocessor_parameters=0,
        fitted_scalars=0,
        deployment=deployment,
        output=args.output,
        mode=args.mode,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
