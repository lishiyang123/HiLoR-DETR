#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from benchmark_runtime import benchmark, enforce_thread_protocol, profile_flops
from common import count_parameters, floating_tree_to_fp32, require, sha256
from scoped_common import trim_retired_branches


def checkpoint_weights(checkpoint: dict):
    ema = checkpoint.get("ema")
    weights = ema.get("module") if isinstance(ema, dict) else None
    if isinstance(weights, dict):
        return weights, "ema.module"
    weights = checkpoint.get("model")
    require(isinstance(weights, dict), "Checkpoint lacks ema.module and model")
    return weights, "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-flavor", choices=("src", "engine"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expect-qccr", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--expect-umr", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--fitted-scalars", type=int, default=0)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("SMOKE", "FORMAL"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    enforce_thread_protocol()

    source = args.source.resolve(strict=True)
    config = args.config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    require(sha256(checkpoint_path) == args.checkpoint_sha256, "Checkpoint SHA256 changed")
    os.chdir(source)
    sys.path.insert(0, str(source))
    if args.source_flavor == "src":
        from src.core import YAMLConfig
    else:
        from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config))
    detector = cfg.model
    postprocessor = cfg.postprocessor
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights, weight_source = checkpoint_weights(checkpoint)
    load_result = detector.load_state_dict(weights, strict=True)
    post_state = checkpoint.get("postprocessor")
    if isinstance(post_state, dict):
        postprocessor.load_state_dict(post_state, strict=True)
    del checkpoint, weights

    actual_qccr = bool(getattr(postprocessor, "qccr_enabled", False))
    actual_umr = bool(getattr(postprocessor, "umr_enabled", False))
    if args.expect_qccr >= 0:
        require(actual_qccr == bool(args.expect_qccr), "QCCR switch mismatch")
    if args.expect_umr >= 0:
        require(actual_umr == bool(args.expect_umr), "UMR switch mismatch")
    predeploy = count_parameters(detector)
    predeploy_profile = profile_flops(detector.float().eval())
    deploy = getattr(detector, "deploy", None)
    require(callable(deploy), "DETR-family detector has no official deploy method")
    deployed = deploy()
    if deployed is not None:
        detector = deployed
    detector = detector.float().eval()
    retired_branches = trim_retired_branches(detector)
    postdeploy = count_parameters(detector)
    detector_ids = {id(parameter) for parameter in detector.parameters()}
    post_parameters = sum(parameter.numel() for parameter in postprocessor.parameters()
                          if id(parameter) not in detector_ids)

    def forward_fn(image):
        return detector(image)

    def postprocess_fn(raw, original_sizes, _original_hw):
        return postprocessor(floating_tree_to_fp32(raw), original_sizes)

    def count_outputs(result):
        total = 0
        for row in result:
            if isinstance(row, dict) and "scores" in row:
                total += len(row["scores"])
            else:
                total += len(row)
        return total

    benchmark(
        name=args.name,
        detector=detector,
        postprocessor=postprocessor,
        forward_fn=forward_fn,
        postprocess_fn=postprocess_fn,
        count_outputs_fn=count_outputs,
        annotation=args.annotation.resolve(strict=True),
        image_root=args.image_root.resolve(strict=True),
        protocol=args.protocol.resolve(strict=True),
        checkpoint=checkpoint_path,
        checkpoint_role="seed0 Val-selected checkpoint",
        source_record={"path": str(source), "flavor": args.source_flavor},
        config_record={"path": str(config), "sha256": sha256(config)},
        predeploy_parameters=predeploy,
        postdeploy_parameters=postdeploy,
        postprocessor_parameters=post_parameters,
        fitted_scalars=args.fitted_scalars,
        deployment={
            "method": "official detector.deploy()",
            "deploy_applied": True,
            "weight_source": weight_source,
            "strict_load": str(load_result),
            "predeploy_profile": predeploy_profile,
            "qccr_enabled": actual_qccr,
            "umr_enabled": actual_umr,
            "native_postprocess": type(postprocessor).__name__,
            "removed_retired_repvg_branches": retired_branches,
        },
        output=args.output,
        mode=args.mode,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
