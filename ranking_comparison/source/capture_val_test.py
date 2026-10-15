#!/usr/bin/env python3
"""Capture native Val/Test outputs without modifying the formal source tree."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def parse_splits(text: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in text.split(",") if item.strip())
    if not values or any(item not in {"val", "test"} for item in values):
        raise argparse.ArgumentTypeError("--splits must contain val and/or test")
    return tuple(dict.fromkeys(values))


def load_formal_capture_module(branch_root: Path):
    path = branch_root / "tools" / "calibration" / "common" / "capture_train_val.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("formal_capture_train_val", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-config", type=Path, required=True)
    parser.add_argument("--test-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--capture-prefix", required=True)
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=("val", "test"),
        help="comma-separated capture set; use 'test' to reuse an existing Val capture",
    )
    args = parser.parse_args()

    branch_root = args.branch_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(branch_root))
    formal_capture = load_formal_capture_module(branch_root)

    checkpoint_sha = formal_capture.sha256(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha:
        raise RuntimeError(
            "detector checkpoint SHA mismatch: "
            f"expected={args.expected_checkpoint_sha}, actual={checkpoint_sha}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to produce the frozen capture")

    from engine.core import YAMLConfig

    device = torch.device("cuda:0")
    config = YAMLConfig(str(args.val_config.resolve()))
    model = config.model.to(device)
    incompatible = model.load_state_dict(
        formal_capture.checkpoint_state(checkpoint), strict=True
    )
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"strict load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    postprocessor = config.postprocessor.to(device).eval()
    if int(postprocessor.num_classes) != args.num_classes:
        raise RuntimeError(
            f"expected {args.num_classes} classes, got {postprocessor.num_classes}"
        )
    if bool(postprocessor.qccr_enabled) or bool(postprocessor.umr_enabled):
        raise RuntimeError("capture configs must disable QCCR and UMR")

    summaries = {}
    config_paths = {
        "val": args.val_config.resolve(),
        "test": args.test_config.resolve(),
    }
    for split in args.splits:
        config_path = config_paths[split]
        summaries[split] = formal_capture.capture_split(
            config_path=config_path,
            split=split,
            cfg_model=model,
            postprocessor=postprocessor,
            device=device,
            checkpoint_path=checkpoint,
            checkpoint_sha=checkpoint_sha,
            output_dir=output_dir,
            capture_prefix=args.capture_prefix,
        )

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "strict_load_missing": missing,
        "strict_load_unexpected": unexpected,
        "splits": summaries,
    }
    (output_dir / "VAL_TEST_CAPTURE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("FROZEN DETECTOR CAPTURE: PASS / " + ",".join(args.splits))


if __name__ == "__main__":
    main()
