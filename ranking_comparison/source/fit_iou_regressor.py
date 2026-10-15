#!/usr/bin/env python3
"""Fit a tiny IoU head on Train and select its epoch by Val regression loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from nearest_neighbor_baselines.capture_io import load_capture, sha256
from nearest_neighbor_baselines.iou_regression import (
    IoURegressorHead,
    build_iou_targets,
)


def batches(records, batch_images: int, order: list[int]):
    for start in range(0, len(order), batch_images):
        indices = order[start : start + batch_images]
        h_l = torch.cat(
            [torch.as_tensor(records[index]["hL"], dtype=torch.float32) for index in indices],
            dim=0,
        )
        targets = torch.cat(
            [build_iou_targets(records[index]) for index in indices], dim=0
        )
        yield h_l, targets


@torch.no_grad()
def validation_loss(model, records, batch_images: int, device: torch.device) -> float:
    total = 0.0
    count = 0
    model.eval()
    order = list(range(len(records)))
    for h_l, targets in batches(records, batch_images, order):
        h_l = h_l.to(device)
        targets = targets.to(device)
        loss = F.binary_cross_entropy_with_logits(
            model(h_l), targets, reduction="sum"
        )
        total += float(loss.item())
        count += int(targets.numel())
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--val-capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-images", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fitting requested but CUDA is unavailable")

    train = load_capture(args.train_capture, mmap=True)
    val = load_capture(args.val_capture, mmap=True)
    if train.get("split") != "train" or val.get("split") != "val":
        raise RuntimeError("IoU regression requires Train fitting and Val selection")
    if train.get("checkpoint_sha256") != val.get("checkpoint_sha256"):
        raise RuntimeError("Train and Val captures belong to different checkpoints")
    for payload in (train, val):
        if any("hL" not in record for record in payload["records"]):
            raise RuntimeError("every capture record must contain hL")

    hidden_dim = int(train["records"][0]["hL"].shape[-1])
    num_classes = int(train.get("postprocessor_num_classes", 5))
    model = IoURegressorHead(hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train["records"])))
        random.Random(args.seed + epoch).shuffle(order)
        train_sum = 0.0
        train_count = 0
        for h_l, targets in batches(train["records"], args.batch_images, order):
            h_l = h_l.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(h_l)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * int(targets.numel())
            train_count += int(targets.numel())
        train_loss = train_sum / max(train_count, 1)
        val_loss = validation_loss(model, val["records"], args.batch_images, device)
        history.append({"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss})
        print(f"epoch {epoch:02d}: train_bce={train_loss:.8f} val_bce={val_loss:.8f}", flush=True)
        if val_loss < best_loss - 1.0e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("IoU-regressor fitting produced no checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = args.output_dir / f"iou_regressor_seed{args.seed}.pt"
    torch.save(best_state, weight_path)
    weight_sha = sha256(weight_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "format_version": 1,
        "method": "frozen_linear_iou_regressor",
        "rescore_formula": "sigmoid(class_logit) * sigmoid(iou_regressor(hL))",
        "target_definition": "maximum IoU with a non-crowd GT of the same class; zero if absent",
        "fit_split": "Train",
        "selection_split": "Val",
        "test_used_for_fit_or_selection": False,
        "seed": int(args.seed),
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "parameter_count": parameter_count,
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_images": int(args.batch_images),
        "max_epochs": int(args.epochs),
        "patience": int(args.patience),
        "best_epoch": best_epoch,
        "best_val_bce": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "source_checkpoint_sha256": train.get("checkpoint_sha256"),
        "train_capture": str(args.train_capture.resolve()),
        "train_capture_sha256": sha256(args.train_capture),
        "val_capture": str(args.val_capture.resolve()),
        "val_capture_sha256": sha256(args.val_capture),
        "weight_sha256": weight_sha,
        "history": history,
    }
    metadata_path = args.output_dir / f"iou_regressor_seed{args.seed}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"IOU REGRESSOR FIT: PASS / best_epoch={best_epoch} val_bce={best_loss:.8f}")
    print(f"weight: {weight_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

