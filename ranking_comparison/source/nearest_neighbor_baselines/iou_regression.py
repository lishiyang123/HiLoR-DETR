"""Tiny frozen IoU-regression head and ``score * predicted IoU`` replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height,
         cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("pairwise_iou expects [N,4] and [M,4]")
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp_min(1.0e-9)


def build_iou_targets(
    record: Mapping[str, Any], *, num_classes: int = 5
) -> torch.Tensor:
    """Max same-class GT IoU for every query-class candidate."""

    boxes = torch.as_tensor(record["pred_boxes"], dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("pred_boxes must have shape [Q,4]")
    size = torch.as_tensor(record["orig_size"], dtype=torch.float32).reshape(-1)
    if size.numel() != 2:
        raise ValueError("orig_size must contain [height,width]")
    height, width = size
    scale = torch.stack((width, height, width, height))
    boxes_abs = cxcywh_to_xyxy(boxes) * scale

    gt_boxes = torch.as_tensor(record["gt_boxes"], dtype=torch.float32).reshape(-1, 4)
    gt_labels = torch.as_tensor(record["gt_labels"], dtype=torch.int64).reshape(-1)
    if "gt_iscrowd" in record:
        keep = torch.as_tensor(record["gt_iscrowd"], dtype=torch.int64).reshape(-1) == 0
        gt_boxes = gt_boxes[keep]
        gt_labels = gt_labels[keep]
    if len(gt_boxes) != len(gt_labels):
        raise ValueError("GT boxes and labels have inconsistent lengths")
    if len(gt_labels) and (int(gt_labels.min()) < 0 or int(gt_labels.max()) >= num_classes):
        raise ValueError("GT label is outside the IoU-regressor class range")

    targets = torch.zeros((len(boxes), num_classes), dtype=torch.float32)
    if len(gt_boxes) == 0:
        return targets
    overlaps = pairwise_iou(boxes_abs, gt_boxes)
    for class_id in range(num_classes):
        class_mask = gt_labels == class_id
        if bool(class_mask.any()):
            targets[:, class_id] = overlaps[:, class_mask].max(dim=1).values
    return targets.clamp_(0.0, 1.0)


class IoURegressorHead(nn.Module):
    """A low-capacity class-conditional IoU head over frozen decoder features."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, h_l: torch.Tensor) -> torch.Tensor:
        if h_l.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected decoder feature dimension {self.hidden_dim}, got {h_l.shape[-1]}"
            )
        return self.output(self.norm(h_l.float()))


class IoURegressorReplay:
    """Load a frozen IoU head and return product-rescored flat candidates."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        metadata: Path,
        capture_checkpoint_sha256: str | None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.metadata_path = Path(metadata).resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("weight_sha256") not in (None, _sha256(self.checkpoint)):
            raise RuntimeError("IoU-regressor metadata/weight SHA-256 mismatch")
        source_sha = self.metadata.get("source_checkpoint_sha256")
        if capture_checkpoint_sha256 and source_sha and source_sha != capture_checkpoint_sha256:
            raise RuntimeError("IoU regressor and capture belong to different detectors")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA IoU-regressor replay requested but CUDA is unavailable")
        self.model = IoURegressorHead(
            hidden_dim=int(self.metadata["hidden_dim"]),
            num_classes=int(self.metadata["num_classes"]),
        ).to(self.device)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
        ):
            raise TypeError("IoU-regressor checkpoint must be a tensor state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def scores(self, record: Mapping[str, Any]) -> np.ndarray:
        logits = torch.as_tensor(
            record["pred_logits"], dtype=torch.float32, device=self.device
        )
        h_l = torch.as_tensor(record["hL"], dtype=torch.float32, device=self.device)
        if logits.shape != (h_l.shape[0], self.model.num_classes):
            raise ValueError("pred_logits and hL do not match the frozen IoU head")
        native_score = torch.sigmoid(logits)
        predicted_iou = torch.sigmoid(self.model(h_l))
        rescored = native_score * predicted_iou
        return rescored.reshape(-1).cpu().numpy().astype(np.float64, copy=False)

    def binding(self) -> dict[str, str | None]:
        return {
            "iou_regressor_checkpoint_sha256": _sha256(self.checkpoint),
            "iou_regressor_metadata_sha256": _sha256(self.metadata_path),
            "source_checkpoint_sha256": self.metadata.get("source_checkpoint_sha256"),
        }

