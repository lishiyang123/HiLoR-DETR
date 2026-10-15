"""Frozen UMR candidate-value head for RT-DETR query-class outputs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hilor_artifacts import sha256_file
from .hilor_qccr import apply_qccr, compute_competition_risk, compute_pairwise_iou


__all__ = ["UMR", "UMRHead", "build_image_features"]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height,
         cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    size1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0)
    size2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0)
    area1 = size1[:, 0] * size1[:, 1]
    area2 = size2[:, 0] * size2[:, 1]
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1.0e-9)


def _rank_matrix(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    order = torch.argsort(flat, descending=True, stable=True)
    ranks = torch.empty(flat.numel(), dtype=torch.float32, device=values.device)
    ranks[order] = torch.arange(flat.numel(), dtype=torch.float32, device=values.device)
    return ranks.reshape_as(values)


def build_image_features(
    z_raw: torch.Tensor,
    boxes: torch.Tensor,
    boxes_abs: torch.Tensor,
    decoder_features: torch.Tensor,
    qccr_logit: torch.Tensor,
    qccr_score: torch.Tensor,
    risk: torch.Tensor,
    topk: int,
):
    """Build the frozen 25-value feature vector for raw global Top-K entries."""
    if z_raw.ndim != 2:
        raise ValueError("per-image z_raw must be [Q,C]")
    num_queries, num_classes = z_raw.shape
    if not 0 < topk <= num_queries * num_classes:
        raise ValueError("UMR topk is outside the query-class candidate pool")

    boxes_xyxy = _cxcywh_to_xyxy(boxes)
    pairwise_iou = _box_iou_xyxy(boxes_abs, boxes_abs)
    hidden_norm = F.normalize(decoder_features, dim=-1)
    cx, cy, width, height = boxes.unbind(-1)
    logw = torch.log(width.clamp_min(1.0e-6))
    logh = torch.log(height.clamp_min(1.0e-6))
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    distance_to_border = torch.minimum(
        torch.minimum(x1, y1), torch.minimum(1.0 - x2, 1.0 - y2)
    )
    area = width * height
    aspect = width / height.clamp_min(1.0e-6)
    sigmoid_raw = torch.sigmoid(z_raw)
    raw_rank = _rank_matrix(z_raw)
    qccr_rank = _rank_matrix(qccr_score)

    _, top_indices = torch.topk(z_raw.flatten(), k=topk, dim=0)
    flat_index = top_indices.sort().values
    query_index = flat_index // num_classes
    class_index = flat_index % num_classes

    rival_scores = qccr_score[:, class_index].transpose(0, 1)
    selected_scores = qccr_score[query_index, class_index]
    higher = (rival_scores > selected_scores[:, None]).float()
    higher_iou = pairwise_iou[query_index] * higher
    has_higher = higher.sum(dim=1) > 0
    maximum_iou, competitor_index = higher_iou.max(dim=1)
    has_overlapping_higher = has_higher & (maximum_iou > 0)
    second_iou = higher_iou.topk(k=2, dim=1).values[:, 1]
    score_gap = (
        qccr_score[competitor_index, class_index] - selected_scores
    ) * has_overlapping_higher.float()
    competitor_cosine = (
        hidden_norm[query_index] * hidden_norm[competitor_index]
    ).sum(dim=-1) * has_overlapping_higher.float()

    mean_by_class = torch.stack([qccr_score[:, c].mean() for c in range(num_classes)])
    std_by_class = torch.stack([qccr_score[:, c].std() for c in range(num_classes)])
    max_by_class = torch.stack([qccr_score[:, c].max() for c in range(num_classes)])

    features = torch.stack(
        (
            cx[query_index], cy[query_index], logw[query_index], logh[query_index],
            (logw - logh)[query_index], z_raw[query_index, class_index],
            sigmoid_raw[query_index, class_index], qccr_logit[query_index, class_index],
            qccr_score[query_index, class_index], risk[query_index, class_index],
            raw_rank[query_index, class_index] / float(num_queries),
            qccr_rank[query_index, class_index] / float(num_queries),
            maximum_iou, second_iou, (higher_iou > 0.50).sum(1).float(),
            (higher_iou > 0.75).sum(1).float(), (higher_iou > 0.90).sum(1).float(),
            score_gap, competitor_cosine, mean_by_class[class_index],
            std_by_class[class_index], max_by_class[class_index],
            distance_to_border[query_index], area[query_index], aspect[query_index],
        ),
        dim=-1,
    ).to(dtype=z_raw.dtype)
    return flat_index, features, class_index, decoder_features[query_index]


class UMRHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        projection_dim: int = 32,
        class_embedding_dim: int = 8,
        num_classes: int = 5,
        num_features: int = 25,
    ) -> None:
        super().__init__()
        self.hidden_projection = nn.Linear(hidden_dim, projection_dim)
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.input_projection = nn.Linear(
            projection_dim + class_embedding_dim + num_features, projection_dim
        )
        self.norm1 = nn.LayerNorm(projection_dim)
        self.ffn = nn.Sequential(
            nn.Linear(projection_dim, 64), nn.SiLU(), nn.Linear(64, projection_dim)
        )
        self.norm2 = nn.LayerNorm(projection_dim)
        self.output = nn.Linear(projection_dim, 1)

    def forward(
        self, hidden: torch.Tensor, classes: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        joined = torch.cat(
            (self.hidden_projection(hidden), self.class_embedding(classes), features),
            dim=-1,
        )
        value = self.norm1(self.input_projection(joined))
        value = self.norm2(value + self.ffn(value))
        return self.output(value).squeeze(-1)


class UMR(nn.Module):
    """Apply frozen UMR bonus, recompute risk, then apply frozen QCCR."""

    def __init__(
        self,
        checkpoint: str,
        num_classes: int = 5,
        num_queries: int = 300,
        topk: int = 300,
        tau: float = 0.0,
        beta: float = 2.0,
        beta0: float = 0.0,
        beta_positive: float = 1.0,
        beta_negative: float = 1.0,
        duplicate_penalty: float = 0.0,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if not checkpoint:
            raise ValueError("UMR requires a validation-selected checkpoint")
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.topk = int(topk)
        if not 0 < self.topk <= self.num_queries * self.num_classes:
            raise ValueError("UMR topk is outside the query-class candidate pool")
        self.tau = float(tau)
        self.beta = float(beta)
        self.qccr_parameters = (
            float(beta0), float(beta_positive), float(beta_negative),
            float(duplicate_penalty),
        )
        self.head = UMRHead(num_classes=self.num_classes)
        checkpoint_path = Path(checkpoint).resolve()
        if expected_checkpoint_sha256 is not None:
            actual_checkpoint_sha256 = sha256_file(checkpoint_path)
            if actual_checkpoint_sha256 != expected_checkpoint_sha256:
                raise RuntimeError("UMR checkpoint SHA-256 mismatch")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise TypeError("UMR checkpoint must be a tensor state_dict")
        # Accept both the optimized donor names and this host's explicit names.
        rename = {
            "h_proj.": "hidden_projection.",
            "cls_emb.": "class_embedding.",
            "in_proj.": "input_projection.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
            "out.": "output.",
        }
        translated = {}
        for key, value in state.items():
            new_key = key
            for old_prefix, new_prefix in rename.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            if new_key in translated:
                raise RuntimeError(f"duplicate UMR state key after translation: {new_key}")
            translated[new_key] = value
        self.head.load_state_dict(translated, strict=True)
        self.head.eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        boxes_abs: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[1:] != (
            self.num_queries,
            self.num_classes,
        ):
            raise ValueError(
                f"expected logits [B,{self.num_queries},{self.num_classes}], "
                f"got {tuple(logits.shape)}"
            )
        if boxes.shape != (*logits.shape[:2], 4) or boxes_abs.shape != boxes.shape:
            raise ValueError("UMR boxes must be [B,Q,4]")
        if decoder_features.shape[:2] != logits.shape[:2]:
            raise ValueError("UMR decoder features must share [B,Q]")
        if decoder_features.shape[-1] != self.head.hidden_projection.in_features:
            raise ValueError("UMR decoder hidden dimension mismatch")
        devices = {logits.device, boxes.device, boxes_abs.device, decoder_features.device}
        if len(devices) != 1:
            raise ValueError("UMR inputs must share one device")

        logits = logits.float()
        boxes = boxes.float()
        boxes_abs = boxes_abs.float()
        decoder_features = decoder_features.float()
        pairwise_iou = compute_pairwise_iou(boxes.detach())
        risk0 = compute_competition_risk(
            logits.detach(), boxes.detach(), iou_matrix=pairwise_iou
        )
        qccr0 = apply_qccr(logits, risk0, *self.qccr_parameters)
        score0 = torch.sigmoid(qccr0)

        corrected = []
        for batch_index in range(logits.shape[0]):
            flat_index, features, classes, hidden = build_image_features(
                logits[batch_index], boxes[batch_index], boxes_abs[batch_index],
                decoder_features[batch_index], qccr0[batch_index], score0[batch_index],
                risk0[batch_index], self.topk,
            )
            predicted_value = torch.sigmoid(self.head(hidden, classes, features))
            bonus = torch.zeros(
                self.num_queries * self.num_classes,
                dtype=torch.float32,
                device=logits.device,
            )
            bonus[flat_index] = torch.relu(predicted_value - self.tau)
            corrected.append(
                logits[batch_index]
                + self.beta * bonus.reshape(self.num_queries, self.num_classes)
            )
        corrected_logits = torch.stack(corrected, dim=0)
        risk1 = compute_competition_risk(
            corrected_logits.detach(), boxes.detach(), iou_matrix=pairwise_iou
        )
        return apply_qccr(corrected_logits, risk1, *self.qccr_parameters)
