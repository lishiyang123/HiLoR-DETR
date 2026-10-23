"""
UMR (Unique-Match Marginal Ranking) inference-only fusion.

This module implements the frozen UMR inference logic documented by the
Train/Val-only calibration workflow shipped under tools/calibration/.

  1.  baseline QCCR on the raw ``pred_logits`` (z_raw) and ``pred_boxes``;
  2.  25-dim per-candidate features via ``build_image_features``;
  3.  M300 = native global Top-300 over the 300 x C raw logits;
  4.  independent per-candidate MLP ``UMRHead``;
  5.  frozen, eval-mode value head fitted on Train, with its epoch
     selected using an internal Train holdout;
  6.  ``m_hat = sigmoid(UMRHead(...))``;
  7.  ``v = M300 * relu(m_hat - tau)``;
  8.  ``z_umr = z_raw + beta * v``;
  9.  re-compute competition risk on ``z_umr`` then apply QCCR params;
 10.  the resulting logits are sigmoid-ed and Top-300 selected by the
     PostProcessor (native path).

The order is intentionally: *baseline QCCR -> features -> correct raw logits
-> recompute QCCR*.  We never add a bonus to the final QCCR logits directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qccr import apply_qccr, compute_competition_risk, compute_pairwise_iou

__all__ = ["UMR", "UMRHead", "build_image_features", "global_query_class_mask"]

NUM_CLASSES = 5
NUM_QUERIES = 300


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two box sets (same function as the audit)."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def rank_matrix(scores_2d: torch.Tensor) -> torch.Tensor:
    """0-based global rank of each flattened entry (0 = highest)."""
    flat = scores_2d.flatten()
    order = torch.argsort(flat, descending=True, stable=True)
    ranks = torch.zeros(flat.numel(), dtype=torch.float32, device=scores_2d.device)
    ranks[order] = torch.arange(
        flat.numel(), dtype=torch.float32, device=scores_2d.device
    )
    return ranks.reshape(scores_2d.shape)


def global_query_class_mask(logits: torch.Tensor, K: int) -> torch.Tensor:
    """Bool mask [Q, C] selecting the global Top-K flat entries of ``logits``."""
    flat = logits.flatten()
    _, top_indices = torch.topk(flat, k=K, dim=0)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=logits.device)
    mask[top_indices] = True
    return mask.reshape(logits.shape)


def build_image_features(
    z_raw: torch.Tensor,
    boxes: torch.Tensor,
    boxes_abs: torch.Tensor,
    hL: torch.Tensor,
    qccr_logit: torch.Tensor,
    qccr_score: torch.Tensor,
    risk: torch.Tensor,
    K: int = NUM_QUERIES,
):
    """Return (flat_idx [K], feats [K,25], cls [K], hL_sel [K,256]) for M300.

    Verbatim reproduction of ``umr_common.build_image_features`` (25 numeric
    dims).  ``boxes`` are normalized cxcywh; ``boxes_abs`` are absolute xyxy
    pixel boxes used only for the scale-invariant pairwise IoU matrix ``Qmat``.
    """
    C = z_raw.shape[-1]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    Qmat = box_iou_xyxy(boxes_abs, boxes_abs)  # [Q, Q]
    hL_norm = F.normalize(hL, dim=-1)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    logw = torch.log(w.clamp(min=1e-6))
    logh = torch.log(h.clamp(min=1e-6))
    logwh = logw - logh
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    dist_b = torch.minimum(
        torch.minimum(x1, y1), torch.minimum(1.0 - x2, 1.0 - y2)
    )
    area = w * h
    aspect = w / h.clamp(min=1e-6)

    sigmoid_raw = torch.sigmoid(z_raw)
    raw_rank = rank_matrix(z_raw)
    qccr_rank = rank_matrix(qccr_score)

    # Select the exact audit Top-K first.  Only those K query-class pairs are
    # consumed by UMRHead, so avoid materialising five full Qx25 feature
    # tables and five full QxQ higher-score masks.
    _, top_indices = torch.topk(z_raw.flatten(), k=K, dim=0)
    flat_idx = top_indices.sort().values
    q_idx = flat_idx // C
    c_idx = flat_idx % C
    # [K,Q]: for each selected (query,class), compare against all rival
    # queries of that candidate's own class.  This is the same set of rows
    # formerly extracted from five QxQ class matrices, with no nonzero()
    # synchronization and no dynamic class groups.
    rival_scores = qccr_score[:, c_idx].transpose(0, 1)
    selected_scores = qccr_score[q_idx, c_idx]
    H = (rival_scores > selected_scores[:, None]).float()
    Qh = Qmat[q_idx] * H
    has_higher = H.sum(dim=1) > 0
    max_iou, comp_idx = Qh.max(dim=1)
    # Competitor-dependent features are defined only for a genuinely
    # overlapping, higher-score candidate.  When Qh is all zero,
    # argmax has no semantic meaning and must not leak query 0.
    has_overlapping_higher = has_higher & (max_iou > 0)
    top2 = Qh.topk(k=2, dim=1).values
    second_iou = top2[:, 1]
    cnt50 = (Qh > 0.50).sum(dim=1).float()
    cnt75 = (Qh > 0.75).sum(dim=1).float()
    cnt90 = (Qh > 0.90).sum(dim=1).float()
    score_gap = (
        qccr_score[comp_idx, c_idx] - selected_scores
    ) * has_overlapping_higher.float()
    cos_comp = (
        hL_norm[q_idx] * hL_norm[comp_idx]
    ).sum(dim=-1) * has_overlapping_higher.float()

    # Keep the original per-column reduction calls so their arithmetic order
    # is unchanged, then gather the statistic of each selected class.
    s_mean_by_class = torch.stack([
        qccr_score[:, c].mean() for c in range(C)
    ])
    s_std_by_class = torch.stack([
        qccr_score[:, c].std() for c in range(C)
    ])
    s_max_by_class = torch.stack([
        qccr_score[:, c].max() for c in range(C)
    ])

    feats = torch.stack((
        cx[q_idx], cy[q_idx], logw[q_idx], logh[q_idx], logwh[q_idx],
        z_raw[q_idx, c_idx], sigmoid_raw[q_idx, c_idx],
        qccr_logit[q_idx, c_idx], qccr_score[q_idx, c_idx],
        risk[q_idx, c_idx],
        raw_rank[q_idx, c_idx] / float(NUM_QUERIES),
        qccr_rank[q_idx, c_idx] / float(NUM_QUERIES),
        max_iou, second_iou, cnt50, cnt75, cnt90,
        score_gap, cos_comp,
        s_mean_by_class[c_idx], s_std_by_class[c_idx],
        s_max_by_class[c_idx],
        dist_b[q_idx], area[q_idx], aspect[q_idx],
    ), dim=-1).to(dtype=z_raw.dtype)

    return flat_idx, feats, c_idx, hL[q_idx]


class UMRHead(nn.Module):
    def __init__(self, hidden_dim=256, proj_dim=32, cls_emb_dim=8,
                 num_classes=5, num_feats=25):
        super().__init__()
        self.h_proj = nn.Linear(hidden_dim, proj_dim)
        self.cls_emb = nn.Embedding(num_classes, cls_emb_dim)
        self.in_proj = nn.Linear(proj_dim + cls_emb_dim + num_feats, proj_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.SiLU(), nn.Linear(64, proj_dim)
        )
        self.ln2 = nn.LayerNorm(proj_dim)
        self.out = nn.Linear(proj_dim, 1)

    def forward(self, hL, cls, feats):
        hidden = self.h_proj(hL)
        class_embedding = self.cls_emb(cls)
        joined = torch.cat([hidden, class_embedding, feats], dim=-1)
        x = self.ln1(self.in_proj(joined))
        x = self.ln2(x + self.ffn(x))
        return self.out(x).squeeze(-1)




class UMR(nn.Module):
    """Frozen UMR fusion wrapper.

    Inputs (per batch): ``z_raw`` [B,Q,C], ``boxes`` [B,Q,4] normalized cxcywh,
    ``boxes_abs`` [B,Q,4] absolute xyxy pixels, ``hL`` [B,Q,256] last decoder
    layer features.  Returns the final QCCR-calibrated logits [B,Q,C] ready for
    sigmoid + native global Top-300 selection.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        topk=300,
        tau=0.0,
        beta=2.0,
        beta0=0.0,
        beta_positive=1.0,
        beta_negative=1.0,
        duplicate_penalty=0.0,
        checkpoint=None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.topk = int(topk)
        self.tau = float(tau)
        self.beta = float(beta)
        self.beta0 = float(beta0)
        self.beta_positive = float(beta_positive)
        self.beta_negative = float(beta_negative)
        self.duplicate_penalty = float(duplicate_penalty)

        self.head = UMRHead(
            hidden_dim=256,
            proj_dim=32,
            cls_emb_dim=8,
            num_classes=self.num_classes,
            num_feats=25,
        )
        if checkpoint is None:
            raise ValueError("UMR requires a checkpoint (umr_checkpoint)")
        # The release artifact is a plain tensor state_dict. Restricting the
        # loader prevents arbitrary pickle objects from being constructed.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise TypeError("UMR checkpoint must be a tensor state_dict")
        self.head.load_state_dict(state)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, z_raw, boxes, boxes_abs, hL):
        if z_raw.ndim != 3:
            raise ValueError(
                f"z_raw must have shape [B,Q,C], got {tuple(z_raw.shape)}"
            )
        if boxes.shape != (*z_raw.shape[:2], 4):
            raise ValueError(
                "boxes must have shape [B,Q,4] matching z_raw, got "
                f"{tuple(boxes.shape)}"
            )
        if boxes_abs.shape != boxes.shape:
            raise ValueError(
                "boxes_abs must match boxes, got "
                f"{tuple(boxes_abs.shape)} and {tuple(boxes.shape)}"
            )
        if hL.ndim != 3 or hL.shape[:2] != z_raw.shape[:2]:
            raise ValueError(
                "hL must have shape [B,Q,D] matching z_raw, got "
                f"{tuple(hL.shape)}"
            )
        expected_hidden = self.head.h_proj.in_features
        if hL.shape[-1] != expected_hidden:
            raise ValueError(
                f"expected hL dimension {expected_hidden}, got {hL.shape[-1]}"
            )
        devices = {z_raw.device, boxes.device, boxes_abs.device, hL.device}
        if len(devices) != 1:
            raise ValueError(
                "z_raw, boxes, boxes_abs and hL must share one device, got "
                + ", ".join(str(device) for device in devices)
            )
        if z_raw.shape[1] != self.num_queries:
            raise ValueError(
                f"expected {self.num_queries} queries, got {z_raw.shape[1]}"
            )
        if z_raw.shape[2] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} classes, got {z_raw.shape[2]}"
            )
        if not 0 < self.topk <= self.num_queries * self.num_classes:
            raise ValueError(
                "umr_topk must be in [1, num_queries * num_classes], got "
                f"{self.topk}"
            )

        z_raw = z_raw.float()
        boxes = boxes.float()
        boxes_abs = boxes_abs.float()
        hL = hL.float()

        # 1. Baseline QCCR (features are built from the *raw* z_raw QCCR).
        # The boxes do not change between the two QCCR passes. UMR's
        # pixel-space feature IoU is deliberately kept separate: scaling
        # can change floating-point rounding even though IoU is invariant
        # in exact arithmetic.
        normalized_iou = compute_pairwise_iou(boxes.detach())
        risk0 = compute_competition_risk(
            z_raw.detach(), boxes.detach(), iou_matrix=normalized_iou
        )
        qccr_logit0 = apply_qccr(
            z_raw, risk0,
            self.beta0, self.beta_positive, self.beta_negative,
            self.duplicate_penalty,
        )
        qccr_score0 = torch.sigmoid(qccr_logit0)

        B = z_raw.shape[0]
        z_umr_list = []
        for b in range(B):
            flat_idx, feats, cls, hL_sel = build_image_features(
                z_raw[b], boxes[b], boxes_abs[b], hL[b],
                qccr_logit0[b], qccr_score0[b], risk0[b],
                K=self.topk,
            )
            m_hat = torch.sigmoid(
                self.head(hL_sel.unsqueeze(0), cls.unsqueeze(0), feats.unsqueeze(0))
            ).squeeze(0)  # [K]

            # Scatter only the K selected bonuses.  This is algebraically the
            # same M300 * relu(m_hat_grid - tau) operation without building
            # two full intermediate grids.
            bonus = torch.zeros(
                self.num_queries * self.num_classes,
                device=z_raw.device, dtype=torch.float32,
            )
            bonus[flat_idx] = torch.relu(m_hat - self.tau)
            bonus = bonus.reshape(self.num_queries, self.num_classes)
            z_umr_list.append(z_raw[b] + self.beta * bonus)

        z_umr = torch.stack(z_umr_list, dim=0)

        # 2. Recompute competition risk on z_umr, then re-apply QCCR.
        risk1 = compute_competition_risk(
            z_umr.detach(), boxes.detach(), iou_matrix=normalized_iou
        )
        final_logits = apply_qccr(
            z_umr, risk1,
            self.beta0, self.beta_positive, self.beta_negative,
            self.duplicate_penalty,
        )
        return final_logits
