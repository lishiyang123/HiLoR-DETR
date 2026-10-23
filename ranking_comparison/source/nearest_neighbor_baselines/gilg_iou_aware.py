"""Independent implementation of Gilg et al. IoU-aware calibration.

The conditioning variable is the minimum Jaccard distance to any higher-score
candidate of the same class.  The calibration map follows the conditionally
dependent multivariate Beta model used by the authors' reference pipeline.

No source from the authors' repository or ``netcal`` is copied here.  The
formula is implemented directly from the published method and public API
documentation so its inputs, parameters, and serialization remain explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .candidate_adapter import CandidateSet, pairwise_iou, stable_descending_order


def minimum_jaccard_to_higher(candidates: CandidateSet) -> np.ndarray:
    """Compute ``min(1-IoU)`` over higher-score same-class candidates.

    The leading candidate of every class receives 1.0.  Exact score ties are
    resolved by immutable candidate id, making repeated runs deterministic.
    """

    result = np.ones(len(candidates.scores), dtype=np.float64)
    for label in np.unique(candidates.labels):
        indices = np.flatnonzero(candidates.labels == label)
        order = stable_descending_order(
            candidates.scores[indices], candidates.candidate_ids[indices]
        )
        ranked = indices[order]
        iou = pairwise_iou(candidates.boxes_xyxy[ranked], candidates.boxes_xyxy[ranked])
        for rank in range(1, len(ranked)):
            result[ranked[rank]] = 1.0 - float(np.max(iou[rank, :rank]))
    return np.clip(result, 0.0, 1.0)


def official_near_duplicate_keep(
    candidates: CandidateSet,
    iou_threshold: float | None = 0.99,
) -> np.ndarray:
    """Reproduce the reference pipeline's optional IoU=0.99 prefilter.

    Passing ``None`` keeps the complete QxC pool.  The switch is deliberately
    exposed because the paper describes a no-NMS mapping while the released
    experiment configuration performs this near-identical-box cleanup.
    """

    keep = np.ones(len(candidates.scores), dtype=bool)
    if iou_threshold is None:
        return keep
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1] or None")

    keep[:] = False
    for label in np.unique(candidates.labels):
        indices = np.flatnonzero(candidates.labels == label)
        order = stable_descending_order(
            candidates.scores[indices], candidates.candidate_ids[indices]
        )
        remaining = indices[order]
        while len(remaining) > 0:
            best = remaining[0]
            keep[best] = True
            if len(remaining) == 1:
                break
            overlaps = pairwise_iou(
                candidates.boxes_xyxy[best : best + 1],
                candidates.boxes_xyxy[remaining[1:]],
            )[0]
            remaining = remaining[1:][overlaps <= iou_threshold]
    return keep


def greedy_tp_labels(
    candidates: CandidateSet,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """COCO-style score-ordered, one-to-one labels for non-crowd CDW1630 GT."""

    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    if len(gt_boxes) != len(gt_labels):
        raise ValueError("GT boxes and labels have inconsistent lengths")
    active = (
        np.ones(len(candidates.scores), dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if active.shape != candidates.scores.shape:
        raise ValueError("active_mask has the wrong shape")

    labels = np.zeros(len(candidates.scores), dtype=np.float64)
    for category in np.unique(candidates.labels):
        det_indices = np.flatnonzero(active & (candidates.labels == category))
        gt_indices = np.flatnonzero(gt_labels == category)
        if len(det_indices) == 0 or len(gt_indices) == 0:
            continue
        det_order = stable_descending_order(
            candidates.scores[det_indices], candidates.candidate_ids[det_indices]
        )
        det_indices = det_indices[det_order]
        overlaps = pairwise_iou(
            candidates.boxes_xyxy[det_indices], gt_boxes[gt_indices]
        )
        unmatched = np.ones(len(gt_indices), dtype=bool)
        for local_det, candidate_index in enumerate(det_indices):
            possible = np.flatnonzero(unmatched)
            if len(possible) == 0:
                break
            local_gt = possible[int(np.argmax(overlaps[local_det, possible]))]
            if overlaps[local_det, local_gt] >= iou_threshold:
                labels[candidate_index] = 1.0
                unmatched[local_gt] = False
    return labels


def _dependent_beta_logits_numpy(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: float,
    epsilon: float,
) -> np.ndarray:
    inputs = np.clip(np.asarray(inputs, dtype=np.float64), epsilon, 1.0 - epsilon)
    odds = inputs / np.clip(1.0 - inputs, epsilon, 1.0)
    features = inputs.shape[1]
    width = features + 1
    alpha_pos = np.clip(weights[:width], epsilon, None)
    alpha_neg = np.clip(weights[width : 2 * width], epsilon, None)
    beta_pos = np.clip(weights[2 * width : 3 * width], epsilon, None)
    beta_neg = np.clip(weights[3 * width :], epsilon, None)

    lambda_pos = beta_pos[1:] / beta_pos[0]
    lambda_neg = beta_neg[1:] / beta_neg[0]
    upper = np.sum(
        alpha_pos[1:] * np.log(lambda_pos)
        - alpha_neg[1:] * np.log(lambda_neg)
        + (alpha_pos[1:] - alpha_neg[1:]) * np.log(odds),
        axis=1,
    )
    lower = (
        np.sum(alpha_neg) * np.log1p(np.sum(lambda_neg * odds, axis=1))
        - np.sum(alpha_pos) * np.log1p(np.sum(lambda_pos * odds, axis=1))
    )
    return float(bias) + upper + lower


@dataclass
class ConditionalBetaCalibrator:
    """Conditionally dependent two-variate Beta calibrator."""

    epsilon: float = 1.0e-12
    weights: np.ndarray | None = None
    bias: float | None = None
    fit_summary: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        seed: int = 123,
        maxiter: int = 1000,
    ) -> "ConditionalBetaCalibrator":
        import torch
        import torch.nn.functional as functional

        values = np.asarray(inputs, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64).reshape(-1)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("inputs must have shape [N, K] with K >= 2")
        if len(values) != len(labels):
            raise ValueError("inputs and targets have inconsistent lengths")
        if set(np.unique(labels)).difference({0.0, 1.0}):
            raise ValueError("targets must be binary")
        if len(np.unique(labels)) != 2:
            raise ValueError("both target classes are required")

        clipped = np.clip(values, self.epsilon, 1.0 - self.epsilon)
        odds = clipped / np.clip(1.0 - clipped, self.epsilon, 1.0)
        odds_tensor = torch.as_tensor(odds, dtype=torch.float64)
        target_tensor = torch.as_tensor(labels, dtype=torch.float64)
        feature_count = values.shape[1]
        width = feature_count + 1
        weight_count = 4 * width

        rng = np.random.default_rng(seed)
        initial = np.concatenate(
            (
                rng.uniform(1.0 + self.epsilon, 2.0, size=weight_count),
                np.asarray([self.epsilon], dtype=np.float64),
            )
        )
        bounds = [(self.epsilon, None)] * weight_count + [(None, None)]

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            tensor = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            weights = tensor[:weight_count]
            bias = tensor[-1]
            alpha_pos = weights[:width]
            alpha_neg = weights[width : 2 * width]
            beta_pos = weights[2 * width : 3 * width]
            beta_neg = weights[3 * width :]
            lambda_pos = beta_pos[1:] / beta_pos[0]
            lambda_neg = beta_neg[1:] / beta_neg[0]
            upper = torch.sum(
                alpha_pos[1:] * torch.log(lambda_pos)
                - alpha_neg[1:] * torch.log(lambda_neg)
                + (alpha_pos[1:] - alpha_neg[1:]) * torch.log(odds_tensor),
                dim=1,
            )
            lower = (
                torch.sum(alpha_neg)
                * torch.log1p(torch.sum(lambda_neg * odds_tensor, dim=1))
                - torch.sum(alpha_pos)
                * torch.log1p(torch.sum(lambda_pos * odds_tensor, dim=1))
            )
            logits = bias + upper + lower
            loss = functional.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            return float(loss.detach().cpu()), tensor.grad.detach().cpu().numpy()

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("conditional Beta optimization returned non-finite parameters")
        self.weights = np.asarray(result.x[:weight_count], dtype=np.float64)
        self.bias = float(result.x[-1])
        self.fit_summary = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", -1)),
            "function_evaluations": int(getattr(result, "nfev", -1)),
            "final_nll": float(result.fun),
            "sample_count": int(len(values)),
            "positive_count": int(np.sum(labels)),
            "feature_count": int(feature_count),
            "seed": int(seed),
        }
        return self

    def decision_function(self, inputs: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("calibrator is not fitted")
        return _dependent_beta_logits_numpy(
            inputs, self.weights, self.bias, self.epsilon
        )

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.decision_function(inputs)
        positive = logits >= 0
        output = np.empty_like(logits, dtype=np.float64)
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        output[~positive] = exp_logits / (1.0 + exp_logits)
        return output

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None or self.bias is None:
            raise RuntimeError("calibrator is not fitted")
        return {
            "format_version": 1,
            "model": "conditionally_dependent_beta",
            "epsilon": float(self.epsilon),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "fit_summary": self.fit_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConditionalBetaCalibrator":
        if payload.get("model") != "conditionally_dependent_beta":
            raise ValueError("unsupported calibration model")
        return cls(
            epsilon=float(payload["epsilon"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            fit_summary=dict(payload.get("fit_summary", {})),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConditionalBetaCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def gilg_scores(
    candidates: CandidateSet,
    calibrator: ConditionalBetaCalibrator,
    *,
    near_duplicate_iou: float | None = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jaccard_min = minimum_jaccard_to_higher(candidates)
    inputs = np.column_stack((candidates.scores, jaccard_min))
    calibrated = calibrator.transform(inputs)
    active = official_near_duplicate_keep(candidates, near_duplicate_iou)
    return calibrated, active, jaccard_min
