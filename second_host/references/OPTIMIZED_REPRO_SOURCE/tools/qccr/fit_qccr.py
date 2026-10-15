#!/usr/bin/env python
"""Fit four QCCR-v1 scalars using image-level five-fold Val CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit


LAMBDA_REG_VALUES = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
)

PARAMETER_NAMES = (
    "beta0",
    "beta_positive",
    "beta_negative",
    "duplicate_penalty",
)


def build_features(logits, risk):
    logits = np.asarray(logits, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)

    return np.stack(
        (
            np.ones_like(logits),
            -np.logaddexp(0.0, -logits),
            np.logaddexp(0.0, logits),
            -risk,
        ),
        axis=1,
    )


def balance_weights(targets):
    targets = np.asarray(targets, dtype=np.float64)
    count = float(targets.size)
    positive_mass = float(targets.sum())
    negative_mass = float((1.0 - targets).sum())

    if positive_mass <= 0.0:
        raise RuntimeError("fold contains no positive target mass")

    if negative_mass <= 0.0:
        raise RuntimeError("fold contains no negative target mass")

    return (
        count / (2.0 * positive_mass),
        count / (2.0 * negative_mass),
    )


def balanced_data_loss(theta, features, targets):
    positive_weight, negative_weight = balance_weights(
        targets
    )

    calibrated = features @ theta

    positive_loss = (
        positive_weight
        * targets
        * np.logaddexp(0.0, -calibrated)
    )
    negative_loss = (
        negative_weight
        * (1.0 - targets)
        * np.logaddexp(0.0, calibrated)
    )

    return float(
        np.mean(positive_loss + negative_loss)
    )


def objective_with_gradient(
    theta,
    features,
    targets,
    risk,
    lambda_reg,
):
    positive_weight, negative_weight = balance_weights(
        targets
    )

    calibrated = features @ theta
    probability = expit(calibrated)

    positive_loss = (
        positive_weight
        * targets
        * np.logaddexp(0.0, -calibrated)
    )
    negative_loss = (
        negative_weight
        * (1.0 - targets)
        * np.logaddexp(0.0, calibrated)
    )

    data_loss = np.mean(
        positive_loss + negative_loss
    )

    derivative_logit = (
        -positive_weight
        * targets
        * (1.0 - probability)
        + negative_weight
        * (1.0 - targets)
        * probability
    )

    data_gradient = (
        features.T @ derivative_logit
    ) / targets.size

    beta0, beta_positive, beta_negative, gamma = theta

    effective_penalty = gamma * risk

    regularizer = (
        beta0 ** 2
        + (beta_positive - 1.0) ** 2
        + (beta_negative - 1.0) ** 2
        + np.mean(effective_penalty ** 2)
    )

    regularizer_gradient = np.asarray(
        (
            2.0 * beta0,
            2.0 * (beta_positive - 1.0),
            2.0 * (beta_negative - 1.0),
            2.0 * gamma * np.mean(risk ** 2),
        ),
        dtype=np.float64,
    )

    total_loss = data_loss + lambda_reg * regularizer
    total_gradient = (
        data_gradient
        + lambda_reg * regularizer_gradient
    )

    return float(total_loss), total_gradient


def fit_parameters(
    features,
    targets,
    risk,
    lambda_reg,
):
    initial = np.asarray(
        (0.0, 1.0, 1.0, 0.0),
        dtype=np.float64,
    )

    bounds = (
        (-20.0, 20.0),
        (1.0e-4, 20.0),
        (1.0e-4, 20.0),
        (0.0, 20.0),
    )

    result = minimize(
        fun=objective_with_gradient,
        x0=initial,
        args=(
            features,
            targets,
            risk,
            float(lambda_reg),
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1.0e-12,
            "gtol": 1.0e-9,
            "maxls": 50,
        },
    )

    if not np.isfinite(result.x).all():
        raise RuntimeError("non-finite fitted parameters")

    if not np.isfinite(result.fun):
        raise RuntimeError("non-finite fitted objective")

    return {
        "parameters": {
            name: float(value)
            for name, value in zip(
                PARAMETER_NAMES,
                result.x,
            )
        },
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective": float(result.fun),
    }


def parameter_array(result):
    return np.asarray(
        [
            result["parameters"][name]
            for name in PARAMETER_NAMES
        ],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    target_path = Path(args.targets).resolve()
    output_path = Path(args.output).resolve()

    payload = torch.load(
        target_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get("source_split") != "val":
        raise RuntimeError("only Val targets may fit QCCR")

    tensors = payload["tensors"]
    fit_mask = tensors["fit_mask"].bool()

    logits = tensors["logits"][fit_mask].numpy()
    risk = tensors["competition_risk"][fit_mask].numpy()
    targets = tensors["targets"][fit_mask].numpy()
    image_ids = tensors["image_id"][fit_mask].numpy()

    features = build_features(logits, risk)

    unique_images = np.unique(image_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_images)

    fold_lookup = {
        int(image_id): int(position % args.folds)
        for position, image_id in enumerate(unique_images)
    }

    fold_ids = np.asarray(
        [fold_lookup[int(image_id)] for image_id in image_ids],
        dtype=np.int64,
    )

    identity = np.asarray(
        (0.0, 1.0, 1.0, 0.0),
        dtype=np.float64,
    )
    identity_loss = balanced_data_loss(
        identity,
        features,
        targets,
    )

    candidates = {}

    for lambda_reg in LAMBDA_REG_VALUES:
        key = f"{lambda_reg:.0e}"
        fold_results = []
        validation_losses = []

        for fold in range(args.folds):
            train_mask = fold_ids != fold
            validation_mask = fold_ids == fold

            fitted = fit_parameters(
                features=features[train_mask],
                targets=targets[train_mask],
                risk=risk[train_mask],
                lambda_reg=lambda_reg,
            )

            theta = parameter_array(fitted)

            train_loss = balanced_data_loss(
                theta,
                features[train_mask],
                targets[train_mask],
            )
            validation_loss = balanced_data_loss(
                theta,
                features[validation_mask],
                targets[validation_mask],
            )

            validation_losses.append(validation_loss)

            fold_results.append({
                "fold": fold,
                "train_image_count": int(
                    np.unique(image_ids[train_mask]).size
                ),
                "validation_image_count": int(
                    np.unique(image_ids[validation_mask]).size
                ),
                "train_balanced_bce": train_loss,
                "validation_balanced_bce": validation_loss,
                **fitted,
            })

        full_fit = fit_parameters(
            features=features,
            targets=targets,
            risk=risk,
            lambda_reg=lambda_reg,
        )

        full_theta = parameter_array(full_fit)
        gamma = full_theta[3]
        effective_penalty = gamma * risk

        candidates[key] = {
            "lambda_reg": lambda_reg,
            "mean_cv_balanced_bce": float(
                np.mean(validation_losses)
            ),
            "std_cv_balanced_bce": float(
                np.std(validation_losses)
            ),
            "folds": fold_results,
            "full_val_fit": {
                **full_fit,
                "balanced_bce": balanced_data_loss(
                    full_theta,
                    features,
                    targets,
                ),
                "effective_penalty_mean": float(
                    effective_penalty.mean()
                ),
                "effective_penalty_p50": float(
                    np.quantile(effective_penalty, 0.50)
                ),
                "effective_penalty_p90": float(
                    np.quantile(effective_penalty, 0.90)
                ),
                "effective_penalty_max": float(
                    effective_penalty.max()
                ),
            },
        }

    provisional_key = min(
        candidates,
        key=lambda key: candidates[key][
            "mean_cv_balanced_bce"
        ],
    )

    result = {
        "format_version": 1,
        "method": "QCCR-v1",
        "selection_status": (
            "PROVISIONAL_BY_CV_BCE;"
            "FINAL_SELECTION_REQUIRES_OOF_COCO"
        ),
        "target_file": str(target_path),
        "source_checkpoint": payload["source_checkpoint"],
        "seed": int(args.seed),
        "fold_count": int(args.folds),
        "fit_candidate_count": int(targets.size),
        "target_mass": float(targets.sum()),
        "identity_balanced_bce": identity_loss,
        "fold_by_image": {
            str(key): value
            for key, value in fold_lookup.items()
        },
        "candidates": candidates,
        "provisional_lambda_reg": candidates[
            provisional_key
        ]["lambda_reg"],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("identity balanced BCE:", identity_loss)
    print()

    for key, candidate in candidates.items():
        parameters = candidate[
            "full_val_fit"
        ]["parameters"]

        print(
            "lambda=%s cv_bce=%.9f "
            "beta0=%+.6f a=%.6f b=%.6f "
            "gamma=%.6f penalty_p90=%.6f"
            % (
                key,
                candidate["mean_cv_balanced_bce"],
                parameters["beta0"],
                parameters["beta_positive"],
                parameters["beta_negative"],
                parameters["duplicate_penalty"],
                candidate["full_val_fit"][
                    "effective_penalty_p90"
                ],
            )
        )

    print()
    print(
        "provisional lambda:",
        result["provisional_lambda_reg"],
    )
    print("fit output:", output_path)
    print("QCCR FIVE-FOLD FIT: PASS")


if __name__ == "__main__":
    main()
