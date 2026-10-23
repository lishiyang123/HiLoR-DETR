#!/usr/bin/env python3
"""Fit QCCR candidates with deterministic image-level five-fold Val CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

try:
    from fit_common import (
        atomic_json, formal_fit_guard, require_output_path, require_sha,
    )
except ImportError:  # imported as tools.fit_qccr in unit tests
    from tools.fit_common import (
        atomic_json, formal_fit_guard, require_output_path, require_sha,
    )


TARGET_SCHEMA = "rtdetr-hilor-qccr-multiiou-targets-v1"
FIT_SCHEMA = "rtdetr-hilor-qccr-fivefold-fit-v1"
LAMBDA_REG_VALUES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
PARAMETER_NAMES = (
    "beta0", "beta_positive", "beta_negative", "duplicate_penalty",
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
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise RuntimeError("fit partition lacks positive or negative target mass")
    return count / (2.0 * positive_mass), count / (2.0 * negative_mass)


def balanced_data_loss(theta, features, targets):
    positive_weight, negative_weight = balance_weights(targets)
    calibrated = features @ theta
    loss = (
        positive_weight * targets * np.logaddexp(0.0, -calibrated)
        + negative_weight * (1.0 - targets) * np.logaddexp(0.0, calibrated)
    )
    return float(np.mean(loss))


def objective_with_gradient(theta, features, targets, risk, lambda_reg):
    positive_weight, negative_weight = balance_weights(targets)
    calibrated = features @ theta
    probability = expit(calibrated)
    data_loss = np.mean(
        positive_weight * targets * np.logaddexp(0.0, -calibrated)
        + negative_weight * (1.0 - targets) * np.logaddexp(0.0, calibrated)
    )
    derivative = (
        -positive_weight * targets * (1.0 - probability)
        + negative_weight * (1.0 - targets) * probability
    )
    data_gradient = (features.T @ derivative) / targets.size
    beta0, beta_positive, beta_negative, duplicate_penalty = theta
    regularizer = (
        beta0**2
        + (beta_positive - 1.0) ** 2
        + (beta_negative - 1.0) ** 2
        + np.mean((duplicate_penalty * risk) ** 2)
    )
    regularizer_gradient = np.asarray(
        (
            2.0 * beta0,
            2.0 * (beta_positive - 1.0),
            2.0 * (beta_negative - 1.0),
            2.0 * duplicate_penalty * np.mean(risk**2),
        ),
        dtype=np.float64,
    )
    return (
        float(data_loss + lambda_reg * regularizer),
        data_gradient + lambda_reg * regularizer_gradient,
    )


def fit_parameters(features, targets, risk, lambda_reg):
    result = minimize(
        objective_with_gradient,
        x0=np.asarray((0.0, 1.0, 1.0, 0.0), dtype=np.float64),
        args=(features, targets, risk, float(lambda_reg)),
        method="L-BFGS-B",
        jac=True,
        bounds=((-20.0, 20.0), (1.0e-4, 20.0), (1.0e-4, 20.0), (0.0, 20.0)),
        options={"maxiter": 1000, "ftol": 1.0e-12, "gtol": 1.0e-9, "maxls": 50},
    )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("QCCR optimizer returned non-finite values")
    if not bool(result.success):
        raise RuntimeError(
            "QCCR optimizer did not converge successfully: "
            f"status={result.status}, message={result.message}"
        )
    if result.x[1] <= 0.0 or result.x[2] <= 0.0 or result.x[3] < 0.0:
        raise RuntimeError("QCCR optimizer violated the runtime parameter domain")
    result = {
        "parameters": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, result.x)
        },
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective": float(result.fun),
    }
    return result


def parameter_array(result):
    return np.asarray(
        [result["parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64
    )


def run_fit(payload, *, target_path, target_sha, seed=0, folds=5):
    if payload.get("schema") != TARGET_SCHEMA or payload.get("source_split") != "val":
        raise RuntimeError("QCCR fitting accepts only bound Val target payloads")
    if payload.get("test_accessed") is not False:
        raise RuntimeError("QCCR target payload does not prove Test exclusion")
    if not isinstance(payload.get("source_data_split_lock_sha256"), str):
        raise RuntimeError("QCCR targets lack canonical data-lock provenance")
    runtime_hashes = payload.get("runtime_source_sha256")
    if not isinstance(runtime_hashes, dict) or not isinstance(
        runtime_hashes.get("hilor_qccr.py"), str
    ):
        raise RuntimeError("QCCR targets lack runtime-source provenance")
    tensors = payload["tensors"]
    fit_mask = tensors["fit_mask"].bool()
    logits = tensors["logits"][fit_mask].numpy()
    risk = tensors["competition_risk"][fit_mask].numpy()
    targets = tensors["targets"][fit_mask].numpy()
    image_ids = tensors["image_id"][fit_mask].numpy()
    features = build_features(logits, risk)
    unique_images = np.unique(image_ids)
    if unique_images.size < folds:
        raise RuntimeError("not enough Val images for the requested image-level folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)
    fold_lookup = {
        int(image_id): int(position % folds)
        for position, image_id in enumerate(unique_images)
    }
    fold_ids = np.asarray([fold_lookup[int(value)] for value in image_ids])
    identity = np.asarray((0.0, 1.0, 1.0, 0.0), dtype=np.float64)
    candidates = {}
    for lambda_reg in LAMBDA_REG_VALUES:
        fold_results, validation_losses = [], []
        for fold in range(folds):
            train = fold_ids != fold
            validation = fold_ids == fold
            fitted = fit_parameters(
                features[train], targets[train], risk[train], lambda_reg
            )
            theta = parameter_array(fitted)
            validation_loss = balanced_data_loss(
                theta, features[validation], targets[validation]
            )
            validation_losses.append(validation_loss)
            fold_results.append(
                {
                    "fold": fold,
                    "train_image_count": int(np.unique(image_ids[train]).size),
                    "validation_image_count": int(np.unique(image_ids[validation]).size),
                    "train_balanced_bce": balanced_data_loss(
                        theta, features[train], targets[train]
                    ),
                    "validation_balanced_bce": validation_loss,
                    **fitted,
                }
            )
        full_fit = fit_parameters(features, targets, risk, lambda_reg)
        theta = parameter_array(full_fit)
        effective_penalty = theta[3] * risk
        key = f"{lambda_reg:.0e}"
        candidates[key] = {
            "lambda_reg": lambda_reg,
            "mean_cv_balanced_bce": float(np.mean(validation_losses)),
            "std_cv_balanced_bce": float(np.std(validation_losses)),
            "folds": fold_results,
            "full_val_fit": {
                **full_fit,
                "balanced_bce": balanced_data_loss(theta, features, targets),
                "effective_penalty_mean": float(effective_penalty.mean()),
                "effective_penalty_p90": float(np.quantile(effective_penalty, 0.90)),
                "effective_penalty_max": float(effective_penalty.max()),
            },
        }
    provisional_key = min(
        candidates, key=lambda key: candidates[key]["mean_cv_balanced_bce"]
    )
    return {
        "schema": FIT_SCHEMA,
        "format_version": 1,
        "method": "QCCR-v1",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "detector_seed": payload.get("detector_seed"),
        "implementation_lock_sha256": payload.get("implementation_lock_sha256"),
        "selection_status": "PROVISIONAL_BY_CV_BCE;FINAL_SELECTION_REQUIRES_OOF_COCO",
        "test_accessed": False,
        "target_file": str(target_path),
        "target_sha256": target_sha,
        "source_capture_sha256": payload["source_capture_sha256"],
        "source_data_split_lock_sha256": payload["source_data_split_lock_sha256"],
        "runtime_source_sha256": runtime_hashes,
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "seed": int(seed),
        "fold_count": int(folds),
        "fit_candidate_count": int(targets.size),
        "target_mass": float(targets.sum()),
        "identity_balanced_bce": balanced_data_loss(identity, features, targets),
        "fold_by_image": {str(key): value for key, value in fold_lookup.items()},
        "candidates": candidates,
        "provisional_lambda_reg": candidates[provisional_key]["lambda_reg"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--expected-target-sha", required=True)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    formal_evidence = formal_fit_guard(
        tool_relative="tools/fit_qccr.py",
        adapter_root_path=args.adapter_root,
    )
    target_path = require_output_path(
        args.targets, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root, label="QCCR target input",
    )
    target_sha = require_sha(target_path, args.expected_target_sha, "QCCR targets")
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    if payload.get("implementation_lock_sha256") != formal_evidence[
        "implementation_lock_sha256"
    ]:
        raise RuntimeError("QCCR targets belong to another implementation lock")
    if payload.get("source_checkpoint_sha256") != args.expected_checkpoint_sha:
        raise RuntimeError("QCCR targets belong to another detector")
    if payload.get("detector_seed") != args.detector_seed:
        raise RuntimeError("QCCR targets belong to another detector seed")
    result = run_fit(
        payload,
        target_path=target_path,
        target_sha=target_sha,
        seed=args.seed,
        folds=args.folds,
    )
    output = require_output_path(
        args.output, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=args.adapter_root,
        label="QCCR fit output",
    )
    atomic_json(output, result)
    print("RT-DETR QCCR FIVE-FOLD FIT: PASS")
    print("Test accessed: NO")


if __name__ == "__main__":
    main()
