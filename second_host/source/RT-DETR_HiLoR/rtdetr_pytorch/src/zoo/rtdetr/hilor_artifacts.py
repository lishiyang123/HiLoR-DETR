"""Load checkpoint-specific RT-DETR mapping and candidate-value assets.

Inference checks the files it consumes and their detector/fitting identities.
Source-code inventories and development protocol locks are not runtime inputs;
the release manifest provides an optional whole-package integrity check.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


SCHEMA = "cdw1630-rtdetr-r18-qccr-umr-inference-binding-v1"
FEATURE_SCHEMA = "umr-25d-rtdetr-final-hidden-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
QCCR_PARAMETERS = (
    "beta0",
    "beta_positive",
    "beta_negative",
    "duplicate_penalty",
)
UMR_SELECTION_PROTOCOL = "CDW1630 Train candidates / Val-only selection"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value, label):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 string")
    return value


def _resolve_bound_file(manifest_path: Path, entry: dict, label: str) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} path is missing")
    path = Path(relative)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} file is missing: {path}")
    expected = _require_sha(entry.get("sha256"), f"{label} sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return path


def _adapter_root() -> Path:
    # .../adapter/source/RT-DETR_HiLoR/rtdetr_pytorch/src/zoo/rtdetr/file.py
    return Path(__file__).resolve().parents[6]


def _resolve_manifest(manifest) -> tuple[Path, str]:
    expected = None
    if isinstance(manifest, dict):
        path_value = manifest.get("path")
        expected = _require_sha(
            manifest.get("sha256"), "inference binding sha256"
        )
        path = Path(path_value) if isinstance(path_value, str) else None
        if path is None or not str(path):
            raise RuntimeError("inference binding path is missing")
        if not path.is_absolute():
            if manifest.get("path_base") != "adapter_root":
                raise RuntimeError(
                    "relative inference binding requires path_base=adapter_root"
                )
            path = _adapter_root() / path
    else:
        path = Path(manifest)
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"inference binding is missing: {manifest_path}")
    actual = sha256_file(manifest_path)
    if expected is not None and actual != expected:
        raise RuntimeError("inference binding SHA-256 mismatch")
    return manifest_path, actual


def load_bound_inference_artifact(manifest, *, require_umr: bool):
    manifest_path, manifest_sha = _resolve_manifest(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("invalid QCCR–UMR inference binding schema")
    if payload.get("format_version") != 2:
        raise RuntimeError("inference binding predates the hardened v2 contract")
    if payload.get("host") != "RT-DETR-R18" or payload.get("dataset") != "CDW1630":
        raise RuntimeError("QCCR–UMR artifact belongs to another host or dataset")
    if payload.get("test_accessed_for_fitting_or_selection") is not False:
        raise RuntimeError("QCCR–UMR binding does not prove Test exclusion")
    expected_mode = "qccr-umr" if require_umr else "qccr"
    if payload.get("mode") != expected_mode:
        raise RuntimeError(f"inference binding mode is not {expected_mode}")
    expected_splits = ["train", "val"] if require_umr else ["val"]
    if payload.get("fit_and_selection_splits") != expected_splits:
        raise RuntimeError("QCCR–UMR binding has an invalid fitting boundary")
    detector_seed = payload.get("detector_seed")
    if detector_seed not in {0, 1, 2}:
        raise RuntimeError("inference binding has an invalid detector seed")
    detector_sha = _require_sha(
        payload.get("detector_checkpoint_sha256"), "detector checkpoint sha256"
    )
    captures = payload.get("captures")
    if not isinstance(captures, dict):
        raise RuntimeError("capture provenance is missing")
    for split in (("train", "val") if require_umr else ("val",)):
        entry = captures.get(split)
        if not isinstance(entry, dict):
            raise RuntimeError(f"{split} capture provenance is missing")
        _require_sha(entry.get("sha256"), f"{split} capture sha256")
        if entry.get("detector_checkpoint_sha256") != detector_sha:
            raise RuntimeError(f"{split} capture belongs to another checkpoint")
    data_lock_sha = _require_sha(
        captures["val"].get("data_split_lock_sha256"),
        "Val capture data-lock sha256",
    )
    if payload.get("source_data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("binding data-lock provenance disagrees with Val capture")
    if payload.get("source_val_capture_sha256") != captures["val"]["sha256"]:
        raise RuntimeError("binding Val-capture provenance disagrees")
    if require_umr and captures["train"].get("data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("Train and Val captures use different data-split locks")

    qccr_entry = payload.get("qccr_artifact")
    if not isinstance(qccr_entry, dict):
        raise RuntimeError("QCCR artifact binding is missing")
    qccr_path = _resolve_bound_file(manifest_path, qccr_entry, "QCCR artifact")
    qccr = json.loads(qccr_path.read_text(encoding="utf-8"))
    if qccr.get("method") != "QCCR-v1" or qccr.get("dataset") != "CDW1630":
        raise RuntimeError("invalid QCCR artifact identity")
    if qccr.get("source_checkpoint_sha256") != detector_sha:
        raise RuntimeError("QCCR artifact belongs to another detector checkpoint")
    if qccr.get("detector_seed") != detector_seed:
        raise RuntimeError("QCCR artifact belongs to another detector seed")
    if qccr.get("test_used_for_selection") is not False:
        raise RuntimeError("QCCR artifact does not prove Val-only selection")
    if qccr.get("source_capture_sha256") != captures["val"]["sha256"]:
        raise RuntimeError("QCCR artifact was not fitted from the bound Val capture")
    if qccr.get("source_data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("QCCR artifact uses another data-split lock")
    if qccr_entry.get("source_capture_sha256") != captures["val"]["sha256"]:
        raise RuntimeError("QCCR binding entry has another Val capture")
    if qccr_entry.get("source_data_split_lock_sha256") != data_lock_sha:
        raise RuntimeError("QCCR binding entry has another data-split lock")
    parameters = qccr.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("QCCR parameters are missing")
    qccr_parameters = {}
    for name in QCCR_PARAMETERS:
        value = float(parameters[name])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite QCCR parameter: {name}")
        qccr_parameters[name] = value
    if qccr_parameters["beta_positive"] <= 0.0:
        raise RuntimeError("beta_positive must be positive")
    if qccr_parameters["beta_negative"] <= 0.0:
        raise RuntimeError("beta_negative must be positive")
    if qccr_parameters["duplicate_penalty"] < 0.0:
        raise RuntimeError("duplicate_penalty must be non-negative")

    result = {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "detector_checkpoint_sha256": detector_sha,
        "detector_seed": detector_seed,
        "qccr_parameters": qccr_parameters,
    }
    if require_umr:
        umr = payload.get("umr")
        if not isinstance(umr, dict):
            raise RuntimeError("UMR binding is missing")
        if umr.get("feature_schema") != FEATURE_SCHEMA:
            raise RuntimeError("UMR feature schema mismatch")
        if int(umr.get("num_classes", -1)) != 5 or int(umr.get("num_queries", -1)) != 300:
            raise RuntimeError("UMR host dimensions mismatch")
        if int(umr.get("topk", -1)) != 300:
            raise RuntimeError("UMR final Top-k must be 300")
        checkpoint_path = _resolve_bound_file(manifest_path, umr, "UMR checkpoint")
        selection_entry = {
            "path": umr.get("selection_path"),
            "sha256": umr.get("selection_sha256"),
        }
        selection_path = _resolve_bound_file(
            manifest_path, selection_entry, "UMR Val selection"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema") != "rtdetr-hilor-umr-val-selection-v1":
            raise RuntimeError("invalid UMR Val-selection schema")
        if selection.get("decision") != "VAL_GO":
            raise RuntimeError("UMR selection is not VAL_GO")
        if (
            selection.get("test_accessed") is not False
            or selection.get("official_test_accessed") is not False
        ):
            raise RuntimeError("UMR selection does not prove Test exclusion")
        if selection.get("protocol") != UMR_SELECTION_PROTOCOL:
            raise RuntimeError("UMR selection uses another Train/Val protocol")
        selected_model = selection.get("selected_model") or {}
        if selected_model.get("sha256") != umr["sha256"]:
            raise RuntimeError("UMR selection and checkpoint SHA disagree")
        if selection.get("source_checkpoint_sha256") != detector_sha:
            raise RuntimeError("UMR selection belongs to another detector")
        if selection.get("detector_seed") != detector_seed:
            raise RuntimeError("UMR selection belongs to another detector seed")
        if selection.get("train_capture_sha256") != captures["train"]["sha256"]:
            raise RuntimeError("UMR selection belongs to another Train capture")
        if selection.get("val_capture_sha256") != captures["val"]["sha256"]:
            raise RuntimeError("UMR selection belongs to another Val capture")
        if selection.get("source_data_split_lock_sha256") != data_lock_sha:
            raise RuntimeError("UMR selection uses another data-split lock")
        if selection.get("qccr_artifact_sha256") != qccr_entry["sha256"]:
            raise RuntimeError("UMR selection belongs to another QCCR artifact")
        if selection.get("qccr_source_capture_sha256") != captures["val"]["sha256"]:
            raise RuntimeError("UMR selection uses another QCCR Val capture")
        if selection.get("feature_schema") != FEATURE_SCHEMA:
            raise RuntimeError("UMR selection feature schema mismatch")
        selected = selection.get("selected") or {}
        tau, beta = float(umr["tau"]), float(umr["beta"])
        if not math.isfinite(tau) or not math.isfinite(beta) or beta <= 0.0:
            raise RuntimeError("UMR tau must be finite and beta must be finite and positive")
        if float(selected.get("tau")) != tau:
            raise RuntimeError("UMR selection and bound tau disagree")
        if float(selected.get("beta")) != beta:
            raise RuntimeError("UMR selection and bound beta disagree")
        if umr.get("source_val_capture_sha256") != captures["val"]["sha256"]:
            raise RuntimeError("UMR binding entry has another Val capture")
        result["umr"] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": umr["sha256"],
            "selection": str(selection_path),
            "selection_sha256": umr["selection_sha256"],
            "topk": int(umr["topk"]),
            "tau": float(umr["tau"]),
            "beta": float(umr["beta"]),
        }
    return result
