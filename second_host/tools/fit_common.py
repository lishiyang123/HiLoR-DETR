#!/usr/bin/env python3
"""Shared guards for the RT-DETR QCCR--UMR fitting tools.

This module deliberately knows only the frozen Train/Val capture contract.  It
does not provide a Test path or a fallback split.
"""

from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


CAPTURE_SCHEMA = "rtdetr-hilor-frozen-query-class-capture-v1"
HOST = "RT-DETR-R18"
DATASET = "CDW1630"
EXPECTED_CLASSES = 5
EXPECTED_QUERIES = 300
EXPECTED_TOPK = 300
ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "cdw1630-rtdetr-r18-hilor-second-host-protocol-lock-v1"
DATA_LOCK_SCHEMA = "cdw1630-rtdetr-r18-hilor-data-split-lock-v1"
IMPLEMENTATION_SCHEMA = "cdw1630-rtdetr-r18-hilor-implementation-lock-v1"
FIT_RUNTIME_SOURCE_NAMES = ("hilor_qccr.py", "hilor_umr.py")
RUNTIME_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "scipy": "scipy",
    "pyyaml": "PyYAML",
    "faster_coco_eval": "faster-coco-eval",
    "numpy": "numpy",
    "pycocotools": "pycocotools",
    "pillow": "Pillow",
}
REQUIRED_RUNTIME_KEYS = ("python", *RUNTIME_DISTRIBUTIONS)
OUTPUT_NAMESPACES = {
    "inference": Path("artifacts/inference"),
    "generated_config": Path("configs/hilor/generated"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{value}\n".encode("utf-8"))
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def adapter_root(path: Path | None = None) -> Path:
    return (ADAPTER_ROOT if path is None else Path(path)).resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_output_path(
    path: Path,
    *,
    namespace: str,
    detector_seed: int,
    adapter_root_path: Path | None = None,
    label: str = "output",
) -> Path:
    """Confine generated evidence to one adapter-owned namespace."""
    if namespace not in OUTPUT_NAMESPACES:
        raise ValueError(f"unknown adapter output namespace: {namespace}")
    if int(detector_seed) not in {0, 1, 2}:
        raise RuntimeError("formal detector seed must be one of 0, 1, 2")
    root = adapter_root(adapter_root_path)
    allowed = (
        root / OUTPUT_NAMESPACES[namespace] / f"seed{int(detector_seed)}"
    ).resolve()
    resolved = Path(path).resolve()
    if resolved == allowed or not _inside(resolved, allowed):
        raise RuntimeError(
            f"{label} must be below adapter-owned "
            f"{OUTPUT_NAMESPACES[namespace] / f'seed{int(detector_seed)}'}: "
            f"{resolved}"
        )
    return resolved


def canonical_data_lock(
    adapter_root_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the data-lock payload and SHA bound by this adapter's protocol.

    Both the protocol binding and the referenced lock file are checked here so
    a copied capture cannot supply its own notion of the canonical split lock.
    """
    root = adapter_root(adapter_root_path)
    protocol_path = root / "provenance" / "PROTOCOL_LOCK.json"
    if not protocol_path.is_file():
        raise RuntimeError(f"adapter protocol lock is missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("invalid adapter protocol-lock schema")
    binding = ((protocol.get("bindings") or {}).get("data_split_lock") or {})
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("protocol lacks the canonical data-lock path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("protocol lacks the canonical data-lock SHA-256")
    lock_path = (root / relative).resolve()
    if not _inside(lock_path, root) or not lock_path.is_file():
        raise RuntimeError("protocol data-lock path escapes or is missing")
    require_sha(lock_path, expected, "canonical data-split lock")
    try:
        data_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read canonical data-split lock: {exc}") from exc
    if not isinstance(data_lock, dict) or data_lock.get("schema") != DATA_LOCK_SCHEMA:
        raise RuntimeError("invalid canonical data-split lock schema")
    return data_lock, expected


def canonical_data_lock_sha(adapter_root_path: Path | None = None) -> str:
    """Return the SHA of the protocol-bound canonical data-split lock."""

    return canonical_data_lock(adapter_root_path)[1]


def canonical_implementation_lock(
    adapter_root_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the current implementation lock through its protocol binding."""

    root = adapter_root(adapter_root_path)
    protocol_path = root / "provenance" / "PROTOCOL_LOCK.json"
    if not protocol_path.is_file():
        raise RuntimeError(f"adapter protocol lock is missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("invalid adapter protocol-lock schema")
    binding = ((protocol.get("bindings") or {}).get("implementation_lock") or {})
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("protocol lacks the canonical implementation-lock path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("protocol lacks the canonical implementation-lock SHA-256")
    lock_path = (root / relative).resolve()
    if not _inside(lock_path, root) or not lock_path.is_file():
        raise RuntimeError("protocol implementation-lock path escapes or is missing")
    require_sha(lock_path, expected, "canonical implementation lock")
    try:
        implementation = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read canonical implementation lock: {exc}") from exc
    if (
        not isinstance(implementation, dict)
        or implementation.get("schema") != IMPLEMENTATION_SCHEMA
    ):
        raise RuntimeError("invalid canonical implementation-lock schema")
    return implementation, expected


def current_runtime_versions() -> dict[str, str]:
    """Return the exact nine-package runtime identity used by formal runs."""

    versions = {"python": platform.python_version()}
    for key, distribution in RUNTIME_DISTRIBUTIONS.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "runtime dependency required by IMPLEMENTATION_LOCK is missing: "
                f"{distribution}"
            ) from exc
    return versions


def published_tool_sha(root: Path, relative: str, recorded_sha: str) -> str:
    """Check current executable files against the shipped file inventory.

    Recorded capture/protocol receipts remain immutable. Standalone fixture or
    author snapshots without an aggregate release retain their recorded checks.
    """
    manifest = root.parent / 'RELEASE_MANIFEST.sha256.json'
    if not manifest.is_file():
        return recorded_sha
    key = root.name + '/' + relative
    rows = json.loads(manifest.read_text(encoding='utf-8'))['files']
    match = next((row for row in rows if row['path'] == key), None)
    if match is None or not isinstance(match.get('sha256'), str) or len(match['sha256']) != 64:
        raise RuntimeError('Current release does not bind executable tool: ' + key)
    return match['sha256']


def runtime_source_path(source_root: Path, name: str) -> Path:
    """Resolve serialized runtime identifiers to descriptive physical files."""
    names = {'hilor_qccr.py': 'competition_mapping.py',
             'hilor_umr.py': 'candidate_value.py',
             'hilor_ufd.py': 'auxiliary_geometry.py'}
    directory = Path(source_root) / 'src/zoo/rtdetr'
    path = directory / names.get(name, name)
    return path if path.is_file() else directory / name


def formal_fit_guard(
    *,
    tool_relative: str,
    adapter_root_path: Path | None = None,
    source_root: Path | None = None,
    require_umr: bool = False,
    check_runtime: bool = True,
    runtime_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the stage tool, relevant runtime sources and fit policy.

    This deliberately hashes only the invoking tool, this shared guard, and
    the one or two HiLoR runtime files used by the stage.  The complete source
    tree remains a protocol/preflight gate rather than a repeated fit-stage
    scan.
    """

    root = adapter_root(adapter_root_path)
    implementation, implementation_sha = canonical_implementation_lock(root)

    if implementation.get("host") != HOST or implementation.get("dataset") != DATASET:
        raise RuntimeError("implementation lock belongs to another host or dataset")
    if implementation.get("public_inference_name") != "QCCR–UMR":
        raise RuntimeError("implementation lock has another public inference name")
    test_policy = implementation.get("test_policy") or {}
    if test_policy.get("selection_or_fitting_allowed") is not False:
        raise RuntimeError("implementation lock does not forbid Test fitting")

    if check_runtime:
        expected_runtime = implementation.get("runtime_environment")
        if not isinstance(expected_runtime, dict) or set(expected_runtime) != set(
            REQUIRED_RUNTIME_KEYS
        ):
            raise RuntimeError(
                "implementation runtime environment must contain exactly the nine locked keys"
            )
        actual_runtime = dict(runtime_versions or current_runtime_versions())
        if set(actual_runtime) != set(REQUIRED_RUNTIME_KEYS):
            raise RuntimeError(
                "live runtime environment did not report exactly the nine locked keys"
            )
        if actual_runtime != expected_runtime:
            mismatches = {
                key: {
                    "expected": expected_runtime.get(key),
                    "actual": actual_runtime.get(key),
                }
                for key in REQUIRED_RUNTIME_KEYS
                if actual_runtime.get(key) != expected_runtime.get(key)
            }
            raise RuntimeError(
                "runtime environment differs from IMPLEMENTATION_LOCK: "
                + json.dumps(mismatches, sort_keys=True)
            )

    tools = implementation.get("tools")
    if not isinstance(tools, dict):
        raise RuntimeError("implementation lock lacks executable-tool hashes")
    validated_tools = {}
    for relative in dict.fromkeys(("tools/fit_common.py", tool_relative)):
        expected_sha = tools.get(relative)
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError(f"implementation lock does not freeze {relative}")
        live = (root / relative).resolve()
        if not _inside(live, root) or not live.is_file():
            raise RuntimeError(f"formal fit tool is missing or escapes adapter: {relative}")
        validated_tools[relative] = require_sha(
            live, published_tool_sha(root, relative, expected_sha), f"formal fit tool {relative}"
        )

    runtime_source_evidence = None
    if source_root is not None:
        expected_sources = implementation.get("fit_runtime_sources")
        if not isinstance(expected_sources, dict) or set(expected_sources) != set(
            FIT_RUNTIME_SOURCE_NAMES
        ):
            raise RuntimeError(
                "implementation lock lacks the exact QCCR/UMR runtime-source hashes"
            )
        names = FIT_RUNTIME_SOURCE_NAMES if require_umr else FIT_RUNTIME_SOURCE_NAMES[:1]
        actual_sources = runtime_source_hashes(
            Path(source_root), require_umr=require_umr
        )
        for name in names:
            expected_sha = expected_sources.get(name)
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise RuntimeError(f"implementation lock does not freeze {name}")
            if actual_sources.get(name) != expected_sha:
                raise RuntimeError(f"formal fit runtime source differs from lock: {name}")
        runtime_source_evidence = actual_sources

    _, data_lock_sha = canonical_data_lock(root)
    if (implementation.get("provenance") or {}).get(
        "data_split_lock_sha256"
    ) != data_lock_sha:
        raise RuntimeError("implementation lock uses another canonical data lock")

    evidence = {
        "implementation_lock_sha256": implementation_sha,
        "validated_tools": validated_tools,
    }
    if runtime_source_evidence is not None:
        evidence["runtime_source_sha256"] = runtime_source_evidence
    return evidence


def validate_capture_implementation_binding(
    capture: dict[str, Any],
    *,
    split: str,
    adapter_root_path: Path | None = None,
) -> str:
    """Bind a capture's config graph and training run to the current release."""

    root = adapter_root(adapter_root_path)
    implementation, implementation_sha = canonical_implementation_lock(root)
    configs = implementation.get("capture_configs")
    if not isinstance(configs, dict) or not isinstance(configs.get(split), dict):
        raise RuntimeError(f"implementation lock lacks the {split} capture config")
    binding = configs[split]
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"implementation lock has invalid {split} capture config path")
    expected_config = (root / relative).resolve()
    if not _inside(expected_config, root) or not expected_config.is_file():
        raise RuntimeError(f"current {split} capture config is missing or escaping")
    captured_config = Path(str(capture.get("config", "")))
    if not captured_config.is_absolute():
        raise RuntimeError(f"{split} capture config path is not absolute")
    if str(captured_config.resolve()) != str(expected_config):
        raise RuntimeError(f"{split} capture config path disagrees with implementation lock")
    if capture.get("config_sha256") != binding.get("sha256"):
        raise RuntimeError(f"{split} capture config SHA-256 disagrees with implementation lock")

    locked_closure = binding.get("include_closure")
    if not isinstance(locked_closure, list) or not locked_closure:
        raise RuntimeError(f"implementation lock lacks {split} capture include closure")
    for record in locked_closure:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid {split} implementation include record")
        relative_path = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
            raise RuntimeError(f"invalid {split} implementation include record")
        live_path = (root / relative_path).resolve()
        if not _inside(live_path, root) or not live_path.is_file():
            raise RuntimeError(f"current {split} capture include is missing or escaping")
        require_sha(live_path, expected_sha, f"current {split} capture include")

    closure = capture.get("config_include_closure")
    if not isinstance(closure, dict) or not isinstance(closure.get("files"), list):
        raise RuntimeError(f"{split} capture lacks config include-closure provenance")
    captured_closure_root = Path(str(closure.get("root", "")))
    if not captured_closure_root.is_absolute():
        raise RuntimeError(f"{split} capture include-closure root is not absolute")
    if str(captured_closure_root.resolve()) != str(expected_config):
        raise RuntimeError(f"{split} capture include-closure root mismatch")
    if capture.get("config_include_closure_sha256") != canonical_json_sha256(closure):
        raise RuntimeError(f"{split} capture include-closure digest mismatch")
    normalized = []
    for record in closure["files"]:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid {split} capture include record")
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            raise RuntimeError(f"{split} capture include path is not absolute")
        path = path.resolve()
        if not _inside(path, root):
            raise RuntimeError(f"{split} capture include path escapes adapter root")
        includes = record.get("includes_in_declared_order")
        if not isinstance(includes, list):
            raise RuntimeError(f"{split} capture include record lacks declared includes")
        for include in includes:
            include_path = Path(str(include))
            if not include_path.is_absolute():
                raise RuntimeError(
                    f"{split} capture declared include path is not absolute"
                )
            include_path = include_path.resolve()
            if not _inside(include_path, root):
                raise RuntimeError(f"{split} capture declared include escapes adapter root")
        normalized.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": record.get("sha256"),
        })
    normalized.sort(key=lambda item: item["path"])
    if normalized != locked_closure:
        raise RuntimeError(
            f"{split} capture include closure disagrees with implementation lock"
        )

    training_run = capture.get("training_run")
    if not isinstance(training_run, dict):
        raise RuntimeError(f"{split} capture lacks formal training-run provenance")
    if training_run.get("implementation_lock_sha256") != implementation_sha:
        raise RuntimeError(
            f"{split} capture training run belongs to another implementation lock"
        )
    return implementation_sha


def runtime_source_hashes(source_root: Path, *, require_umr: bool) -> dict[str, str]:
    root = Path(source_root).resolve()
    runtime_dir = root / "src" / "zoo" / "rtdetr"
    names = ["hilor_qccr.py"]
    if require_umr:
        names.append("hilor_umr.py")
    hashes = {}
    for name in names:
        path = runtime_source_path(root, name)
        if not path.is_file():
            raise RuntimeError(f"required runtime source is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(finite(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def load_capture(
    path: Path,
    *,
    expected_sha: str,
    split: str,
    expected_checkpoint_sha: str | None = None,
    adapter_root_path: Path | None = None,
    detector_seed: int | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise RuntimeError("only Train and Val captures are selection-eligible")
    require_sha(path, expected_sha, f"{split} capture")
    capture = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(capture, dict) or capture.get("schema") != CAPTURE_SCHEMA:
        raise RuntimeError(f"invalid {split} capture schema")
    if capture.get("format_version") != 4:
        raise RuntimeError(f"{split} capture predates the hardened v4 contract")
    if capture.get("host") != HOST:
        raise RuntimeError(f"{split} capture belongs to another host")
    if capture.get("method") != "frozen_native_detector":
        raise RuntimeError(f"{split} capture is not a frozen native detector capture")
    if capture.get("split") != split:
        raise RuntimeError(f"capture split mismatch: expected={split}")
    if capture.get("test_accessed") is not False:
        raise RuntimeError(f"{split} capture does not prove Test exclusion")
    if capture.get("selection_eligible") is not True:
        raise RuntimeError(f"{split} capture is not selection-eligible")
    validate_capture_implementation_binding(
        capture, split=split, adapter_root_path=adapter_root_path
    )
    if not isinstance(capture.get("data_split_lock_sha256"), str):
        raise RuntimeError(f"{split} capture lacks data-lock provenance")
    data_lock, canonical_lock_sha = canonical_data_lock(adapter_root_path)
    if capture.get("data_split_lock_sha256") != canonical_lock_sha:
        raise RuntimeError(
            f"{split} capture does not use the protocol-bound data-split lock"
        )
    split_binding = (data_lock.get("splits") or {}).get(split)
    if not isinstance(split_binding, dict):
        raise RuntimeError(f"canonical data lock lacks the {split} split")
    canonical_image_tree_sha = split_binding.get("image_tree_sha256")
    if not isinstance(canonical_image_tree_sha, str) or len(
        canonical_image_tree_sha
    ) != 64:
        raise RuntimeError(f"canonical {split} split lacks image-tree provenance")
    if capture.get("image_tree_sha256") != canonical_image_tree_sha:
        raise RuntimeError(
            f"{split} capture image-tree provenance disagrees with canonical data lock"
        )
    for field in (
        "annotation_sha256",
        "image_id_set_sha256",
        "file_name_set_sha256",
    ):
        if capture.get(field) != split_binding.get(field):
            raise RuntimeError(
                f"{split} capture {field} disagrees with canonical data lock"
            )
    canonical_image_count = int(split_binding.get("image_count", -1))
    if canonical_image_count <= 0:
        raise RuntimeError(f"canonical {split} split has invalid image_count")
    if int(capture.get("image_count", -1)) != canonical_image_count:
        raise RuntimeError(
            f"{split} capture image_count disagrees with canonical data lock"
        )
    checkpoint_sha = capture.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise RuntimeError(f"{split} capture lacks checkpoint provenance")
    if expected_checkpoint_sha is not None and checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(f"{split} capture belongs to another detector")
    if detector_seed is not None:
        if int(detector_seed) not in {0, 1, 2}:
            raise RuntimeError("formal detector seed must be one of 0, 1, 2")
        checkpoint = capture.get("checkpoint")
        if not isinstance(checkpoint, str) or Path(checkpoint).parent.name != (
            f"seed{int(detector_seed)}"
        ):
            raise RuntimeError(f"{split} capture belongs to another detector seed")
    if int(capture.get("postprocessor_num_classes", -1)) != EXPECTED_CLASSES:
        raise RuntimeError(f"{split} capture class count is not {EXPECTED_CLASSES}")
    if int(capture.get("postprocessor_num_top_queries", -1)) != EXPECTED_TOPK:
        raise RuntimeError(f"{split} capture Top-K is not {EXPECTED_TOPK}")

    records = capture.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{split} capture has no records")
    image_ids = [int(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"{split} capture contains duplicate image ids")
    if len(image_ids) != canonical_image_count:
        raise RuntimeError(
            f"{split} capture record count disagrees with canonical image_count"
        )
    actual_image_id_sha = sha256_lines(sorted(image_ids))
    if actual_image_id_sha != split_binding.get("image_id_set_sha256"):
        raise RuntimeError(
            f"{split} capture record image ids disagree with canonical data lock"
        )
    if actual_image_id_sha != capture.get("image_id_set_sha256"):
        raise RuntimeError(f"{split} capture image-id provenance disagrees with records")
    summary = capture.get("summary") or {}
    if int(summary.get("image_count", -1)) != len(records):
        raise RuntimeError(f"{split} capture image-count summary mismatch")
    if summary.get("image_tree_sha256") != canonical_image_tree_sha:
        raise RuntimeError(f"{split} capture image-tree summary mismatch")
    query_counts = summary.get("query_counts")
    if query_counts != [EXPECTED_QUERIES]:
        raise RuntimeError(f"{split} capture query geometry is not locked")
    resolved_config_sha = capture.get("resolved_config_sha256")
    if not isinstance(resolved_config_sha, str) or len(resolved_config_sha) != 64:
        raise RuntimeError(f"{split} capture lacks resolved-config provenance")
    if summary.get("resolved_config_sha256") != resolved_config_sha:
        raise RuntimeError(f"{split} capture resolved-config provenance disagrees")
    expected_tie_policy = "record_native_torch_topk_exact_boundary_ties"
    if summary.get("topk_boundary_tie_policy") != expected_tie_policy:
        raise RuntimeError(f"{split} capture has an unknown Top-K tie policy")
    if int(summary.get("topk_boundary_audit_count", -1)) != len(records):
        raise RuntimeError(f"{split} capture Top-K audit coverage mismatch")
    tie_count = sum(
        bool((record.get("topk_boundary_audit") or {}).get("boundary_tie"))
        for record in records
    )
    if int(summary.get("topk_boundary_tie_count", -1)) != tie_count:
        raise RuntimeError(f"{split} capture Top-K tie-count summary mismatch")

    for record in records:
        logits = record.get("pred_logits")
        boxes = record.get("pred_boxes")
        hidden = record.get("hL")
        if not torch.is_tensor(logits) or tuple(logits.shape) != (
            EXPECTED_QUERIES,
            EXPECTED_CLASSES,
        ):
            raise RuntimeError(f"{split} record has invalid logits geometry")
        if not torch.is_tensor(boxes) or tuple(boxes.shape) != (
            EXPECTED_QUERIES,
            4,
        ):
            raise RuntimeError(f"{split} record has invalid box geometry")
        if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[0] != EXPECTED_QUERIES:
            raise RuntimeError(f"{split} record has invalid decoder features")
        if tuple(record["post_query_idx"].shape) != (EXPECTED_TOPK,):
            raise RuntimeError(f"{split} record has invalid native Top-K indices")
        if tuple(record["post_labels"].shape) != (EXPECTED_TOPK,):
            raise RuntimeError(f"{split} record has invalid native Top-K labels")
        boundary = record.get("topk_boundary_audit")
        if (
            not isinstance(boundary, dict)
            or not isinstance(boundary.get("boundary_tie"), bool)
            or boundary.get("policy") != expected_tie_policy
        ):
            raise RuntimeError(f"{split} record lacks a valid Top-K boundary audit")
    return capture


def validate_annotation(path: Path, expected_sha: str, capture: dict[str, Any]) -> str:
    actual = require_sha(path, expected_sha, "Val annotation")
    if actual != capture.get("annotation_sha256"):
        raise RuntimeError("Val annotation and capture provenance disagree")
    if str(path.resolve()) != str(Path(capture["annotation"]).resolve()):
        raise RuntimeError("Val annotation path is not the capture-bound path")
    return actual


def install_source_root(source_root: Path, *, require_umr: bool):
    root = source_root.resolve()
    qccr_path = runtime_source_path(root, "hilor_qccr.py")
    if not qccr_path.is_file():
        raise RuntimeError(f"invalid RT-DETR source root: missing {qccr_path}")
    if not require_umr:
        spec = importlib.util.spec_from_file_location(
            "_rtdetr_fit_hilor_qccr", qccr_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load QCCR runtime source: {qccr_path}")
        qccr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qccr)
        return qccr.apply_qccr, qccr.compute_competition_risk, None, None

    umr_path = runtime_source_path(root, "hilor_umr.py")
    if not umr_path.is_file():
        raise RuntimeError(f"invalid RT-DETR source root: missing {umr_path}")
    sys.path.insert(0, str(root))
    from src.zoo.rtdetr.hilor_qccr import (  # pylint: disable=import-outside-toplevel
        apply_qccr,
        compute_competition_risk,
    )
    from src.zoo.rtdetr.hilor_umr import (  # pylint: disable=import-outside-toplevel
        UMRHead,
        build_image_features,
    )
    return apply_qccr, compute_competition_risk, UMRHead, build_image_features


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1.0e-9)


def xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ]


def exact_metrics(evaluation: dict[str, Any], num_classes: int = EXPECTED_CLASSES):
    precision = np.asarray(evaluation["precision"], dtype=np.float64)
    recall = np.asarray(evaluation["recall"], dtype=np.float64)
    params = evaluation["params"]
    ious = np.asarray(params.iouThrs)
    areas = list(params.areaRngLbl)
    max_dets = list(params.maxDets)
    categories = [int(value) for value in params.catIds]
    if categories != list(range(num_classes)) or precision.shape[2] != num_classes:
        raise RuntimeError("COCO evaluation category IDs are not contiguous 0..C-1")

    def mean_valid(value):
        array = np.asarray(value, dtype=np.float64)
        array = array[array > -1]
        return float(array.mean()) if array.size else float("nan")

    def threshold(value):
        found = np.where(np.isclose(ious, value))[0]
        if len(found) != 1:
            raise RuntimeError(f"missing IoU threshold {value}")
        return int(found[0])

    def ap(iou=None, area="all", max_det=100):
        selected = precision[:, :, :, areas.index(area), max_dets.index(max_det)]
        if iou is not None:
            selected = precision[
                threshold(iou), :, :, areas.index(area), max_dets.index(max_det)
            ]
        return mean_valid(selected)

    def ar(area="all", max_det=100):
        return mean_valid(recall[:, :, areas.index(area), max_dets.index(max_det)])

    result = {
        "AP": ap(), "AP50": ap(0.50), "AP75": ap(0.75),
        "APs": ap(area="small"), "APm": ap(area="medium"),
        "APl": ap(area="large"), "AR1": ar(max_det=1),
        "AR10": ar(max_det=10), "AR100": ar(max_det=100),
        "ARs": ar(area="small"), "ARm": ar(area="medium"),
        "ARl": ar(area="large"),
    }
    for threshold_value in (0.80, 0.85, 0.90, 0.95):
        result[f"AP{int(threshold_value * 100)}"] = ap(threshold_value)
    return result


def run_coco(annotation: Path, detections: list[dict[str, Any]], image_ids: list[int]):
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as error:
        raise RuntimeError(
            "faster-coco-eval is required for formal Val selection"
        ) from error
    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(
        coco_gt,
        iouType="bbox",
        print_function=lambda *_args, **_kwargs: None,
        separate_eval=True,
    )
    evaluator.cocoDt = coco_dt
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator.eval
