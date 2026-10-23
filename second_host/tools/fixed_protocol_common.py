#!/usr/bin/env python3
"""Shared validation for the versioned, predeclared QCCR--UMR protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    from fit_common import (
        adapter_root,
        canonical_data_lock,
        canonical_implementation_lock,
        require_sha,
        runtime_source_hashes,
        published_tool_sha,
        sha256_file,
    )
except ImportError:  # imported as tools.fixed_protocol_common in unit tests
    from tools.fit_common import (
        adapter_root,
        canonical_data_lock,
        canonical_implementation_lock,
        require_sha,
        runtime_source_hashes,
        published_tool_sha,
        sha256_file,
    )


LOCK_SCHEMA = "cdw1630-rtdetr-r18-qccr-umr-fixed-protocol-lock-v1"
LOCK_RELATIVE = Path("provenance/INFERENCE_FIXED_PROTOCOL_LOCK.json")
SOURCE_RELATIVE = Path("source/RT-DETR_HiLoR/rtdetr_pytorch")
POLICY_NAME = "predeclared-fixed-hyperparameters"
DONOR_POLICY_RELATIVE = Path("tools/calibration/fixed_protocol/PROTOCOL_LOCK.json")
DONOR_POLICY_SHA256 = "b4e4c1ae48762d746309b15147ef2bf3503db9e860f8d3272f925a16d3fddc27"

QCCR_LAMBDA_REG = 1.0e-3
UMR_POSITIVE_WEIGHT = 4
UMR_HEAD_SEED = 0
UMR_TAU = 0.0
UMR_BETA = 2.0
UMR_TOPK = 300
FIXED_TOOL_RELATIVES = {
    "tools/fixed_protocol_common.py",
    "tools/freeze_qccr_fixed.py",
    "tools/train_umr_fixed.py",
    "tools/freeze_inference_binding_fixed.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exact(payload: dict[str, Any], path: str, expected: Any) -> None:
    current: Any = payload
    for key in path.split("."):
        _require(isinstance(current, dict) and key in current, f"missing {path}")
        current = current[key]
    _require(
        type(current) is type(expected) and current == expected,
        f"{path} must be {expected!r}, got {current!r}",
    )


def validate_fixed_protocol_payload(payload: dict[str, Any]) -> None:
    """Reject any policy drift before a fit artifact is read."""

    _exact(payload, "schema", LOCK_SCHEMA)
    _exact(payload, "status", "fixed-before-test-access")
    _exact(payload, "host", "RT-DETR-R18")
    _exact(payload, "dataset", "CDW1630")
    _exact(payload, "selection_policy.name", POLICY_NAME)
    _exact(payload, "selection_policy.adaptive_selection_performed", False)
    _exact(payload, "selection_policy.adaptive_candidate_evidence_used", False)
    _exact(payload, "selection_policy.val_metrics_computed", False)
    _exact(payload, "selection_policy.test_accessed", False)
    _exact(payload, "qccr.lambda_reg", QCCR_LAMBDA_REG)
    _exact(payload, "qccr.fit_split", "Val")
    _exact(payload, "qccr.oof_gate_used", False)
    _exact(payload, "umr.positive_weight", UMR_POSITIVE_WEIGHT)
    _exact(payload, "umr.head_seed", UMR_HEAD_SEED)
    _exact(payload, "umr.tau", UMR_TAU)
    _exact(payload, "umr.beta", UMR_BETA)
    _exact(payload, "umr.topk", UMR_TOPK)
    _exact(payload, "umr.fit_split", "Train")
    _exact(payload, "umr.val_grid_used", False)
    _exact(payload, "test.allowed_for_fitting_or_selection", False)
    _exact(payload, "test.allowed_after_complete_freeze", "formal evaluation only")
    _exact(payload, "compatibility.qccr_decision", "OOF_VAL_GO")
    _exact(payload, "compatibility.umr_decision", "VAL_GO")
    _exact(
        payload,
        "compatibility.compatibility_fields_are_not_observed_gates",
        True,
    )
    _exact(payload, "source_policy.name", "DEIM fixed QCCR-UMR protocol")
    _exact(payload, "source_policy.relative_path", DONOR_POLICY_RELATIVE.as_posix())
    _exact(payload, "source_policy.sha256", DONOR_POLICY_SHA256)
    _exact(payload, "source_policy.inherited_without_parameter_change", True)


def _require_binding(
    payload: dict[str, Any], name: str, relative: str, digest: str
) -> None:
    binding = (payload.get("bindings") or {}).get(name)
    _require(isinstance(binding, dict), f"fixed protocol lacks {name} binding")
    _require(binding.get("path") == relative, f"fixed protocol {name} path mismatch")
    _require(binding.get("sha256") == digest, f"fixed protocol {name} SHA mismatch")


def load_fixed_protocol(
    lock_path: Path,
    expected_sha: str,
    *,
    adapter_root_path: Path | None,
    tool_relative: str,
    required_locked_tools: Iterable[str] = (),
) -> tuple[dict[str, Any], str, str, str]:
    """Load the fixed lock and validate its full implementation/SHA chain.

    Returns ``(payload, lock_sha, implementation_sha, data_lock_sha)``.
    """

    root = adapter_root(adapter_root_path)
    canonical_path = (root / LOCK_RELATIVE).resolve()
    supplied = Path(lock_path).expanduser().resolve()
    _require(
        supplied == canonical_path,
        f"fixed protocol lock must be {LOCK_RELATIVE.as_posix()}",
    )
    lock_sha = require_sha(supplied, expected_sha, "fixed inference protocol lock")
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read fixed inference protocol lock: {exc}") from exc
    _require(isinstance(payload, dict), "fixed protocol lock root must be an object")
    validate_fixed_protocol_payload(payload)

    implementation, implementation_sha = canonical_implementation_lock(root)
    _, data_lock_sha = canonical_data_lock(root)
    _require_binding(
        payload,
        "implementation_lock",
        "provenance/IMPLEMENTATION_LOCK.json",
        implementation_sha,
    )
    _require_binding(
        payload,
        "data_split_lock",
        "provenance/DATA_SPLIT_LOCK.json",
        data_lock_sha,
    )

    donor_binding = payload.get("source_policy") or {}
    donor_lock_record = donor_binding.get("optimized_donor_lock") or {}
    donor_lock_path = root / "provenance/OPTIMIZED_DONOR_LOCK.json"
    donor_lock_sha = sha256_file(donor_lock_path)
    _require(
        donor_lock_record
        == {
            "path": "provenance/OPTIMIZED_DONOR_LOCK.json",
            "sha256": donor_lock_sha,
        },
        "fixed protocol optimized-donor-lock binding mismatch",
    )
    donor_lock = json.loads(donor_lock_path.read_text(encoding="utf-8"))
    _require(
        donor_binding.get("donor_content_aggregate_sha256")
        == donor_lock.get("content_aggregate_sha256"),
        "fixed protocol donor tree aggregate mismatch",
    )
    donor_policy_candidates = (
        root / "references/OPTIMIZED_REPRO_SOURCE" / DONOR_POLICY_RELATIVE,
        root.parent / "remote_reference/OPTIMIZED_REPRO_SOURCE" / DONOR_POLICY_RELATIVE,
    )
    donor_policy_paths = [path for path in donor_policy_candidates if path.is_file()]
    _require(donor_policy_paths, "bound DEIM fixed protocol source is unavailable")
    for donor_policy_path in donor_policy_paths:
        require_sha(donor_policy_path, DONOR_POLICY_SHA256, "DEIM fixed protocol source")
    donor_policy = json.loads(donor_policy_paths[0].read_text(encoding="utf-8"))
    expected_donor_policy = {
        "dataset": "CDW1630",
        "detector_seed": 0,
        "qccr_lambda_reg": QCCR_LAMBDA_REG,
        "umr_positive_weight": UMR_POSITIVE_WEIGHT,
        "umr_head_seed": UMR_HEAD_SEED,
        "umr_tau": UMR_TAU,
        "umr_beta": UMR_BETA,
        "umr_topk": UMR_TOPK,
        "selection": {
            "detector": "Val",
            "qccr_coefficients": "new-checkpoint Val fit with fixed regularization",
            "umr_weights": "new-checkpoint Train fit",
            "inference_hyperparameters": (
                "fixed identically for detector seeds 0, 1 and 2"
            ),
            "test_used_for_parameter_selection": False,
        },
    }
    _require(
        donor_policy == expected_donor_policy,
        "DEIM fixed protocol source content is not the bound policy",
    )

    source_root = (root / SOURCE_RELATIVE).resolve()
    runtime_hashes = runtime_source_hashes(source_root, require_umr=True)
    _require(
        payload.get("fit_runtime_sources") == runtime_hashes,
        "fixed protocol runtime-source bindings differ from the adapter",
    )
    _require(
        implementation.get("fit_runtime_sources") == runtime_hashes,
        "implementation lock and fixed protocol runtime sources disagree",
    )

    locked_tools = implementation.get("tools")
    _require(isinstance(locked_tools, dict), "implementation lock lacks tool hashes")
    fixed_existing = payload.get("existing_locked_tools")
    _require(
        isinstance(fixed_existing, dict),
        "fixed protocol lacks inherited implementation-tool hashes",
    )
    for relative in ("tools/fit_common.py", *tuple(required_locked_tools)):
        expected = locked_tools.get(relative)
        _require(
            isinstance(expected, str) and len(expected) == 64,
            f"implementation lock does not bind {relative}",
        )
        _require(
            fixed_existing.get(relative) == expected,
            f"fixed protocol inherited tool binding mismatch: {relative}",
        )
        _require(
            sha256_file(root / relative) == published_tool_sha(root, relative, expected),
            f"inherited implementation tool changed: {relative}",
        )

    fixed_tools = payload.get("fixed_protocol_tools")
    _require(isinstance(fixed_tools, dict), "fixed protocol lacks fixed-tool hashes")
    _require(
        set(fixed_tools) == FIXED_TOOL_RELATIVES,
        "fixed protocol tool set is incomplete or contains an unbound tool",
    )
    _require(tool_relative in fixed_tools, f"fixed protocol does not bind {tool_relative}")
    for relative, expected in fixed_tools.items():
        _require(
            isinstance(expected, str) and len(expected) == 64,
            f"fixed protocol does not bind {relative}",
        )
        _require(
            sha256_file(root / relative) == published_tool_sha(root, relative, expected),
            f"fixed protocol tool changed: {relative}",
        )
    return payload, lock_sha, implementation_sha, data_lock_sha


def fixed_lock_record(root: Path, lock_sha: str) -> dict[str, str]:
    return {"path": LOCK_RELATIVE.as_posix(), "sha256": lock_sha}
