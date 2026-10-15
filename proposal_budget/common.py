#!/usr/bin/env python3
"""Dependency-light helpers for the sorting-proposal budget evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


SOURCE_PACKAGE = "proposal_budget"
STATE_PACKAGE = "CDW_SORTING_PROPOSAL_BUDGET_EVAL_STATE_v1"
RELEASE_NAME = "HiLoR-DETR"
MAIN_PACKAGE = "main_host"
DATASET_PACKAGE = "dataset_analysis"
PREDICTION_CHILD = "DENSITY_OVERLAP_STRATIFIED_EVAL_v1"
METHODS = ("B0", "Full")
SEEDS = (0, 1, 2)
BUDGETS = tuple(range(1, 11))
IOU_THRESHOLDS = (0.75, 0.80)
MAIN_BUDGET = 5
MAIN_IOU = 0.75
EXPECTED_TEST_IMAGES = 237
EXPECTED_CANDIDATES_PER_IMAGE = 300


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), "Missing JSON: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: Any) -> None:
    _atomic_replace(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_text(path: Path, value: str) -> None:
    _atomic_replace(path, (value.rstrip() + "\n").encode("utf-8"))


def file_record(path: Path, relative_to: Path = None) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_file_record(record: Dict[str, Any], base: Path = None) -> Path:
    path = Path(str(record["path"]))
    if base is not None and not path.is_absolute():
        path = base / path
    require(path.is_file(), "Bound file is missing: %s" % path)
    require(path.stat().st_size == int(record["bytes"]), "Bound file size changed: %s" % path)
    require(sha256(path) == record["sha256"], "Bound file SHA changed: %s" % path)
    return path


def manifest_index(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("files")
    require(isinstance(rows, list), "Unsupported manifest schema: %s" % path)
    return {str(row["path"]): row for row in rows}


def verify_manifest_member(manifest: Path, root: Path, member: Path) -> None:
    index = manifest_index(manifest)
    relative = member.resolve().relative_to(root.resolve()).as_posix()
    row = index.get(relative)
    require(row is not None, "File is not bound by manifest: %s" % relative)
    size = row.get("bytes", row.get("size", -1))
    require(int(size) == member.stat().st_size, "Manifest size mismatch: %s" % relative)
    require(row.get("sha256") == sha256(member), "Manifest SHA mismatch: %s" % relative)


def package_rows(root: Path, manifest_relative: str) -> List[Dict[str, Any]]:
    manifest = root / manifest_relative
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == manifest or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def rebuild_manifest(root: Path, manifest_relative: str = "manifests/PACKAGE_MANIFEST.sha256.json") -> int:
    rows = package_rows(root, manifest_relative)
    write_json(root / manifest_relative, {
        "algorithm": "sha256",
        "format_version": 1,
        "package": root.name,
        "files": rows,
    })
    return len(rows)


def verify_manifest(root: Path, manifest_relative: str = "manifests/PACKAGE_MANIFEST.sha256.json") -> int:
    payload = read_json(root / manifest_relative)
    expected = {row["path"]: row for row in payload.get("files", [])}
    observed = {row["path"]: row for row in package_rows(root, manifest_relative)}
    require(
        set(expected) == set(observed),
        "Package manifest path mismatch; missing=%r extra=%r"
        % (sorted(set(expected) - set(observed))[:5], sorted(set(observed) - set(expected))[:5]),
    )
    for relative, row in expected.items():
        current = observed[relative]
        require(int(row.get("bytes", -1)) == current["bytes"], "Manifest byte mismatch: %s" % relative)
        require(row.get("sha256") == current["sha256"], "Manifest SHA mismatch: %s" % relative)
    return len(expected)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    require(bool(rows), "Cannot average an empty sequence")
    return sum(rows) / float(len(rows))


def sample_std(values: Iterable[float]) -> float:
    rows = list(values)
    require(bool(rows), "Cannot compute standard deviation of an empty sequence")
    if len(rows) == 1:
        return 0.0
    center = mean(rows)
    return (sum((value - center) ** 2 for value in rows) / float(len(rows) - 1)) ** 0.5
