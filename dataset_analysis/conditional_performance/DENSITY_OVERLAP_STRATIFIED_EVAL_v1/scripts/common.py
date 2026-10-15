#!/usr/bin/env python3
"""Shared, Python-3.8-compatible helpers for the stratified analysis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


PACKAGE_NAME = "CDW_DENSITY_OVERLAP_STRATIFIED_ANALYSIS_v1"
STATE_NAME = "CDW_DENSITY_OVERLAP_STRATIFIED_ANALYSIS_STATE_v1"
CHILD_NAME = "DENSITY_OVERLAP_STRATIFIED_EVAL_v1"
SEEDS = (0, 1, 2)
METHODS = ("B0", "Full")
METRICS = ("AP", "AP50", "AP75", "AR100")


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
    _atomic_replace(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def write_text(path: Path, value: str) -> None:
    _atomic_replace(path, (value.rstrip() + "\n").encode("utf-8"))


def file_record(path: Path, relative_to: Path = None) -> Dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(relative_to.resolve()).as_posix() if relative_to else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_file_record(record: Dict[str, Any], base: Path = None) -> Path:
    path = Path(record["path"])
    if base is not None and not path.is_absolute():
        path = base / path
    require(path.is_file(), "Bound file is missing: %s" % path)
    require(path.stat().st_size == int(record["bytes"]), "Bound file size changed: %s" % path)
    require(sha256(path) == record["sha256"], "Bound file SHA changed: %s" % path)
    return path


def package_rows(root: Path, manifest_name: str = "manifest.json") -> List[Dict[str, Any]]:
    rows = []
    manifest = root / manifest_name
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == manifest or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def rebuild_manifest(root: Path, manifest_name: str = "manifest.json") -> int:
    rows = package_rows(root, manifest_name)
    write_json(root / manifest_name, {
        "algorithm": "sha256",
        "format_version": 1,
        "package": root.name,
        "files": rows,
    })
    return len(rows)


def verify_manifest(root: Path, manifest_name: str = "manifest.json") -> int:
    payload = read_json(root / manifest_name)
    expected = {row["path"]: row for row in payload.get("files", [])}
    observed = {row["path"]: row for row in package_rows(root, manifest_name)}
    require(
        set(expected) == set(observed),
        "Manifest path-set mismatch; missing=%r extra=%r"
        % (sorted(set(expected) - set(observed))[:5], sorted(set(observed) - set(expected))[:5]),
    )
    for relative, row in expected.items():
        now = observed[relative]
        require(
            int(row.get("bytes", -1)) == now["bytes"] and row.get("sha256") == now["sha256"],
            "Manifest mismatch: %s" % relative,
        )
    return len(expected)


def manifest_index(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("files")
    require(isinstance(rows, list), "Unsupported manifest schema: %s" % path)
    return {str(row["path"]): row for row in rows}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    require(bool(values), "Cannot compute mean of an empty sequence")
    return sum(values) / float(len(values))


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    require(bool(values), "Cannot compute standard deviation of an empty sequence")
    if len(values) == 1:
        return 0.0
    center = mean(values)
    return (sum((value - center) ** 2 for value in values) / float(len(values) - 1)) ** 0.5

