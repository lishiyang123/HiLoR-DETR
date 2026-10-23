#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".json-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def fps_cv(repeat_means_ms: list[float]) -> float:
    rates = [1000.0 / value for value in repeat_means_ms]
    if len(rates) < 2 or statistics.mean(rates) == 0:
        return 0.0
    return 100.0 * statistics.stdev(rates) / statistics.mean(rates)


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def floating_tree_to_fp32(value):
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: floating_tree_to_fp32(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(floating_tree_to_fp32(item) for item in value)
    if isinstance(value, list):
        return [floating_tree_to_fp32(item) for item in value]
    return value

