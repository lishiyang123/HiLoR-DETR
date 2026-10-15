#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path, algorithm: str = "sha256") -> str:
    require(path.is_file(), f"Missing file: {path}")
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha256(path: Path) -> str:
    return digest(path, "sha256")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise RuntimeError(f"Unsupported class-name schema: {type(value).__name__}")


def coco_row(image_id: int, category_id: int, coordinates: list[float], score: float,
             class_count: int) -> dict[str, Any]:
    import math
    require(0 <= category_id < class_count, "Predicted class id is outside the frozen class space")
    require(math.isfinite(score) and 0.0 <= score <= 1.0, "Invalid detection score")
    require(len(coordinates) == 4 and all(math.isfinite(float(value)) for value in coordinates),
            "Invalid detection coordinates")
    x1, y1, x2, y2 = (float(value) for value in coordinates)
    require(x2 >= x1 and y2 >= y1, "Negative prediction box extent")
    return {"image_id": int(image_id), "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}


def selected_files(root: Path, names: Iterable[str]) -> dict[str, Any]:
    rows = []
    for name in names:
        path = root / name
        rows.append({"name": name, "state": "PRESENT" if path.is_file() else "ABSENT",
                     "bytes": path.stat().st_size if path.is_file() else None,
                     "sha256": sha256(path) if path.is_file() else None})
    return {"root": str(root), "rows": rows, "tree_sha256": canonical_digest(rows)}


def manifest_text(root: Path, names: Iterable[str]) -> str:
    return "".join(f"{sha256(root / name)}  {name}\n" for name in names)
