#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def sha256(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sample_sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5

def normalize_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise RuntimeError(f"Unsupported class-name mapping: {type(value).__name__}")

def coco_row(image_id: int, category_id: int, xyxy: list[float], score: float,
             class_count: int) -> dict[str, Any]:
    import math
    require(0 <= category_id < class_count, "Predicted class id outside frozen class space")
    require(math.isfinite(score) and 0 <= score <= 1, "Invalid score")
    require(len(xyxy) == 4 and all(math.isfinite(value) for value in xyxy), "Invalid coordinates")
    x1, y1, x2, y2 = map(float, xyxy)
    require(x2 >= x1 and y2 >= y1, "Negative prediction extent")
    return {"image_id": int(image_id), "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
