#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


PACKAGE_NAME = "CDW_B0_QCCR_UMR_CONTROL_v1"
STATE_NAME = "CDW_B0_QCCR_UMR_CONTROL_STATE_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), "Missing JSON: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def yaml_path(path):
    return json.dumps(Path(path).resolve().as_posix())


def package_root():
    return Path(__file__).resolve().parent


def state_root(project):
    return Path(project).resolve() / STATE_NAME
