#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the portable source payload."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


INCLUDED_TOP_LEVEL = (
    "README.md", "CURRENT_STATUS.md", "RESULTS_MANIFEST.sha256",
    "requirements.txt", "protocol", "source", "tests"
)
EXCLUDED_PARTS = {"__pycache__", "state", "results", "remote_reference"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "SOURCE_MANIFEST.sha256")).resolve()
    if root not in output.parents:
        raise RuntimeError("manifest output must stay inside the study directory")

    files: list[Path] = []
    for name in INCLUDED_TOP_LEVEL:
        item = root / name
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(
                path for path in item.rglob("*")
                if path.is_file()
                and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
                and path.suffix != ".pyc"
            )
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in files
        if path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"SOURCE MANIFEST: PASS / {len(lines)} files / {output}")


if __name__ == "__main__":
    main()
