#!/usr/bin/env python3
"""Optional integrity check against the single release-wide file inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator


EXPECTED_ASSETS = [
    (1, 'main_host', 'main_host', 'core'),
    (2, 'ablations', 'ablations', 'core'),
    (3, 'btc', 'generalization/btc', 'generalization'),
    (4, 'benchmarks', 'benchmarks', 'extension'),
    (5, 'sensitivity', 'sensitivity', 'extension'),
    (6, 'numerical_evidence', 'numerical_evidence', 'extension'),
    (7, 'qualitative_evidence', 'qualitative_evidence', 'extension'),
    (8, 'gate_sensitivity', 'gate_sensitivity', 'extension'),
    (9, 'bootstrap', 'bootstrap', 'extension'),
    (10, 'dataset_analysis', 'dataset_analysis', 'extension'),
    (11, 'second_host', 'second_host', 'second_host'),
    (12, 'efficiency', 'efficiency', 'efficiency'),
    (13, 'ranking_comparison', 'ranking_comparison', 'ranking_comparison'),
    (14, 'proposal_budget', 'proposal_budget', 'offline_analysis'),
]
MANIFEST_NAME = 'RELEASE_MANIFEST.sha256.json'
INDEX_NAME = 'RELEASE_INDEX.json'
_DIGEST_CACHE = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGEST_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        _DIGEST_CACHE[key] = digest.hexdigest()
    return _DIGEST_CACHE[key]


def external_asset_records(root: Path) -> dict:
    """Weights fetched from their provider are not redistributed in archives."""
    for base in (root, *root.parents):
        catalog = base / 'EXTERNAL_ASSETS.json'
        if catalog.is_file() and (base / 'verify_release.py').is_file():
            result = {}
            for row in json.loads(catalog.read_text(encoding='utf-8'))['files']:
                relative = Path(row['path'])
                require(not relative.is_absolute() and '..' not in relative.parts,
                        'Unsafe external asset path')
                result[str(base / relative)] = row
            return result
    return {}


def checked_files(root: Path) -> Iterator[Path]:
    external = external_asset_records(root)
    for directory, dirs, files in os.walk(root, followlinks=False):
        # A Git checkout and normal Python execution create these locally;
        # neither is part of the distributed scientific assets.
        dirs[:] = sorted(name for name in dirs if name not in {'.git', '__pycache__'})
        base = Path(directory)
        for name in dirs:
            require(not (base / name).is_symlink(), 'Symlink directory: ' + str(base / name))
        for name in sorted(files):
            path = base / name
            if path.suffix == '.pyc' or name == '.git':
                continue
            require(not path.is_symlink(), 'Symlink file: ' + str(path))
            if str(path) in external:
                row = external[str(path)]
                require(path.stat().st_size == row['bytes'], 'External asset size mismatch: ' + str(path))
                require(sha256_file(path) == row['sha256'], 'External asset SHA mismatch: ' + str(path))
                continue
            yield path


def validate_layout(release: Path) -> None:
    require(release.is_dir(), 'Release directory missing: ' + str(release))
    for _, _, relative, _ in EXPECTED_ASSETS:
        require((release / relative).is_dir(), 'Release area missing: ' + relative)


def build_index(release: Path) -> dict:
    """Navigation only; file identity is defined once, in MANIFEST_NAME."""
    return dict(schema='hilordetr.release-index.v1', status='RELEASE_READY',
                layout='flat', asset_count=len(EXPECTED_ASSETS), assets=[
                    dict(order=order, name=name, relative_path=relative, scope=scope)
                    for order, name, relative, scope in EXPECTED_ASSETS])


def write_index(release: Path) -> None:
    (release / INDEX_NAME).write_text(json.dumps(build_index(release), indent=2) + '\n', encoding='utf-8')


def manifest_entries(release: Path) -> list[dict]:
    return [dict(path=path.relative_to(release).as_posix(), bytes=path.stat().st_size,
                 sha256=sha256_file(path))
            for path in sorted(checked_files(release), key=lambda p: p.relative_to(release).as_posix())
            if path.relative_to(release).as_posix() != MANIFEST_NAME]


def write_manifest(release: Path) -> None:
    entries = manifest_entries(release)
    payload = dict(schema='cdw1630.final-paper-release.sha256.v1', algorithm='sha256',
                   root='.', self_excluded=MANIFEST_NAME, file_count=len(entries),
                   total_bytes=sum(row['bytes'] for row in entries), files=entries)
    (release / MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def verify_index(release: Path) -> None:
    saved = json.loads((release / INDEX_NAME).read_text(encoding='utf-8'))
    require(saved == build_index(release), 'Release navigation index differs from the directory layout')


def verify_manifest(release: Path) -> None:
    payload = json.loads((release / MANIFEST_NAME).read_text(encoding='utf-8'))
    saved, actual = payload['files'], manifest_entries(release)
    require(payload.get('algorithm') == 'sha256', 'Unsupported checksum algorithm')
    require(payload.get('file_count') == len(actual), 'Release file count mismatch')
    require(payload.get('total_bytes') == sum(row['bytes'] for row in actual), 'Release size mismatch')
    require([row['path'] for row in saved] == [row['path'] for row in actual], 'Release file list mismatch')
    for before, after in zip(saved, actual):
        require(before == after, 'Release file changed: ' + after['path'])


def refresh(release: Path) -> None:
    """Author packaging operation; not required for training or evaluation."""
    validate_layout(release)
    write_index(release)
    write_manifest(release)
    verify(release)


def verify(release: Path) -> None:
    validate_layout(release)
    verify_index(release)
    verify_manifest(release)
    print('RELEASE VERIFICATION: PASS')
    print('Release-wide relative-path/size/SHA-256 binding: PASS')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('refresh', 'verify'))
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    refresh(release) if args.action == 'refresh' else verify(release)


if __name__ == '__main__':
    main()
