#!/usr/bin/env python3
"""Optional integrity check against the single release-wide file inventory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Iterator


EXPECTED_ASSETS = [
    (1, 'main_host', 'main_host', 'core'),
    (2, 'btc', 'generalization/btc', 'generalization'),
    (3, 'benchmarks', 'benchmarks', 'extension'),
    (4, 'sensitivity', 'sensitivity', 'extension'),
    (5, 'numerical_evidence', 'numerical_evidence', 'extension'),
    (6, 'bootstrap', 'bootstrap', 'extension'),
    (7, 'dataset_analysis', 'dataset_analysis', 'extension'),
    (8, 'second_host', 'second_host', 'second_host'),
    (9, 'efficiency', 'efficiency', 'efficiency'),
    (10, 'ranking_comparison', 'ranking_comparison', 'ranking_comparison'),
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


def checked_files(root: Path, *, verify_external: bool = True) -> Iterator[Path]:
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
                if verify_external:
                    require(path.stat().st_size == row['bytes'], 'External asset size mismatch: ' + str(path))
                    require(sha256_file(path) == row['sha256'], 'External asset SHA mismatch: ' + str(path))
                continue
            yield path


def missing_input_message(release: Path, relatives, limit: int | None = 20) -> str:
    """Explain missing inputs using the same groups as the download builder."""
    spec = importlib.util.spec_from_file_location(
        '_hilor_download_groups', Path(__file__).with_name('prepare_downloads.py'))
    downloads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloads)
    names = sorted(set(relatives))
    groups = {}
    for name in names:
        groups.setdefault(downloads.group_for(name), []).append(name)
    lines = ['Missing required files (%d) under %s:' % (len(names), release)]
    for group in downloads.GROUPS:
        members = groups.get(group, [])
        if not members:
            continue
        lines.append('  [%s] %d file(s)' % (group, len(members)))
        preview = members if limit is None else members[:limit]
        lines.extend('    ' + name for name in preview)
        if len(preview) < len(members):
            lines.append('    ... %d more; use --all-missing for the complete list.' %
                         (len(members) - len(preview)))
    lines.extend([
        'Extract the required archives into the same release directory.',
        'See DISTRIBUTION.md#downloads-by-task for the required weight parts and inputs.',
    ])
    return '\n'.join(lines)


def require_inputs(release: Path, paths) -> None:
    """Check input presence before any hashing or loading of scientific assets."""
    missing = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = release / path
        try:
            relative = path.relative_to(release).as_posix()
        except ValueError:
            raise RuntimeError('Input path is outside the release: ' + str(path)) from None
        if not path.is_file():
            missing.append(relative)
    if missing:
        raise RuntimeError(missing_input_message(release, missing, limit=None))


def validate_layout(release: Path) -> None:
    require(release.is_dir(), 'Release directory missing: ' + str(release))
    for _, _, relative, _ in EXPECTED_ASSETS:
        require((release / relative).is_dir(), 'Release area missing: ' + relative)


def build_index(release: Path) -> dict:
    """Navigation only; file identity is defined once, in MANIFEST_NAME."""
    return dict(schema='hilordetr.release-index.v1',
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


def verify_manifest(release: Path, *, missing_limit: int | None = 20) -> None:
    payload = json.loads((release / MANIFEST_NAME).read_text(encoding='utf-8'))
    saved = payload['files']
    require(payload.get('algorithm') == 'sha256', 'Unsupported checksum algorithm')
    expected = {row['path'] for row in saved}
    require(len(expected) == len(saved), 'Duplicate paths in the release inventory')
    actual_paths = {path.relative_to(release).as_posix()
                    for path in checked_files(release, verify_external=False)
                    if path.relative_to(release).as_posix() != MANIFEST_NAME}
    missing, unexpected = sorted(expected - actual_paths), sorted(actual_paths - expected)
    problems = []
    if missing:
        problems.append(missing_input_message(release, missing, missing_limit))
    if unexpected:
        preview = unexpected if missing_limit is None else unexpected[:missing_limit]
        lines = ['Unexpected release files (%d):' % len(unexpected)]
        lines.extend('  ' + name for name in preview)
        if len(preview) < len(unexpected):
            lines.append('  ... %d more; use --all-missing for the complete list.' %
                         (len(unexpected) - len(preview)))
        problems.append('\n'.join(lines))
    if problems:
        raise RuntimeError('\n'.join(problems))
    actual = manifest_entries(release)
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


def verify(release: Path, *, missing_limit: int | None = 20) -> None:
    require(release.is_dir(), 'Release directory missing: ' + str(release))
    require_inputs(release, [release / MANIFEST_NAME])
    verify_manifest(release, missing_limit=missing_limit)
    validate_layout(release)
    verify_index(release)
    print('RELEASE VERIFICATION: PASS')
    print('Release-wide relative-path/size/SHA-256 binding: PASS')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('refresh', 'verify'))
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--all-missing', action='store_true',
                        help='Print every missing/unexpected path rather than a grouped preview')
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    refresh(release) if args.action == 'refresh' else verify(
        release, missing_limit=None if args.all_missing else 20)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print('RELEASE VERIFICATION FAILED:\n' + str(error), file=sys.stderr)
        raise SystemExit(2)
