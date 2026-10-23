#!/usr/bin/env python3
"""Relocation-safe entry to existing results and optional offline replays.

No action launches detector training or inference. Candidate replays use CUDA
FP32 postprocessing and require a new output path outside the release.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

from verify_release import require_inputs

ROOT = Path(__file__).resolve().parent
NN = ROOT / 'ranking_comparison'
PRIMARY = ROOT / 'main_host'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_bound_inputs():
    require_inputs(ROOT, [ROOT / 'RELEASE_MANIFEST.sha256.json'])
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.sha256.json').read_text())
    saved = {row['path']: row for row in manifest['files']}
    inputs = [PRIMARY / 'datasets/CDW1630/annotations/instances_test2017.json']
    inputs += [ROOT / name for name in saved
               if name.startswith('ranking_comparison/state/captures/') and name.endswith('.pt')]
    require_inputs(ROOT, inputs)
    for path in inputs:
        rel = path.relative_to(ROOT).as_posix()
        if rel not in saved:
            raise RuntimeError('Input is not registered in the release inventory: ' + rel)
        row = saved[rel]
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Release input changed: ' + rel)
    print('PORTABLE STUDY INPUTS: PASS (%d files)' % len(inputs))


def new_output(value):
    if value is None:
        raise ValueError('--output is required for replay')
    output = value.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Write replay outputs outside the frozen release')
    if output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(output))
    return output


def nearest_replay(output):
    command = [sys.executable, '-B', str(NN / 'run.py'),
               '--stage', 'replay', '--output', str(output)]
    subprocess.run(command, cwd=NN, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'check-inputs', 'nearest-replay'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'list':
        for folder in ('second_host', 'efficiency',
                       'ranking_comparison'):
            print(folder + '/README.md')
    elif args.action == 'check-inputs':
        check_bound_inputs()
    else:
        check_bound_inputs()
        output = new_output(args.output)
        nearest_replay(output)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print('STUDY COMMAND FAILED:\n' + str(error), file=sys.stderr)
        raise SystemExit(2)
