#!/usr/bin/env python3
"""Run the unchanged measurement stages in a fresh external working copy."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
RELEASE = PACKAGE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['check', 'smoke', 'inference', 'latency-smoke', 'latency', 'training-smoke', 'training'])
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--env-root', type=Path)
    args = parser.parse_args()
    work = args.workdir.resolve()
    if work.exists() or work == RELEASE or RELEASE in work.parents:
        raise ValueError('Choose a fresh working directory outside this frozen release')
    work.mkdir(parents=True)
    for name in ['scripts', 'vendor', 'configs/templates']:
        shutil.copytree(PACKAGE / name, work / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['run_scoped.py', 'SCOPED_BINDINGS.json', 'SCOPE.json', 'RESULTS.json']:
        shutil.copy2(PACKAGE / name, work / name)
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.env_root:
        env['EFFICIENCY_ENV_ROOT'] = str(args.env_root.resolve())
    subprocess.run([sys.executable, '-B', str(work / 'run_scoped.py'),
                    '--stage', args.stage, '--asset-root', str(RELEASE),
                    '--output', str(work / 'measured_outputs')],
                   cwd=work, env=env, check=True)


if __name__ == '__main__':
    main()
