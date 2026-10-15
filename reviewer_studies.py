#!/usr/bin/env python3
"""Relocation-safe entry to existing results and optional offline replays.

No action launches training or inference. Replays require a fresh output path
outside this frozen release, and call the original evaluators unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NN = ROOT / 'ranking_comparison'
BUDGET = ROOT / 'proposal_budget'
PRIMARY = ROOT / 'main_host'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_bound_inputs():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.sha256.json').read_text())
    saved = {row['path']: row for row in manifest['files']}
    inputs = [PRIMARY / 'datasets/CDW1630/annotations/instances_test2017.json']
    binding = json.loads((BUDGET / 'manifests/INPUT_BINDINGS.json').read_text())
    for row in binding['predictions'].values():
        path = ROOT / row['prediction']['path']
        if sha(path) != row['prediction']['sha256']:
            raise RuntimeError('Historical prediction binding mismatch: ' + str(path))
        inputs.append(path)
    inputs += list((NN / 'state/captures').glob('*.pt'))
    for path in inputs:
        rel = path.relative_to(ROOT).as_posix()
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


def budget_replay(output):
    sys.path.insert(0, str(BUDGET))
    from evaluate import run_evaluation
    protocol = BUDGET / 'protocol/ACTION_BUDGET_PROTOCOL_LOCK.json'
    predictions = ROOT / 'dataset_analysis/conditional_performance/DENSITY_OVERLAP_STRATIFIED_EVAL_v1/predictions'
    report = run_evaluation(PRIMARY / 'datasets/CDW1630/annotations/instances_test2017.json',
                            predictions, output, sha(protocol))
    recorded = json.loads((BUDGET / 'results/ACTION_BUDGET_RESULTS.json').read_text())
    for field in ('per_seed_rows', 'three_seed_summary_rows'):
        if report[field] != recorded[field]:
            raise RuntimeError('Offline replay differs: ' + field)
    print('BUDGET RESULT REPLAY: EXACT MATCH (120 per-seed rows)')


def nearest_replay(output):
    calibration = PRIMARY / 'checkpoints/CDW1630/Ours/seed0/calibration'
    command = [sys.executable, '-B', str(NN / 'source/run_frozen_comparison.py'),
               '--capture', str(NN / 'state/captures/gccs_seed0_test_capture.pt'),
               '--annotation', str(PRIMARY / 'datasets/CDW1630/annotations/instances_test2017.json'),
               '--split', 'test', '--output-root', str(output),
               '--linear-iou-threshold', '0.95', '--gaussian-sigma', '2.0',
               '--gilg-reference-params', str(NN / 'state/fitted/gilg_original_seed0.json'),
               '--gilg-complete-pool-params', str(NN / 'state/fitted/gilg_complete_pool_seed0.json'),
               '--formal-source', str(PRIMARY / 'source/FINAL_CODE'),
               '--qccr-params', str(calibration / 'qccr_params.json'),
               '--umr-checkpoint', str(calibration / 'umr_mlp_head_seed0_pw4.pt'),
               '--umr-metadata', str(calibration / 'umr_mlp_head_seed0_pw4.metadata.json'),
               '--formal-replay-device', 'cpu',
               '--iou-reg-checkpoint', str(NN / 'state/fitted/iou_regression/iou_regressor_seed123.pt'),
               '--iou-reg-metadata', str(NN / 'state/fitted/iou_regression/iou_regressor_seed123.metadata.json'),
               '--iou-reg-device', 'cpu']
    subprocess.run(command, cwd=NN, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'check-inputs', 'budget-replay', 'nearest-replay'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'list':
        for folder in ('second_host', 'efficiency',
                       'ranking_comparison', 'proposal_budget'):
            print(folder + '/RELEASE_ENTRY.md')
    elif args.action == 'check-inputs':
        check_bound_inputs()
    else:
        check_bound_inputs()
        output = new_output(args.output)
        if args.action == 'budget-replay':
            budget_replay(output)
        else:
            nearest_replay(output)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    main()
