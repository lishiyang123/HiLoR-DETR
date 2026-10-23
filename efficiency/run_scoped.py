#!/usr/bin/env python3
"""Explicit, sequential stages; no automatic training after an inference run."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'scripts'))
from scoped_common import bindings, refresh_manifest, roots, write_json


def short_tmp_root(project=None):
    # multiprocessing appends /pymp-xxxxxxxx/listener-xxxxxxxx. Linux AF_UNIX
    # socket addresses cannot live under our long package/output directory.
    default = roots(project)['PROJECT_ROOT'].parent / '.eff_tmp'
    path = Path(os.environ.get('EFFICIENCY_TMP_ROOT', default)).resolve()
    if len(os.fsencode(str(path))) > 60:
        raise ValueError('EFFICIENCY_TMP_ROOT must be a short data-disk path (<=60 bytes) for AF_UNIX sockets')
    return path


def export_inference_tables(output, bound):
    """Export separate complexity and latency tables; never mix archived timings."""
    retained = json.loads((PACKAGE / 'RESULTS.json').read_text())['rows']
    complexity = {row['Model']: dict(row) for row in retained}
    latency = []
    for name in dict.fromkeys(bound['complexity_models'] + bound['latency_models']):
        path = output / name / 'MEASUREMENT.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result['status'] != 'MEASURED':
            raise RuntimeError('Nonformal/unstable measurements cannot enter the formal table')
        if name == 'deim':
            old = next(row for row in retained if row['Model'] == 'DEIM')
            if abs(old['Params_M'] - result['parameters']['total_M']) > 1e-12:
                raise RuntimeError('DEIM deployment parameter count differs from the retained row')
        if name in bound['complexity_models']:
            complexity[result['model']] = {'Model': result['model'], 'Params_M': result['parameters']['total_M'],
                                          'Detector_GFLOPs': result['flops']['forward_GFLOPs']}
        if name in bound['latency_models'] and 'latency' in result:
            latency.append({'Model': result['model'], 'Forward_ms': result['latency']['forward_ms'],
                            'Postprocess_ms': result['latency']['postprocess_ms'],
                            'Total_ms': result['latency']['total_ms']})
    for filename, rows in [('COMPLEXITY_TABLE.csv', list(complexity.values())), ('LATENCY_TABLE.csv', latency)]:
        if rows:
            with (output / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['check', 'smoke', 'complexity', 'inference', 'latency-smoke', 'latency', 'training-smoke', 'training'])
    parser.add_argument('--asset-root')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    bound = bindings(args.asset_root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or PACKAGE / ('checks' if args.stage == 'check' else 'outputs') / (args.stage + '_' + stamp)).resolve()
    if output.exists():
        raise RuntimeError('Choose a new output directory; existing results are not overwritten')
    output.mkdir(parents=True)
    paired = args.stage in ('latency-smoke', 'latency')
    if paired:
        selected = ['deim', 'gccs', 'hilor']
        if args.models:
            raise ValueError('Paired latency requires all three modes; do not select a subset')
    elif args.stage == 'complexity':
        selected = bound['complexity_models']
    elif args.stage in ('check', 'smoke', 'inference'):
        selected = ['deim', 'gccs', 'hilor', 'r18', 'r18_hilor']
        if args.stage == 'check':
            selected += ['train_deim', 'train_gccs']
    else:
        selected = ['train_deim', 'train_gccs']
    if args.models:
        if not set(args.models).issubset(selected):
            raise ValueError('Requested models are outside this stage')
        selected = args.models
    env = dict(os.environ)
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')})
    # Do not inherit a previous torchrun session into a single-GPU measurement.
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    for key in ('TMPDIR', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        target = short_tmp_root(args.asset_root) if key == 'TMPDIR' else PACKAGE / 'outputs/cache' / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    if args.asset_root:
        env['EFFICIENCY_ASSET_ROOT'] = str(Path(args.asset_root).resolve())
    for name in (['paired_latency'] if paired else selected):
        if paired:
            python = bound['models']['deim']['python']
            assert all(bound['models'][m]['python'] == python for m in selected)
            command = [python, '-B', '-u', str(PACKAGE / 'scripts/measure_paired_latency.py'),
                       '--task', 'smoke' if args.stage == 'latency-smoke' else 'formal', '--output', str(output)]
        else:
            row = bound['models'][name]
            script = ('measure_complexity.py' if args.stage == 'complexity' else
                      'measure_training.py' if name.startswith('train_') else 'measure_scoped.py')
            task = 'cpu-check' if args.stage == 'check' else ('smoke' if 'smoke' in args.stage else 'formal')
            command = [row['python'], '-B', '-u', str(PACKAGE / 'scripts' / script),
                       '--model', name, '--task', task, '--output', str(output)]
        if args.asset_root:
            command += ['--asset-root', args.asset_root]
        print('\nRunning:', name, args.stage, flush=True)
        with (output / (name + '.log')).open('w') as handle:
            process = subprocess.Popen(command, cwd=PACKAGE, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        if code:
            write_json(output / 'STATUS.json', {'status': 'FAILED', 'model': name, 'exit_code': code, 'stage': args.stage})
            raise SystemExit(code)
    write_json(output / 'STATUS.json', {'status': 'CPU_PASS_GPU_PENDING' if args.stage == 'check' else 'STAGE_COMPLETE',
                                      'stage': args.stage, 'models': selected, 'publication_ready': False})
    if args.stage in ('inference', 'complexity'):
        export_inference_tables(output, bound)
    if args.stage == 'training':
        costs = []
        for name in selected:
            result = json.loads((output / name / 'TRAINING_COST.json').read_text())
            if result['status'] != 'MEASURED' or result['completed_training_epochs'] != 102:
                raise RuntimeError('Training table requires complete 102-epoch runs')
            costs.append({'Model': result['model'], 'Mean_training_epoch_s': result['mean_training_epoch_time_s'],
                          'Peak_training_allocated_MiB': result['peak_training_allocated_memory_MiB']})
        with (output / 'TRAINING_COST_TABLE.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(costs[0]))
            writer.writeheader()
            writer.writerows(costs)
    if args.stage == 'check':
        scope_path = PACKAGE / 'SCOPE.json'
        scope = json.loads(scope_path.read_text())
        scope['status'] = 'CPU_CHECKED_READY_FOR_GPU_SMOKE' if len(selected) == 7 else 'PARTIAL_CPU_CHECK_GPU_PENDING'
        scope['last_cpu_check'] = str(output.relative_to(PACKAGE)) if output.is_relative_to(PACKAGE) else str(output)
        write_json(scope_path, scope)
    refresh_manifest()
    print('\nCompleted stage:', args.stage, '\nOutputs:', output, flush=True)
    if args.stage == 'check':
        print('CPU checks are not CUDA equivalence, full backward or timing validation.', flush=True)


if __name__ == '__main__':
    main()
