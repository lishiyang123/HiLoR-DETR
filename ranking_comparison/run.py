"""Reproduce the seed-0 candidate-ranking comparison from saved candidates."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / 'source'
sys.path.insert(0, str(SOURCE))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from nearest_neighbor_baselines.coco_evaluator import evaluate_coco_detections

METRICS = ('AP', 'AP50', 'AP75', 'AP80', 'AP85', 'AP90', 'AP95', 'AR100')
METHODS = (
    ('soft_nms_linear', 'soft_nms_linear'),
    ('soft_nms_gaussian', 'soft_nms_gaussian'),
    ('gilg_reference', 'gilg_original'),
    ('gilg_complete_pool', 'gilg_original'),
    ('iou_regression', 'iou_reg_rescore'),
)


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def flat_metrics(metrics):
    return {name: float(metrics[name] if name in metrics else metrics['threshold_ap'][name])
            for name in METRICS}


def require(paths):
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative = [str(path.relative_to(ROOT)) for path in missing]
        raise FileNotFoundError('Missing inputs:\n  ' + '\n  '.join(relative)
                                + '\nSee DISTRIBUTION.md#downloads-by-task.')


def command(script, *arguments):
    subprocess.run([sys.executable, '-u', '-B', str(SOURCE / script), *map(str, arguments)],
                   check=True)


def prediction_file(folder, method):
    path = folder / (method + '_test_detections.json.gz')
    return path if path.is_file() else path.with_suffix('')


def reference_row(settings, method, name):
    path = ROOT / settings['reference_results'][name]
    result = read(path)
    if result.get('status') != 'EVALUATED':
        raise ValueError('Incomplete reference result: ' + str(path))
    if result['weight_sha256'] != settings['checkpoint_sha256']:
        raise ValueError('Reference detector does not match the common candidate pool')
    return dict(method=method,
                **{key: 100.0 * float(result['metrics'][key]) for key in METRICS})


def save_rows(output, rows):
    for row in rows:
        row['delta_AP_pp'] = row['AP'] - rows[0]['AP']
    columns = ['method', *METRICS, 'delta_AP_pp']
    with (output / 'comparison.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n',
                                            encoding='utf-8')
    for row in rows:
        print('{method:22s} AP={AP:.6f} delta={delta_AP_pp:+.6f}'.format(**row))


def saved_metrics(settings, output):
    annotation = ROOT / settings['annotations']['test']
    jobs = [('Native GCCS', ROOT / settings['reference_predictions']['native'])]
    jobs += [(name, prediction_file(HERE / 'results' / name, method)) for name, method in METHODS]
    jobs += [('Complete CCVR', ROOT / settings['reference_predictions']['full'])]
    rows = [dict(method=name, **flat_metrics(evaluate_coco_detections(annotation, read(path))))
            for name, path in jobs]
    save_rows(output, rows)


def fit(settings, output):
    captures = {key: ROOT / value for key, value in settings['captures'].items()}
    from nearest_neighbor_baselines.capture_io import load_capture
    for name in ('train', 'val'):
        if load_capture(captures[name], mmap=True)['checkpoint_sha256'] != settings['checkpoint_sha256']:
            raise ValueError(name + ' capture belongs to a different detector')
    for name in ('fitted', 'selection', 'val'):
        (output / name).mkdir()
    sweeps = []
    for method in ('linear', 'gaussian'):
        target = output / 'val' / (method + '.json')
        command('sweep_soft_nms.py', '--capture', captures['val'],
                '--annotation', ROOT / settings['annotations']['val'], '--split', 'val',
                '--method', method, '--values', ','.join(map(str, settings['soft_nms'][method])),
                '--topk', settings['topk'], '--output', target)
        sweeps.append(target)
    command('select_soft_nms.py', '--sweeps', *sweeps,
            '--output', output / 'selection/soft_nms.json')
    gilg = settings['gilg']
    for name, extra in (('gilg_reference', []), ('gilg_complete_pool', ['--keep-complete-pool'])):
        command('fit_on_val.py', '--capture', captures['val'],
                '--output', output / 'fitted' / (name + '.json'), '--seed', gilg['seed'],
                '--maxiter', gilg['maxiter'], '--label-iou', gilg['label_iou'],
                '--near-duplicate-iou', gilg['near_duplicate_iou'], *extra)
    head = settings['iou_regression']
    command('fit_iou_regressor.py', '--train-capture', captures['train'],
            '--val-capture', captures['val'], '--output-dir', output / 'fitted/iou_regression',
            '--seed', head['seed'], '--epochs', head['epochs'], '--patience', head['patience'],
            '--batch-images', head['batch_images'], '--learning-rate', head['learning_rate'],
            '--weight-decay', head['weight_decay'], '--device', 'cpu')


def replay(settings, output, state):
    capture = ROOT / settings['captures']['test']
    from nearest_neighbor_baselines.capture_io import load_capture
    if load_capture(capture, mmap=True)['checkpoint_sha256'] != settings['checkpoint_sha256']:
        raise ValueError('Test capture belongs to a different detector')
    selected = read(state / 'selection/soft_nms.json')['methods']
    head_seed = settings['iou_regression']['seed']
    head = state / 'fitted/iou_regression' / ('iou_regressor_seed%d' % head_seed)
    extras = {
        'soft_nms_linear': ['--iou-threshold', selected['soft_nms_linear']['selected']['grid_value']],
        'soft_nms_gaussian': ['--sigma', selected['soft_nms_gaussian']['selected']['grid_value']],
        'gilg_reference': ['--gilg-params', state / 'fitted/gilg_reference.json',
                           '--near-duplicate-iou', settings['gilg']['near_duplicate_iou']],
        'gilg_complete_pool': ['--gilg-params', state / 'fitted/gilg_complete_pool.json',
                              '--keep-complete-pool'],
        'iou_regression': ['--iou-reg-checkpoint', head.with_suffix('.pt'),
                           '--iou-reg-metadata', head.with_suffix('.metadata.json')],
    }
    root = output / 'results'
    root.mkdir()
    rows = [reference_row(settings, 'Native GCCS', 'native')]
    for name, method in METHODS:
        folder = root / name
        command('evaluate_on_test.py', '--capture', capture,
                '--annotation', ROOT / settings['annotations']['test'], '--expected-split', 'test',
                '--method', method, '--topk', settings['topk'], '--output-dir', folder, *extras[name])
        summary = read(folder / (method + '_test_summary.json'))
        rows.append(dict(method=name, **flat_metrics(summary['metrics'])))
    rows.append(reference_row(settings, 'Complete CCVR', 'full'))
    save_rows(output, rows)


def inputs_for(settings, stage):
    paths = [ROOT / settings['annotations']['test']]
    if stage == 'metrics':
        paths += [ROOT / value for value in settings['reference_predictions'].values()]
        paths += [prediction_file(HERE / 'results' / name, method) for name, method in METHODS]
    if stage in ('fit', 'all'):
        paths += [ROOT / settings['captures'][name] for name in ('train', 'val')]
        paths += [ROOT / settings['annotations']['val']]
    if stage in ('replay', 'all'):
        paths += [ROOT / settings['captures']['test']]
        paths += [ROOT / value for value in settings['reference_results'].values()]
    if stage == 'replay':
        paths += [HERE / 'state/selection/soft_nms.json',
                  HERE / 'state/fitted/gilg_reference.json', HERE / 'state/fitted/gilg_complete_pool.json']
        head = HERE / 'state/fitted/iou_regression' / ('iou_regressor_seed%d' % settings['iou_regression']['seed'])
        paths += [head.with_suffix('.pt'), head.with_suffix('.metadata.json')]
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('metrics', 'replay', 'fit', 'all'), default='metrics')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = read(HERE / 'settings.json')
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Choose an output directory outside the release')
    if output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(output))
    require(inputs_for(settings, args.stage))
    if args.stage != 'metrics':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Candidate replay/fitting uses CUDA FP32 to match main-pipeline postprocessing. '
                               'Use --stage metrics for CPU-only evaluation of saved predictions.')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    if args.stage == 'metrics':
        saved_metrics(settings, output)
    if args.stage in ('fit', 'all'):
        fit(settings, output)
    if args.stage in ('replay', 'all'):
        replay(settings, output, output if args.stage == 'all' else HERE / 'state')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
