"""Rebuild RT-DETR three-stage tables from the published Test evaluation arrays."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_summary(root):
    import numpy as np
    import torch
    from coco_test_evaluation import named_metrics
    adapter = root / 'second_host'
    annotation = root / 'main_host/datasets/CDW1630/annotations/instances_test2017.json'
    result = dict(schema='rtdetr-three-stage-test-summary-v2', created_at_utc=datetime.now(timezone.utc).isoformat(),
        operation='reporting from saved Test evaluation arrays', seeds=[0, 1, 2], sample_sd_denominator='n-1',
        annotation=dict(path=annotation.relative_to(root).as_posix(), bytes=annotation.stat().st_size, sha256=digest(annotation)),
        evaluator='faster-coco-eval', evaluator_version='1.7.2', methods={},
        test_used_for_selection_fitting_or_tuning=False)
    for mode in ('native', 'gsa', 'full'):
        folder = (root / 'benchmarks/02_RT_DETR_R18/results/test_reeval_v1'
                  if mode == 'native' else adapter / ('evaluations_gsa' if mode == 'gsa' else 'evaluations'))
        values, sources = {}, {}
        for seed in range(3):
            path = folder / f'seed{seed}/coco_bbox_eval.pth'
            values[str(seed)] = named_metrics(torch.load(path, map_location='cpu', weights_only=False))
            sources[str(seed)] = dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=digest(path),
                                     predictions_sha256=digest(path.parent / 'predictions.json'))
        stats = {}
        for metric in values['0']:
            numbers = [100 * values[str(seed)][metric] for seed in range(3)]
            stats[metric] = dict(mean=float(np.mean(numbers)), sample_sd=float(np.std(numbers, ddof=1)),
                                 values=numbers, unit='percentage points')
        result['methods'][mode] = dict(per_seed_fraction=values, sources=sources, three_seed_percentage_points=stats)
    result['deltas_percentage_points'] = {}
    for metric in result['methods']['native']['three_seed_percentage_points']:
        means = {mode: result['methods'][mode]['three_seed_percentage_points'][metric]['mean'] for mode in ('native', 'gsa', 'full')}
        result['deltas_percentage_points'][metric] = dict(gsa_vs_native=means['gsa']-means['native'],
            full_vs_gsa=means['full']-means['gsa'], full_vs_native=means['full']-means['native'])
    return result


def write_tables(summary, output):
    output.mkdir(parents=True)
    (output / 'RTDETR_THREE_STAGE_ALL_AND_STRICT_IOU.json').write_text(json.dumps(summary, indent=2) + '\n')
    metrics = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl', 'AP80', 'AP85', 'AP90', 'AP95']
    with (output / 'RTDETR_PER_SEED.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['configuration', 'seed', *metrics])
        for mode in ('native', 'gsa', 'full'):
            for seed in range(3):
                values = summary['methods'][mode]['per_seed_fraction'][str(seed)]
                writer.writerow([mode, seed, *[100*values[key] for key in metrics]])
    rows = []
    for metric in metrics:
        stats = [summary['methods'][mode]['three_seed_percentage_points'][metric] for mode in ('native', 'gsa', 'full')]
        delta = summary['deltas_percentage_points'][metric]
        rows.append([metric, *[value for item in stats for value in (item['mean'], item['sample_sd'])],
                     delta['gsa_vs_native'], delta['full_vs_gsa'], delta['full_vs_native']])
    with (output / 'RTDETR_MEAN_STD_AND_DELTA.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'native_mean', 'native_sample_sd', 'gsa_mean', 'gsa_sample_sd',
                         'full_mean', 'full_sample_sd', 'gsa_vs_native', 'full_vs_gsa', 'full_vs_native'])
        writer.writerows(rows)
    lines = ['# RT-DETR-R18 final Test results', '',
             'Values are percentages; ± is the sample standard deviation over detector seeds 0/1/2. Deltas are percentage points.', '',
             '| Metric | Native | +GSA | Full HiLoR-DETR | GSA gain | CCVR increment | Full gain |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells = [row[0], *[f'{row[i]:.3f} ± {row[i+1]:.3f}' for i in (1, 3, 5)], *[f'{v:+.3f}' for v in row[7:]]]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'RTDETR_PAPER_TABLE.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        parser.error('Use a new output outside the release')
    write_tables(make_summary(ROOT), output)
    print('RT TEST TABLES REBUILT FROM SAVED ARRAYS')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
