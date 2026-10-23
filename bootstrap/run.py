"""CPU paired image Bootstrap for DEIM and HiLoR-DETR, seeds 0, 1 and 2."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import metrics

SEEDS = (0, 1, 2)
METRICS = ('AP', 'AP50', 'AP75', 'AR100')
REPLICATES = 5000
RNG_SEED = 20260915


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(task):
    project, output, seed = task
    started = time.monotonic()
    if hasattr(os, 'nice'):
        os.nice(10)
    inputs = read(project/'dataset_analysis/inputs.json')
    annotation = project/inputs['annotation']
    gt = read(annotation)
    ids = sorted(int(row['id']) for row in gt['images'])
    if len(ids) != 237 or len(gt['annotations']) != 1071:
        raise ValueError('Unexpected CDW1630 Test image or GT count')
    evaluators, points = {}, {}
    for method in ('deim', 'hilor'):
        item = inputs['predictions'][method][str(seed)]
        predictions = read(project/item['path'])
        counts = Counter(int(row['image_id']) for row in predictions)
        if set(counts) != set(ids) or set(counts.values()) != {300}:
            raise ValueError(f'{method} seed {seed}: expected 300 outputs per image')
        evaluator = metrics.make_evaluator(annotation, predictions, ids)
        point = metrics.metric_values(evaluator.eval, evaluator.params)
        replay = metrics.resampled_metrics(evaluator, np.arange(len(ids)))
        for key in METRICS:
            if abs(point[key]-item['formal_metrics'][key]) > 1e-12:
                raise ValueError(f'{method} seed {seed}: {key} formal-result mismatch')
            if abs(point[key]-replay[key]) > 1e-12:
                raise ValueError(f'{method} seed {seed}: {key} cache replay mismatch')
        evaluators[method], points[method] = evaluator, point

    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    rows = []
    for replicate in range(REPLICATES):
        sample = rng.integers(0, len(ids), size=len(ids), endpoint=False)
        deim = metrics.resampled_metrics(evaluators['deim'], sample)
        hilor = metrics.resampled_metrics(evaluators['hilor'], sample)
        row = {'seed': seed, 'replicate': replicate}
        for key in METRICS:
            row['delta_'+key+'_pp'] = 100.0*(hilor[key]-deim[key])
        if not all(np.isfinite(row['delta_'+key+'_pp']) for key in METRICS):
            raise ValueError(f'Non-finite replicate, seed {seed}, sample {replicate}')
        rows.append(row)
        if (replicate+1) % 250 == 0 or replicate+1 == REPLICATES:
            print(f'seed {seed}: {replicate+1}/{REPLICATES}; '
                  f'{time.monotonic()-started:.1f} s', flush=True)

    summary = []
    for key in METRICS:
        summary.append(dict(
            seed=seed, metric=key,
            DEIM_point_percent=100.0*points['deim'][key],
            HiLoR_point_percent=100.0*points['hilor'][key],
            point_delta_pp=100.0*(points['hilor'][key]-points['deim'][key]),
            **metrics.percentile_summary([row['delta_'+key+'_pp'] for row in rows]),
        ))
    seed_dir = output/f'seed{seed}'
    seed_dir.mkdir()
    write_csv(seed_dir/'replicates.csv', rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output.exists() or output == project or project in output.parents:
        parser.error('Choose a new output directory outside the release')
    if version('faster-coco-eval') != '1.7.2':
        raise RuntimeError('Use faster-coco-eval 1.7.2')
    output.mkdir(parents=True)
    tasks = [(project, output, seed) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        summaries = list(pool.map(run_seed, tasks))
    rows = [row for summary in summaries for row in summary]
    write_csv(output/'summary.csv', rows)
    seed_summary = []
    for key in METRICS:
        selected = [row for row in rows if row['metric'] == key]
        row = dict(metric=key, seed_count=len(SEEDS))
        for name in ('DEIM_point_percent', 'HiLoR_point_percent', 'point_delta_pp'):
            values = np.asarray([item[name] for item in selected])
            row[name+'_mean'] = float(values.mean())
            row[name+'_sample_sd'] = float(values.std(ddof=1))
        seed_summary.append(row)
    write_csv(output/'seed_summary.csv', seed_summary)
    print('Completed:', output, flush=True)
    for row in rows:
        print(f"seed {row['seed']} {row['metric']}: delta {row['point_delta_pp']:+.6f} pp, "
              f"95% CI [{row['ci_lower_pp']:+.6f}, {row['ci_upper_pp']:+.6f}]", flush=True)


if __name__ == '__main__':
    main()
