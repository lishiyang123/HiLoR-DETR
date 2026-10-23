"""Recompute main-host and four-configuration tables from saved evaluations."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

METRICS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
           'AR100', 'ARs', 'ARm', 'ARl', 'AP80', 'AP85', 'AP90', 'AP95')
METHODS = ('DEIM', 'GSA', 'HiLoR-DETR')
SYSTEM_METHODS = ('DEIM', 'DEIM+CCVR', 'GSA', 'HiLoR-DETR')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(release, output):
    main = release/'main_host'
    spec = importlib.util.spec_from_file_location('_main_exact', main/'metrics.py')
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    index = read(main/'manifests/CDW1630_THREE_SEED_RELEASE.json')
    rows = []
    for run in index['runs']:
        values = exact.extract(release/run['evaluation'])
        formal = read(release/run['result'])['metrics']
        if any(abs(values[k]-formal[k]) > 1e-12 for k in METRICS):
            raise ValueError('Evaluation and result differ: '+run['result'])
        rows.append(dict(seed=run['seed'], method=run['method'],
                         **{key: values[key] for key in METRICS}))
    expected = {(seed, method) for seed in range(3) for method in METHODS}
    if len(rows) != 9 or {(r['seed'], r['method']) for r in rows} != expected:
        raise ValueError('Expected three methods for each of three seeds')
    rows.sort(key=lambda r: (r['seed'], METHODS.index(r['method'])))
    stage = []
    for method in METHODS:
        for metric in METRICS:
            values = [r[metric] for r in rows if r['method'] == method]
            stage.append(dict(method=method, metric=metric,
                              mean=statistics.fmean(values), sample_sd=statistics.stdev(values)))
    comparison = []
    deltas = []
    for metric in METRICS:
        values = {method: [next(r[metric] for r in rows
                               if r['seed'] == seed and r['method'] == method)
                           for seed in range(3)] for method in METHODS}
        pair = [100*(b-a) for a, b in zip(values['DEIM'], values['HiLoR-DETR'])]
        comparison.append(dict(metric=metric,
            B0_mean=statistics.fmean(values['DEIM']), B0_sample_std=statistics.stdev(values['DEIM']),
            Ours_mean=statistics.fmean(values['HiLoR-DETR']), Ours_sample_std=statistics.stdev(values['HiLoR-DETR']),
            paired_delta_pp_mean=statistics.fmean(pair), paired_delta_pp_sample_std=statistics.stdev(pair),
            **{f'delta_seed{seed}_pp': pair[seed] for seed in range(3)}))
        for left, right in (('DEIM', 'GSA'), ('GSA', 'HiLoR-DETR'), ('DEIM', 'HiLoR-DETR')):
            change = [100*(b-a) for a,b in zip(values[left], values[right])]
            deltas.append(dict(before=left, after=right, metric=metric,
                               mean_pp=statistics.fmean(change), sample_sd_pp=statistics.stdev(change),
                               **{f'seed{seed}_pp': change[seed] for seed in range(3)}))
    paired = [dict(row, method='B0' if row['method']=='DEIM' else 'Ours')
              for row in rows if row['method'] != 'GSA']
    output.mkdir(parents=True)
    write_csv(output/'CDW_FINAL_THREE_SEED_PER_SEED.csv', paired)
    write_csv(output/'CDW_FINAL_THREE_SEED_MEAN_STD.csv', comparison)
    write_csv(output/'stage_per_seed.csv', rows)
    write_csv(output/'stage_summary.csv', stage)
    write_csv(output/'stage_deltas.csv', deltas)
    (output/'CDW_FINAL_THREE_SEED_SUMMARY.json').write_text(json.dumps(dict(
        dataset='CDW1630 Test', seeds=[0,1,2], units='AP/AR fractions; deltas in percentage points',
        per_seed=rows, stage_summary=stage, paired_comparison=comparison, stage_deltas=deltas),
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    system = []
    for run in index['system_runs']:
        values = exact.extract(release/run['evaluation'])
        formal = read(release/run['result'])['metrics']
        if any(abs(values[k]-formal[k]) > 1e-12 for k in METRICS):
            raise ValueError('Evaluation and result differ: '+run['result'])
        system.append(dict(method=run['method'], seed=run['seed'],
                           **{key: values[key] for key in METRICS}))
    if len(system) != 12 or {(r['method'], r['seed']) for r in system} != {
            (method, seed) for method in SYSTEM_METHODS for seed in range(3)}:
        raise ValueError('Expected four configurations for each of three seeds')
    system_summary, system_deltas = [], []
    for method in SYSTEM_METHODS:
        for key in METRICS:
            values = [r[key] for r in system if r['method'] == method]
            system_summary.append(dict(method=method, metric=key,
                mean=statistics.fmean(values), sample_sd=statistics.stdev(values)))
    for left, right in (('DEIM', 'DEIM+CCVR'), ('DEIM', 'GSA'),
                        ('GSA', 'HiLoR-DETR'), ('DEIM', 'HiLoR-DETR')):
        for key in METRICS:
            pairs = [100*(next(r[key] for r in system if r['method']==right and r['seed']==seed)
                          -next(r[key] for r in system if r['method']==left and r['seed']==seed))
                     for seed in range(3)]
            system_deltas.append(dict(before=left, after=right, metric=key,
                **{f'seed{seed}_pp': pairs[seed] for seed in range(3)},
                mean_pp=statistics.fmean(pairs), sample_sd_pp=statistics.stdev(pairs)))
    write_csv(output/'system_per_seed.csv', system)
    write_csv(output/'system_summary.csv', system_summary)
    write_csv(output/'system_deltas.csv', system_deltas)
    for method in METHODS:
        item = next(r for r in stage if r['method']==method and r['metric']=='AP')
        print(f"{method}: AP {100*item['mean']:.6f} +/- {100*item['sample_sd']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    if output.exists() or output == release or release in output.parents:
        parser.error('Choose a new output directory outside the release')
    generate(release, output)


if __name__ == '__main__':
    main()
