"""CPU reproduction of condition-group and per-class results."""

import argparse
from collections import Counter
import csv
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))

import numpy as np
import analyze_dataset as dataset
import metrics

METHODS = {'DEIM': 'deim', 'HiLoR-DETR': 'hilor'}
SEEDS = (0, 1, 2)
GROUP_METRICS = ('AP', 'AP50', 'AP75', 'AR100')
CLASS_METRICS = ('AP', 'AP75', 'AR100')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    center = mean(values)
    return (sum((value-center)**2 for value in values)/(len(values)-1))**.5


def make_groups():
    parsed = {}
    base = ROOT / 'main_host/datasets/CDW1630/annotations'
    for split in dataset.SPLITS:
        annotation = read(base / ('instances_'+split.lower()+'2017.json'))
        parsed[split] = dict(images={int(row['id']):row for row in annotation['images']},
                             annotations=annotation['annotations'],
                             categories={int(row['id']):row['name'] for row in annotation['categories']})
    thresholds, groups, _, _ = dataset.grouping_tables(parsed)
    saved = read(HERE / 'grouping/test_image_groups.json')
    if thresholds != read(HERE / 'grouping/thresholds.json') or groups != saved:
        raise ValueError('Annotation-derived groups differ from the supplied grouping files')
    if thresholds['density']['strategy'] != 'fixed' or thresholds['density']['threshold'] != 5:
        raise ValueError('The density cutoff must be fixed at five')
    return groups


def save_groups(rows, groups, output):
    paired, summary = [], []
    for name, group in groups['groups'].items():
        for seed in SEEDS:
            pair = {method:next(row for row in rows if row['group']==name
                                and row['method']==method and row['seed']==seed) for method in METHODS}
            paired.append(dict(group=name,seed=seed,
                               **{'delta_'+key+'_pp':pair['HiLoR-DETR'][key]-pair['DEIM'][key]
                                  for key in GROUP_METRICS}))
        row = dict(group=name,image_count=group['image_count'],gt_count=group['gt_count'])
        for metric in GROUP_METRICS:
            for method,label in [('DEIM','DEIM'),('HiLoR-DETR','HiLoR_DETR')]:
                values = [next(item[metric] for item in rows if item['group']==name
                               and item['method']==method and item['seed']==seed) for seed in SEEDS]
                row[label+'_'+metric+'_mean'] = mean(values)
                row[label+'_'+metric+'_sample_sd'] = sample_sd(values)
            values = [item['delta_'+metric+'_pp'] for item in paired if item['group']==name]
            row['paired_delta_'+metric+'_mean_pp'] = mean(values)
            row['paired_delta_'+metric+'_sample_sd_pp'] = sample_sd(values)
        summary.append(row)
    write_csv(output/'per_seed.csv',rows)
    write_csv(output/'paired_deltas.csv',paired)
    write_csv(output/'summary.csv',summary)


def save_classes(rows, categories, counts, output):
    summary = []
    for category in categories:
        category_id = int(category['id'])
        name = 'Timber' if category['name']=='timber' else category['name']
        for metric in CLASS_METRICS:
            values = {method:np.asarray([next(row[metric] for row in rows
                                            if row['method']==method and row['seed']==seed
                                            and row['category_id']==category_id) for seed in SEEDS])
                      for method in METHODS}
            delta = values['HiLoR-DETR']-values['DEIM']
            summary.append(dict(category_id=category_id,category_name=name,instance_count=counts[category_id],
                                metric=metric,DEIM_mean_percent=float(values['DEIM'].mean()),
                                DEIM_sample_sd_percent=float(values['DEIM'].std(ddof=1)),
                                HiLoR_DETR_mean_percent=float(values['HiLoR-DETR'].mean()),
                                HiLoR_DETR_sample_sd_percent=float(values['HiLoR-DETR'].std(ddof=1)),
                                paired_delta_mean_pp=float(delta.mean()),
                                paired_delta_sample_sd_pp=float(delta.std(ddof=1))))
    count_rows = [dict(category_id=int(row['id']),category_name='Timber' if row['name']=='timber' else row['name'],
                       instance_count=counts[int(row['id'])]) for row in categories]
    count_rows.append(dict(category_id='all',category_name='All',instance_count=sum(counts.values())))
    write_csv(output/'classwise_per_seed.csv',rows)
    write_csv(output/'classwise_mean_std.csv',summary)
    write_csv(output/'class_instance_counts.csv',count_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','groups','classes'),default='all')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output==ROOT or ROOT in output.parents:
        parser.error('Choose a new output directory outside the release')
    if version('faster-coco-eval')!='1.7.2':
        raise RuntimeError('Use faster-coco-eval 1.7.2')
    inputs = read(HERE/'inputs.json')
    annotation_path = ROOT/inputs['annotation']
    annotation = read(annotation_path)
    ids = sorted(int(row['id']) for row in annotation['images'])
    if len(ids)!=inputs['image_count'] or len(annotation['annotations'])!=inputs['gt_count']:
        raise ValueError('Test annotation does not match the saved prediction inputs')
    categories = sorted(annotation['categories'],key=lambda row:int(row['id']))
    counts = Counter(int(row['category_id']) for row in annotation['annotations'])
    groups = make_groups() if args.stage in ('all','groups') else None
    group_rows,class_rows = [],[]
    for method,label in METHODS.items():
        for seed in SEEDS:
            item = inputs['predictions'][label][str(seed)]
            predictions = read(ROOT/item['path'])
            prediction_counts = Counter(int(row['image_id']) for row in predictions)
            if set(prediction_counts)!=set(ids) or set(prediction_counts.values())!={300}:
                raise ValueError('Predictions do not contain 300 outputs per Test image')
            overall,evaluation = metrics.evaluate(annotation_path,predictions,ids)
            if max(abs(overall[key]-item['formal_metrics'][key]) for key in GROUP_METRICS)>1e-12:
                raise ValueError('Saved predictions differ from their formal Test metrics')
            if args.stage in ('all','classes'):
                for row in metrics.per_class(evaluation,categories):
                    class_rows.append(dict(method=method,seed=seed,category_id=row['category_id'],
                                           category_name=row['category_name'],instance_count=counts[row['category_id']],
                                           **{key:row[key] for key in CLASS_METRICS}))
            if groups is not None:
                for name,group in groups['groups'].items():
                    values,_ = metrics.evaluate(annotation_path,predictions,group['image_ids'])
                    group_rows.append(dict(group=name,method=method,seed=seed,image_count=group['image_count'],
                                           gt_count=group['gt_count'],**{key:100*values[key] for key in GROUP_METRICS}))
            print(method,'seed',seed,'evaluated from saved predictions.',flush=True)
    output.mkdir(parents=True)
    if groups is not None:
        save_groups(group_rows,groups,output/'groups')
    if args.stage in ('all','classes'):
        save_classes(class_rows,categories,counts,output/'classes')
    print('Saved:',output,flush=True)


if __name__=='__main__':
    main()
