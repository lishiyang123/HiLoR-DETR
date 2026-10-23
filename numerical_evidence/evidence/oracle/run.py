"""Compute full-query oracle geometry from saved DEIM and GCCS Test boxes."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from geometry import best_iou, cxcywh_to_pixel_xyxy, summarize

SEEDS = (0, 1, 2)
MODELS = {'DEIM': 'native', 'GCCS': 'gccs'}
METRICS = ('mean_best_iou', 'median_best_iou', 'oracle_coverage_iou_ge_075')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_boxes(args, model, seed, images):
    if args.boxes is not None:
        path = args.boxes/model.lower()/f'seed{seed}.npz'
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['image_ids']
            raw = saved['pred_boxes_cxcywh_normalized']
            sizes_hw = saved['orig_sizes_hw']
    else:
        import torch

        source = args.captures/f'{MODELS[model]}_seed{seed}/evaluate_test'
        result = read(source/'RESULT.json')
        if result['status'] != 'EVALUATED' or result['seed'] != seed:
            raise ValueError(f'Unexpected formal result identity: {source}')
        data = torch.load(source/'test_capture.pt', map_location='cpu', weights_only=False)
        if (data['split'] != 'test' or data['raw_box_format'] != 'normalized_cxcywh'
                or data['orig_size_order'] != 'height,width'):
            raise ValueError('Unexpected Test capture coordinate format')
        records = sorted(data['records'], key=lambda row: int(row['image_id']))
        ids = np.asarray([int(row['image_id']) for row in records], dtype=np.int64)
        raw = np.stack([row['pred_boxes'].numpy() for row in records])
        sizes_hw = np.asarray([row['orig_size'].tolist() for row in records], dtype=np.int64)

    if ids.tolist() != sorted(images) or raw.shape != (237, 300, 4):
        raise ValueError(f'{model} seed{seed}: incomplete image or query pool')
    if raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Expected finite FP32 detector outputs')
    expected_sizes = np.asarray([[images[int(i)]['height'], images[int(i)]['width']] for i in ids])
    if not np.array_equal(sizes_hw, expected_sizes):
        raise ValueError('Capture and annotation image sizes differ')
    if args.boxes is None:
        path = args.output/'boxes'/model.lower()/f'seed{seed}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, image_ids=ids,
                                pred_boxes_cxcywh_normalized=raw, orig_sizes_hw=sizes_hw)
    return {int(i): cxcywh_to_pixel_xyxy(boxes, images[int(i)]['width'], images[int(i)]['height'])
            for i, boxes in zip(ids, raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        default=Path(__file__).resolve().parents[3])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--captures', type=Path)
    source.add_argument('--boxes', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.captures is None and args.boxes is None:
        args.boxes = Path(__file__).resolve().parent/'results/boxes'
    args.release, args.output = args.release.resolve(), args.output.resolve()
    if args.output.exists() or args.output == args.release or args.release in args.output.parents:
        parser.error('Choose a new output directory outside the release')
    annotation = read(args.release/'main_host/datasets/CDW1630/annotations/instances_test2017.json')
    images = {int(row['id']): row for row in annotation['images']}
    targets = sorted(annotation['annotations'], key=lambda row: (int(row['image_id']), int(row['id'])))
    if len(images) != 237 or len(targets) != 1071 or len({int(a['id']) for a in targets}) != 1071:
        raise ValueError('Unexpected Test image or annotation count')
    if any(a.get('iscrowd', 0) for a in targets):
        raise ValueError('Crowd GT is outside the original analysis scope')
    by_image = defaultdict(list)
    for target in targets:
        by_image[int(target['image_id'])].append(target)
    args.output.mkdir(parents=True)
    per_seed, per_gt = [], []
    for seed in SEEDS:
        values, argmax = {}, {}
        for model in MODELS:
            pool = load_boxes(args, model, seed, images)
            best, query = {}, {}
            invalid = sum(int((boxes[:, 2:] <= boxes[:, :2]).any(axis=1).sum()) for boxes in pool.values())
            for image_id in sorted(images):
                items = by_image[image_id]
                gt = np.asarray([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2],
                                  a['bbox'][1]+a['bbox'][3]] for a in items], dtype=np.float64).reshape(-1, 4)
                u, q = best_iou(pool[image_id], gt)
                best.update({a['id']: float(v) for a, v in zip(items, u)})
                query.update({a['id']: int(v) for a, v in zip(items, q)})
            values[model] = np.asarray([best[a['id']] for a in targets])
            argmax[model] = query
            row = dict(seed=seed, model=model, **summarize(values[model]),
                       covered_gt_iou_ge_075=int((values[model] >= .75).sum()),
                       nonpositive_prediction_count=invalid)
            per_seed.append(row)
            print(model, 'seed', seed, json.dumps(row), flush=True)
        for index, target in enumerate(targets):
            per_gt.append(dict(seed=seed, image_id=target['image_id'], annotation_id=target['id'],
                               category_id=target['category_id'], DEIM_best_iou=values['DEIM'][index],
                               GCCS_best_iou=values['GCCS'][index],
                               delta=values['GCCS'][index]-values['DEIM'][index],
                               DEIM_argmax_query_0based=argmax['DEIM'][target['id']],
                               GCCS_argmax_query_0based=argmax['GCCS'][target['id']]))
    summary = []
    for metric in METRICS:
        left = np.asarray([row[metric] for row in per_seed if row['model'] == 'DEIM'])
        right = np.asarray([row[metric] for row in per_seed if row['model'] == 'GCCS'])
        summary.append(dict(metric=metric, DEIM_mean=float(left.mean()),
                            DEIM_sample_sd=float(left.std(ddof=1)), GCCS_mean=float(right.mean()),
                            GCCS_sample_sd=float(right.std(ddof=1)),
                            paired_delta_mean=float((right-left).mean()),
                            paired_delta_sample_sd=float((right-left).std(ddof=1))))
    write_csv(args.output/'per_gt.csv', per_gt)
    write_csv(args.output/'per_seed.csv', per_seed)
    write_csv(args.output/'summary.csv', summary)
    print('Three-seed summary:', json.dumps(summary, indent=2), flush=True)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
