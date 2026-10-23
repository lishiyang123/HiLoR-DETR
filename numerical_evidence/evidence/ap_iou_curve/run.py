"""Export three-seed AP-IoU curves and paired differences from formal Test outputs."""

import argparse
from collections import Counter
import csv
import gzip
import importlib.util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np

SEEDS = (0, 1, 2)
MODELS = {'DEIM': ('deim', 'native'), 'HiLoR': ('hilor', 'full')}
IOU_LABELS = tuple(f'AP{value:.2f}' for value in
                   (.50, .55, .60, .65, .70, .75, .80, .85, .90, .95))
IOU_VALUES = tuple(float(label[2:]) for label in IOU_LABELS)


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fail(message: str) -> None:
    raise RuntimeError(message)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        fail(f"Expected exactly three seeds, got {len(values)}")
    return statistics.fmean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument('--work', type=Path,
                        help='Read native_seedN/full_seedN/evaluate_test/eval.pth instead of replaying saved predictions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    if output.exists() or output == release or release in output.parents:
        parser.error('Choose a new output directory outside the release')
    if version('faster-coco-eval') != '1.7.2':
        raise RuntimeError('Use faster-coco-eval 1.7.2')
    source = release/'main_host/source/FINAL_CODE/tools/qccr/metrics.py'
    extract = module(source, '_ap_iou_exact').extract_exact_metrics
    inputs = read(release/'dataset_analysis/inputs.json')
    annotation_path = release/inputs['annotation']
    annotation = read(annotation_path)
    image_ids = sorted(int(row['id']) for row in annotation['images'])
    category_ids = sorted(int(row['id']) for row in annotation['categories'])
    if len(image_ids) != 237 or len(set(image_ids)) != 237 or len(annotation['annotations']) != 1071:
        raise ValueError('Unexpected Test image or GT count')
    replay = None
    if args.work is None:
        replay = module(release/'dataset_analysis/source/metrics.py', '_ap_iou_replay')
    values = {method: {} for method in MODELS}
    for method, (label, folder) in MODELS.items():
        for seed in SEEDS:
            item = inputs['predictions'][label][str(seed)]
            if args.work is not None:
                import torch

                base = args.work/f'{folder}_seed{seed}/evaluate_test'
                formal = read(base/'RESULT.json')
                if formal['seed'] != seed or formal['status'] != 'EVALUATED':
                    raise ValueError('Unexpected saved formal result identity')
                evaluation = torch.load(base/'eval.pth', map_location='cpu', weights_only=False)
                if 'precision' not in evaluation:
                    evaluation = evaluation['eval']
                reference = formal['metrics']
            else:
                predictions = read(release/item['path'])
                counts = Counter(int(row['image_id']) for row in predictions)
                if set(counts) != set(image_ids) or set(counts.values()) != {300}:
                    raise ValueError('Expected 300 saved predictions per Test image')
                _, evaluation = replay.evaluate(annotation_path, predictions, image_ids)
                reference = item['formal_metrics']
            params = evaluation['params']
            if (set(map(int, params.imgIds)) != set(image_ids)
                    or sorted(map(int, params.catIds)) != category_ids
                    or list(map(int, params.maxDets)) != [1, 10, 100]
                    or not np.allclose(params.iouThrs, IOU_VALUES, rtol=0, atol=1e-12)):
                raise ValueError('Evaluation image/category/threshold/budget scope differs')
            exact = extract(evaluation)
            for key in ('AP', 'AP50', 'AP75', 'AR100'):
                if abs(exact['overall'][key]-reference[key]) > 1e-12:
                    raise ValueError(f'{method} seed{seed}: {key} differs from formal result')
                if abs(reference[key]-item['formal_metrics'][key]) > 1e-12:
                    raise ValueError('Saved evaluation and shared-prediction identities differ')
            curve = {label: float(exact['per_iou_ap'][label]) for label in IOU_LABELS}
            if not all(math.isfinite(value) for value in curve.values()):
                raise ValueError('Nonfinite AP-IoU curve value')
            if abs(statistics.fmean(curve.values())-exact['overall']['AP']) > 1e-12:
                raise ValueError('Mean of the ten thresholds differs from overall AP')
            values[method][seed] = curve
            print(method, 'seed', seed, 'read from',
                  'formal eval.pth' if args.work is not None else 'shared formal predictions', flush=True)

    rows = []
    for label, iou in zip(IOU_LABELS, IOU_VALUES):
        deim = [values['DEIM'][seed][label] for seed in SEEDS]
        hilor = [values['HiLoR'][seed][label] for seed in SEEDS]
        deltas = [100.0*(right-left) for left, right in zip(deim, hilor)]
        deim_mean, deim_sd = mean_std(deim)
        hilor_mean, hilor_sd = mean_std(hilor)
        delta_mean, delta_sd = mean_std(deltas)
        rows.append(dict(
            iou=iou,
            **{f'DEIM_seed{seed}': deim[seed] for seed in SEEDS},
            **{f'HiLoR_seed{seed}': hilor[seed] for seed in SEEDS},
            **{f'delta_seed{seed}_pp': deltas[seed] for seed in SEEDS},
            DEIM_mean=deim_mean, DEIM_sample_sd=deim_sd,
            HiLoR_mean=hilor_mean, HiLoR_sample_sd=hilor_sd,
            paired_delta_mean_pp=delta_mean, paired_delta_sample_sd_pp=delta_sd,
        ))
    output.mkdir(parents=True)
    with (output/'curve.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('IoU, DEIM AP (%), HiLoR AP (%), paired delta +/- sample SD (pp)', flush=True)
    for row in rows:
        print(f"{row['iou']:.2f}, {100*row['DEIM_mean']:.6f}, {100*row['HiLoR_mean']:.6f}, "
              f"{row['paired_delta_mean_pp']:+.6f} +/- {row['paired_delta_sample_sd_pp']:.6f}", flush=True)
    print('Saved:', output/'curve.csv', flush=True)


if __name__ == '__main__':
    main()
