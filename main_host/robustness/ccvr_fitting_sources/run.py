"""Recompute CCVR fitting-source tables, Val metrics, or fold heads on CPU."""

import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fitting_inputs(release, index):
    counts = index['split_counts']
    for split in ('train', 'val'):
        ann = read(release / index['annotations'][split])
        assert len(ann['images']) == counts[split]['images']
        assert len(ann['annotations']) == counts[split]['gt']
    fold_map, seeds = None, {}
    for seed, source in index['seeds'].items():
        mapping = read(release / source['mapping_fits'])
        if fold_map is None:
            fold_map = mapping['fold_by_image']
        assert mapping['fold_by_image'] == fold_map
        assert mapping['fixed_lambda_reg'] == index['settings']['qccr']['lambda_reg']
        parameters = read(release / source['qccr_artifact'])
        assert parameters['source_checkpoint_sha256'] == source['checkpoint_sha256']
        assert parameters['parameters'] == mapping['candidates']['1e-03']['full_val_fit']['parameters']
        metadata = read(release / source['formal_umr_metadata'])
        assert metadata['source_checkpoint_sha256'] == source['checkpoint_sha256']
        assert fit.sha256(release / source['formal_umr_head']) == metadata['weight_sha256']
        entry = {key: dict(path=str(release / source[key]))
                 for key in ('train_capture', 'val_capture', 'formal_umr_head', 'qccr_fits') if key in source}
        entry['qccr_fits'] = dict(path=str(release / source['mapping_fits']))
        entry['detector_checkpoint'] = dict(sha256=source['checkpoint_sha256'])
        seeds[seed] = entry
    assert len(fold_map) == counts['val']['images']
    lock = dict(seeds=seeds, shared_fold_by_image=fold_map,
                val_annotation=dict(path=str(release / index['annotations']['val'])))
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=HERE.parents[2])
    parser.add_argument('--stage', choices=('tables', 'metrics', 'fit'), default='tables')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release, output = args.release.resolve(strict=True), args.output.resolve()
    if output.exists() or output == release or release in output.parents:
        parser.error('Use a new output directory outside the release')
    index = read(HERE / 'inputs.json')
    torch = fit.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    payloads = []
    if args.stage == 'fit':
        lock = fitting_inputs(release, index)
        sys.path.insert(0, str(release / index['source']))
        from engine.deim.competition_mapping import apply_qccr, compute_competition_risk
        from engine.deim.candidate_value import UMRHead, box_cxcywh_to_xyxy, build_image_features
        runtime = dict(apply_qccr=apply_qccr, compute_competition_risk=compute_competition_risk,
                       CandidateValueHead=UMRHead, box_cxcywh_to_xyxy=box_cxcywh_to_xyxy,
                       build_image_features=build_image_features)
        protocol = dict(index['settings'], split_counts=index['split_counts'])
        output.mkdir(parents=True)
        for seed in (0, 1, 2):
            payloads.append(fit.run_seed(seed, lock['seeds'][str(seed)], lock, output, protocol, runtime))
    else:
        output.mkdir(parents=True)
        for seed in (0, 1, 2):
            payload = read(HERE / ('seed%d/OOF_METRICS.json' % seed))
            assert payload['detector_seed'] == seed and not payload['test_accessed']
            assert payload['val_image_count'] == index['split_counts']['val']['images']
            if args.stage == 'metrics':
                payload = copy.deepcopy(payload)
                for row in fit.ROWS:
                    path = HERE / ('seed%d/%s_predictions.json.gz' % (seed, row))
                    with gzip.open(path, 'rt', encoding='utf-8') as stream:
                        detections = json.load(stream)
                    ids = sorted({int(d['image_id']) for d in detections})
                    assert len(detections) == len(ids) * 300 == payload['detections_per_row']
                    metrics = fit.run_coco(release / index['annotations']['val'], detections, ids)
                    for metric, value in metrics.items():
                        target = payload['metrics'][row][metric]
                        assert (target is None and value != value) or (target is not None and abs(value-target) < 1e-12)
                    payload['metrics'][row] = metrics
                baseline = payload['metrics']['qccr_only']
                payload['delta_vs_qccr_only_pp'] = {
                    row: {k: 100 * (payload['metrics'][row][k] - baseline[k]) for k in fit.METRICS}
                    for row in fit.ROWS}
                fit.write_json_atomic(output / ('seed%d/OOF_METRICS.json' % seed), fit.finite(payload))
            payloads.append(payload)
    fit.aggregate(payloads, output)
    summary = read(output / 'THREE_SEED_SUMMARY.json')
    for row in summary['results']:
        delta = row['delta_vs_qccr_only_pp']['AP']
        print('%s: Val AP %.6f; paired increment %+.6f +/- %.6f pp' %
              (row['row'], 100 * row['metrics']['AP']['mean'], delta['mean'], delta['sample_std']))


if __name__ == '__main__':
    main()
