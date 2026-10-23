"""Seed-0 CCVR coefficients, stage increments and COCO matching diagnostics."""

import argparse
from collections import defaultdict
import contextlib
import csv
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch
from torchvision.ops import box_convert


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def absolute_boxes(record):
    boxes = box_convert(record['pred_boxes'].float(), in_fmt='cxcywh', out_fmt='xyxy')
    return boxes * record['orig_size'][[1, 0, 1, 0]]


def capture_pair(release, before, after, rows_before, rows_after, expected_ids):
    left = torch.load(release / before, map_location='cpu', weights_only=False)
    right = torch.load(release / after, map_location='cpu', weights_only=False)
    records = [{int(r['image_id']): r for r in cap['records']} for cap in (left, right)]
    assert set(records[0]) == set(records[1]) == expected_ids
    for cap in (left, right):
        assert cap['raw_box_format'] == 'normalized_cxcywh'
        assert cap['post_box_format'] == 'absolute_xyxy'
        assert cap['orig_size_order'] == 'height,width'
    queries = []
    for phase, predictions in enumerate((rows_before, rows_after)):
        grouped = defaultdict(list)
        for row in predictions:
            grouped[int(row['image_id'])].append(row)
        phase_queries = {}
        for image_id in sorted(expected_ids):
            rec = records[phase][image_id]
            for key in ('pred_boxes', 'pred_logits', 'hL', 'orig_size'):
                assert torch.equal(records[0][image_id][key], records[1][image_id][key]), (image_id, key)
            assert rec['pred_boxes'].shape == (300, 4) and rec['pred_logits'].shape == (300, 5)
            boxes = absolute_boxes(rec)
            query_by_box = defaultdict(list)
            for query, values in enumerate(boxes.tolist()):
                query_by_box[tuple(values)].append(query)
            xywh = rec['post_boxes'].clone()
            xywh[:, 2:] -= xywh[:, :2]
            predictions_image = grouped[image_id]
            assert len(predictions_image) == len(rec['post_scores']) == 300
            candidates = []
            for rank, row in enumerate(predictions_image):
                assert row['bbox'] == xywh[rank].tolist()
                assert row['category_id'] == int(rec['post_labels'][rank])
                assert row['score'] == float(rec['post_scores'][rank])
                possible = query_by_box[tuple(rec['post_boxes'][rank].tolist())]
                assert len(possible) == 1, (image_id, rank, possible)
                candidates.append((possible[0], int(row['category_id'])))
            assert len(set(candidates)) == 300
            assert all(predictions_image[k]['score'] >= predictions_image[k + 1]['score'] for k in range(299))
            phase_queries[image_id] = candidates
        queries.append(phase_queries)
    return records[0], queries


def interpolated_ap(tp, n_gt):
    tp = np.cumsum(tp, dtype=np.float64)
    fp = np.arange(1, len(tp) + 1) - tp
    rc = tp / n_gt
    pr = tp / (tp + fp + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    positions = np.searchsorted(rc, np.linspace(0, 1, 101), side='left')
    sampled = np.zeros(101)
    valid = positions < len(pr)
    sampled[valid] = pr[positions[valid]]
    return float(sampled.mean())


def evaluate(shared, annotation, rows, reference=None, queries=None):
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = shared.evaluate_bbox(shared.fast_ground_truth(annotation), rows)
    metrics = shared.named_metrics(evaluator.eval)
    if reference is not None:
        assert all(abs(metrics[k] - reference[k]) < 1e-12 for k in metrics), (metrics, reference)
    if queries is None:
        return metrics, None, None
    image_ids = list(evaluator.params.imgIds)
    categories = list(evaluator.params.catIds)
    thresholds = list(evaluator.params.iouThrs)
    predictions, gt = defaultdict(list), defaultdict(list)
    counter = defaultdict(int)
    for row in rows:
        image_id = int(row['image_id'])
        position = counter[image_id]
        counter[image_id] += 1
        query, category = queries[image_id][position]
        assert category == row['category_id']
        predictions[image_id, category].append(dict(row, query=query, image_rank=position + 1))
    for row in annotation['annotations']:
        assert not row.get('iscrowd', 0) and not row.get('ignore', 0)
        assert 0 < row['area'] < 1e10
        gt[int(row['image_id']), int(row['category_id'])].append(row)
    details, ap_by_category = {}, []
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    for ci, category in enumerate(categories):
        ordered_images = []
        n_gt = 0
        for image_id in image_ids:
            gts = gt[image_id, category]
            n_gt += len(gts)
            ds = predictions[image_id, category]
            indices = np.argsort([-d['score'] for d in ds], kind='mergesort')[:100]
            ds = [ds[i] for i in indices]
            ious = np.asarray(evaluator.ious[image_id, category])
            matches = np.full((len(thresholds), len(ds)), -1, dtype=np.int64)
            if len(gts) and len(ds):
                assert ious.shape == (len(ds), len(gts))
                for ti, threshold in enumerate(thresholds):
                    used = set()
                    for di in range(len(ds)):
                        best = min(threshold, 1 - 1e-10)
                        chosen = -1
                        for gi in range(len(gts)):
                            if gi not in used and ious[di, gi] >= best:
                                chosen, best = gi, ious[di, gi]
                        if chosen != -1:
                            used.add(chosen)
                            matches[ti, di] = int(gts[chosen]['id'])
            ordered_images.extend(dict(d, matches=matches[:, di]) for di, d in enumerate(ds))
        assert n_gt > 0
        order = np.argsort([-d['score'] for d in ordered_images], kind='mergesort')
        ordered = [ordered_images[i] for i in order]
        for ti, threshold in enumerate(thresholds):
            ap = interpolated_ap(np.array([r['matches'][ti] >= 0 for r in ordered]), n_gt)
            assert abs(ap - float(precision[ti, :, ci].mean())) < 1e-11, (category, threshold, ap)
            total_fp, image_fp, hits = 0, defaultdict(int), {}
            for rank, row in enumerate(ordered, 1):
                gid = int(row['matches'][ti])
                if gid >= 0:
                    hits[gid] = dict(rank=rank, query=row['query'], image_id=int(row['image_id']),
                        image_rank=row['image_rank'], score=row['score'], fp_before=total_fp,
                        same_image_fp_before=image_fp[row['image_id']],
                        other_image_fp_before=total_fp - image_fp[row['image_id']])
                else:
                    total_fp += 1
                    image_fp[row['image_id']] += 1
            details[category, ti] = hits
            ap_by_category.append(dict(category_id=int(category), iou=float(threshold), AP=ap,
                                       gt_count=n_gt, matched_gt=len(hits), unmatched_predictions=total_fp))
    return metrics, details, ap_by_category


def native_mapping(release, records, parameters, module):
    mapping = module.QueryCompetitionConditionalRecalibration(**parameters).eval()
    rows, queries = [], {}
    with torch.inference_mode():
        for image_id, record in sorted(records.items()):
            scores = mapping(record['pred_logits'][None], record['pred_boxes'][None]).sigmoid().flatten()
            values, indices = torch.topk(scores, 300, sorted=True)
            qs, labels = indices // 5, indices % 5
            boxes = absolute_boxes(record)[qs]
            xywh = boxes.clone()
            xywh[:, 2:] -= xywh[:, :2]
            queries[image_id] = list(zip(qs.tolist(), labels.tolist()))
            rows.extend(dict(image_id=image_id, category_id=label, bbox=box, score=score)
                        for label, box, score in zip(labels.tolist(), xywh.tolist(), values.tolist()))
    return rows, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.release.resolve(strict=True), args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        parser.error('Use a new output directory outside the release')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    shared = load_module(root / 'coco_test_evaluation.py', '_ccvr_coco')
    mapping_module = load_module(root / 'main_host/source/FINAL_CODE/engine/deim/competition_mapping.py', '_ccvr_mapping')
    index = read(root / 'main_host/manifests/CDW1630_THREE_SEED_RELEASE.json')
    annotation_path = 'main_host/datasets/CDW1630/annotations/instances_test2017.json'
    annotation = read(root / annotation_path)
    image_ids = {int(r['id']) for r in annotation['images']}
    assert len(image_ids) == 237 and len(annotation['annotations']) == 1071
    systems = {r['method']: r for r in index['system_runs'] if r['seed'] == 0}
    mapping_run = index['mapping_stage']
    summary, coefficients, stage_rows, increments, matching_rows, gt_rows, class_rows = {}, [], [], [], [], [], []
    inputs = dict(dataset='CDW1630', split='Test', detector_seed=0, images=237, gt=1071,
                  evaluator='faster-coco-eval 1.7.2', max_dets=[1, 10, 100], annotation=annotation_path,
                  diagnostics='COCO one-to-one matching, all areas; no score threshold or added NMS', runs={})
    for detector, full_name, calibration in (
        ('DEIM', 'DEIM+CCVR', 'main_host/checkpoints/CDW1630/B0/seed0/calibration'),
        ('GCA', 'HiLoR-DETR', 'main_host/checkpoints/CDW1630/Ours/seed0/calibration'),
    ):
        before, after = systems[detector], systems[full_name]
        references = [read(root / r['result']) for r in (before, after)]
        parameters_payload = read(root / calibration / 'qccr_params.json')
        parameters = parameters_payload['parameters']
        assert references[0]['weight_sha256'] == references[1]['weight_sha256'] == parameters_payload['source_checkpoint_sha256']
        assert before['checkpoint'] == after['checkpoint']
        coefficients.append(dict(detector=detector, seed=0, beta0=parameters['beta0'],
            beta_positive=parameters['beta_positive'], beta_negative=parameters['beta_negative'],
            kappa=parameters['duplicate_penalty'], lambda_map=parameters_payload['selected_lambda_reg']))
        raw_predictions = [read(root / r['predictions']) for r in (before, after)]
        records, queries = capture_pair(root, before['test_capture'], after['test_capture'],
                                       *raw_predictions, image_ids)
        stages = []
        for phase, run in enumerate((before, after)):
            print(detector, ('before', 'after')[phase], 'evaluating saved predictions', flush=True)
            evaluated = evaluate(shared, annotation, raw_predictions[phase], references[phase]['metrics'], queries[phase])
            stages.append(evaluated)
            class_rows.extend(dict(detector=detector, stage=('before', 'after')[phase], **r) for r in evaluated[2])
        if detector == 'GCA':
            mapped_metrics, _, _ = evaluate(shared, annotation, read(root / mapping_run['predictions']),
                                            read(root / mapping_run['result'])['metrics'])
            mapped_source = mapping_run['result']
        else:
            rows, _ = native_mapping(root, records, parameters, mapping_module)
            mapped_metrics, _, _ = evaluate(shared, annotation, rows)
            mapped_source = 'CPU FP32 mapping-only replay of the retained raw capture'
        bmetrics, ametrics = stages[0][0], stages[1][0]
        for stage, values, source in [('native', bmetrics, before['result']), ('mapping', mapped_metrics, mapped_source),
                                     ('full', ametrics, after['result'])]:
            stage_rows.append(dict(detector=detector, seed=0, stage=stage, source=source, **values))
        for label, left, right in [('mapping', bmetrics, mapped_metrics),
                                   ('value_update_and_recomputation', mapped_metrics, ametrics),
                                   ('complete_ccvr', bmetrics, ametrics)]:
            increments.append(dict(detector=detector, seed=0, stage_increment=label,
                                   **{k + '_pp': 100 * (right[k] - left[k]) for k in bmetrics}))
        for ti, threshold in enumerate(np.linspace(.5, .95, 10)):
            if ti < 5:
                continue
            bhits = {gid: row for (cat, t), values in stages[0][1].items() if t == ti for gid, row in values.items()}
            ahits = {gid: row for (cat, t), values in stages[1][1].items() if t == ti for gid, row in values.items()}
            common = set(bhits) & set(ahits)
            gained, lost = set(ahits) - set(bhits), set(bhits) - set(ahits)
            assert len(ahits) - len(bhits) == len(gained) - len(lost)
            row = dict(detector=detector, seed=0, iou=float(threshold), matched_gt_before=len(bhits),
                       matched_gt_after=len(ahits), gained_matched_gt=len(gained), lost_matched_gt=len(lost),
                       common_matched_gt=len(common), same_matched_gt_set=not (gained or lost),
                       same_query_matched_gt=sum(bhits[g]['query'] == ahits[g]['query'] for g in common))
            for field in ('rank', 'fp_before', 'same_image_fp_before', 'other_image_fp_before'):
                changes = [ahits[g][field] - bhits[g][field] for g in sorted(common)]
                row.update({field + '_delta_mean': float(np.mean(changes)),
                            field + '_delta_median': float(np.median(changes)),
                            field + '_improved_gt': sum(v < 0 for v in changes),
                            field + '_worsened_gt': sum(v > 0 for v in changes),
                            field + '_unchanged_gt': sum(v == 0 for v in changes)})
            row['AP_before'] = float(np.mean([r['AP'] for r in stages[0][2] if np.isclose(r['iou'], threshold)]))
            row['AP_after'] = float(np.mean([r['AP'] for r in stages[1][2] if np.isclose(r['iou'], threshold)]))
            row['delta_AP_pp'] = 100 * (row['AP_after'] - row['AP_before'])
            matching_rows.append(row)
            for gt in annotation['annotations']:
                gid = int(gt['id'])
                detail = dict(detector=detector, seed=0, iou=float(threshold), image_id=gt['image_id'],
                              gt_id=gid, category_id=gt['category_id'], matched_before=gid in bhits,
                              matched_after=gid in ahits)
                for phase, hits in [('before', bhits), ('after', ahits)]:
                    for field in ('query', 'image_rank', 'rank', 'score', 'fp_before',
                                  'same_image_fp_before', 'other_image_fp_before'):
                        detail[phase + '_' + field] = hits.get(gid, {}).get(field)
                gt_rows.append(detail)
        summary[detector] = dict(metrics_before=bmetrics, metrics_mapping=mapped_metrics,
                                 metrics_after=ametrics, same_raw_candidates=True)
        inputs['runs'][detector] = dict(before=before, after=after,
            mapping_parameters=calibration + '/qccr_params.json', mapping_stage_source=mapped_source)
        print(detector, 'AP', 100 * bmetrics['AP'], '->', 100 * mapped_metrics['AP'], '->', 100 * ametrics['AP'], flush=True)
    output.mkdir(parents=True)
    for filename, rows in [('coefficients.csv', coefficients), ('stage_metrics.csv', stage_rows),
                           ('stage_increments.csv', increments), ('matching.csv', matching_rows),
                           ('matched_gt.csv', gt_rows), ('class_thresholds.csv', class_rows)]:
        write_csv(output / filename, rows)
    write_json(output / 'inputs.json', inputs)
    write_json(output / 'summary.json', dict(scope=inputs, detectors=summary, matching=matching_rows))
    text = ['# CCVR seed-0 diagnostic', '',
        'CDW1630 Test: 237 images, 1,071 GT; detector seeds fixed to 0.',
        'The diagnostic uses the saved formal predictions, not independently recaptured outputs.',
        'Within each detector, the before/after raw boxes, logits, decoder features and image sizes are identical.',
        'No detector inference, training, fitting or parameter selection is performed. GPU is disabled.', '',
        'COCO matching uses faster-coco-eval 1.7.2, all areas and maxDets 1/10/100.',
        'Detailed one-to-one matching uses the evaluator\'s own IoU matrices; per-class AP at every threshold agrees with its accumulated precision.',
        'The comparison uses IoU 0.75, 0.80, 0.85, 0.90 and 0.95.', '',
        'stage_metrics.csv and class_thresholds.csv store AP/AR as fractions. Increment columns ending in _pp are percentage points.',
        'DEIM mapping-only is a CPU FP32 replay through the unchanged competition_mapping.py; the other five stage values come from saved formal predictions.',
        'Matching and ranking diagnostics use only formal before/after outputs, not the mapping-only replay.', '',
        'matched_gt.csv includes every GT, including unmatched ones. Blank ranks mean unmatched.',
        'rank is the dataset-wide rank within one category after COCO\'s per-image/per-category top-100 limit, not an image-level Top-300 rank.',
        'Negative paired rank or preceding-FP changes indicate improvement; ties preserve the original evaluator input order.',
        'These descriptive associations are not a causal decomposition of the AP increment.', '',
        'Run from outside the release:', '',
        '```bash', 'python -B main_host/analysis/ccvr/run.py --release . --output /new/external/ccvr_seed0', '```', '']
    (output / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved', output, flush=True)
    for row in matching_rows:
        print(row['detector'], f"IoU {row['iou']:.2f}", row['matched_gt_before'], '->', row['matched_gt_after'],
              'gained/lost', row['gained_matched_gt'], row['lost_matched_gt'], 'AP delta', row['delta_AP_pp'], flush=True)


if __name__ == '__main__':
    main()
