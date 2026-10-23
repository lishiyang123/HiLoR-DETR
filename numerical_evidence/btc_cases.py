"""Recompute the selected BTC figure numbers from saved final detections.

This is an offline numerical check, not a plotting or aggregate-AP program.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path


def iou(a, b):
    x, y, w, h = a
    u, v, s, t = b
    inter = max(0., min(x+w, u+s)-max(x, u)) * max(0., min(y+h, v+t)-max(y, v))
    union = w*h+s*t-inter
    return inter/union if union > 0 else 0.


def load_case(root, tag, method):
    path = root / 'evidence/btc_predictions' / (tag+'_'+method+'.json')
    record = json.loads(path.read_text())
    assert len(record['detections']) == 300, path
    for d in record['detections']:
        assert d['image_id'] == record['image']['id']
        assert len(d['bbox']) == 4 and all(math.isfinite(x) for x in d['bbox'])
        assert all(x >= 0 for x in d['bbox'][2:])
        assert math.isfinite(d['score']) and 0 <= d['score'] <= 1
    return record


def selected_pair(root, tag, annotation_id, rule):
    result = {'tag': tag, 'annotation_id': annotation_id, 'selection_rule': rule}
    reference = None
    for method in ['deim', 'hilor']:
        data = load_case(root, tag, method)
        gt = next(a for a in data['ground_truth'] if a['id'] == annotation_id)
        if reference is None:
            reference = (data['image'], gt, data['image_sha256'])
            result.update(image=data['image'], image_sha256=data['image_sha256'],
                          category=next(c['name'] for c in data['categories']
                                        if c['id'] == gt['category_id']),
                          gt_bbox=gt['bbox'])
        assert reference == (data['image'], gt, data['image_sha256'])
        detections = sorted(enumerate(data['detections']), key=lambda x: -x[1]['score'])
        same = [dict(d, rank=rank, candidate_index=index,
                     iou=iou(d['bbox'], gt['bbox']))
                for rank, (index, d) in enumerate(detections, 1)
                if d['category_id'] == gt['category_id']]
        visible = [d for d in same if d['score'] >= .30]
        if rule == 'highest_score_valid':
            eligible = [d for d in visible if d['iou'] >= .75]
            best = max(eligible, key=lambda d: d['score'])
        else:
            best = max(visible, key=lambda d: d['iou'])
        result[method] = best
        result[method+'_max_iou_all_final_top300_same_class'] = max(d['iou'] for d in same)
    return result


def compute(root):
    protocol = json.loads((root/'BTC_CASE_PROTOCOL.json').read_text())
    intro = dict(figure='F1', dataset='BTC', detector_seed=0,
                 pairs=[selected_pair(root, x['tag'], x['annotation_id'], x['rule'])
                        for x in protocol['F1']])
    for pair in intro['pairs']:
        assert pair['deim']['iou'] < .75 <= pair['hilor']['iou']
        assert pair['hilor']['score'] > pair['deim']['score']
    qualitative = dict(figure='F7', dataset='BTC', detector_seed=0, cases=[])
    for item in protocol['F7']:
        case = selected_pair(root, item['tag'], item['annotation_id'], item['rule'])
        case['role'] = item['role']
        qualitative['cases'].append(case)
    ranking, crossing, limitation = qualitative['cases']
    assert min(ranking[m]['iou'] for m in ['deim', 'hilor']) >= .75
    assert ranking['hilor']['rank'] < ranking['deim']['rank']
    assert crossing['deim']['iou'] < .75 <= crossing['hilor']['iou']
    assert all(limitation[m+'_max_iou_all_final_top300_same_class'] < .75
               for m in ['deim', 'hilor'])
    background = load_case(root, protocol['F2']['tag'], 'hilor')
    framework = dict(figure='F2', role='illustrative_detection_background', dataset='BTC',
                     image=background['image'], image_sha256=background['image_sha256'],
                     display_threshold=.90, ground_truth_filtering=False,
                     extra_nms=False, training_assignment_capture=False,
                     query_ids_or_assignment_ious_reported=False,
                     detections=[d for d in background['detections'] if d['score'] >= .90])
    framework['displayed_detection_count'] = len(framework['detections'])
    return {
        'evidence/introduction_comparison/INTRO_FIGURE_EVIDENCE.json': intro,
        'evidence/qualitative_cases/QUALITATIVE_CASE_EVIDENCE.json': qualitative,
        'evidence/framework_example/DETECTION_BACKGROUND.json': framework}


def metric_csv(computed):
    stream = io.StringIO(newline='')
    fields = ['figure', 'role', 'tag', 'annotation_id', 'category', 'method',
              'score', 'iou', 'rank', 'max_iou_all_final_top300_same_class']
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for path, record in computed.items():
        for case in record.get('pairs', record.get('cases', [])):
            for method in ['deim', 'hilor']:
                writer.writerow(dict(figure=record['figure'], role=case.get('role', 'paired_localization'),
                                     tag=case['tag'], annotation_id=case['annotation_id'],
                                     category=case['category'], method=method,
                                     **{k: case[method][k] for k in ['score', 'iou', 'rank']},
                                     max_iou_all_final_top300_same_class=
                                     case[method+'_max_iou_all_final_top300_same_class']))
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='New directory outside the release; otherwise verify only')
    args = parser.parse_args()
    root = args.root.resolve()
    computed = compute(root)
    csv_path = 'evidence/qualitative_cases/CASE_METADATA.csv'
    if args.output:
        output = args.output.resolve()
        release = root.parent
        if output == release or release in output.parents:
            parser.error('Use a new output directory outside the release')
        output.mkdir(parents=True, exist_ok=False)
        for relative, record in computed.items():
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2)+'\n')
        (output/csv_path).write_text(metric_csv(computed))
    else:
        for relative, record in computed.items():
            assert json.loads((root/relative).read_text()) == record, relative
        assert (root/csv_path).read_text() == metric_csv(computed), csv_path
        print('BTC figure numerical evidence: PASS (10 paired rows; F2 detection background)')
        print('No training, inference, plotting or aggregate metric changes.')


if __name__ == '__main__':
    main()
