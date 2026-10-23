"""COCO subset evaluation and per-class metric extraction."""

from contextlib import redirect_stdout
import io
from pathlib import Path

import numpy as np


def mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1]
    if valid.size == 0:
        raise ValueError('No valid COCO entries')
    return float(valid.mean())


def index_exact(values, target):
    found = np.flatnonzero(np.isclose(np.asarray(values, dtype=float), float(target)))
    if len(found) != 1:
        raise ValueError('Missing or repeated COCO threshold: ' + str(target))
    return int(found[0])


def overall(evaluation):
    params = evaluation['params']
    area = list(params.areaRngLbl).index('all')
    maximum = list(map(int, params.maxDets)).index(100)
    precision = np.asarray(evaluation['precision'])[:, :, :, area, maximum]
    return dict(AP=mean_valid(precision),
                AP50=mean_valid(precision[index_exact(params.iouThrs, .5)]),
                AP75=mean_valid(precision[index_exact(params.iouThrs, .75)]),
                AR100=mean_valid(np.asarray(evaluation['recall'])[:, :, area, maximum]))


def per_class(evaluation, categories):
    params = evaluation['params']
    expected = [int(category['id']) for category in categories]
    if list(map(int, params.catIds)) != expected:
        raise ValueError('COCO category order differs from the annotation')
    area = list(params.areaRngLbl).index('all')
    maximum = list(map(int, params.maxDets)).index(100)
    iou75 = index_exact(params.iouThrs, .75)
    precision, recall = np.asarray(evaluation['precision']), np.asarray(evaluation['recall'])
    return [dict(category_id=int(category['id']),
                 category_name='Timber' if category['name'] == 'timber' else category['name'],
                 AP=100*mean_valid(precision[:, :, i, area, maximum]),
                 AP75=100*mean_valid(precision[iou75, :, i, area, maximum]),
                 AR100=100*mean_valid(recall[:, i, area, maximum]))
            for i, category in enumerate(categories)]


def evaluate(annotation, predictions, image_ids):
    from faster_coco_eval import COCO, COCOeval_faster
    selected = set(map(int, image_ids))
    detections = [row for row in predictions if int(row['image_id']) in selected]
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(Path(annotation)))
        dt = gt.loadRes(detections)
        evaluator = COCOeval_faster(gt, iouType='bbox',
                                   print_function=lambda *_args, **_kwargs: None,
                                   separate_eval=True)
        evaluator.cocoDt = dt
        evaluator.params.imgIds = sorted(selected)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    result = dict(evaluator.eval, params=evaluator.params)
    return overall(result), result
