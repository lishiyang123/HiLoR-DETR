"""Shared COCO bbox evaluator for formal Test and RT training/Val entries.

Use explicit imports rather than monkey-patching pycocotools globally.
COCO input records, class IDs, box coordinates and detection order are preserved.
"""
import copy
import importlib.metadata
import json
from pathlib import Path

EVALUATOR_VERSION = '1.7.2'


def evaluator_classes():
    from faster_coco_eval import COCO, COCOeval_faster
    version = importlib.metadata.version('faster-coco-eval')
    if version != EVALUATOR_VERSION:
        raise RuntimeError(f'Expected faster-coco-eval {EVALUATOR_VERSION}; found {version}')
    return COCO, COCOeval_faster


def fast_ground_truth(dataset):
    COCO, _ = evaluator_classes()
    gt = COCO()
    gt.dataset = copy.deepcopy(dataset)
    gt.createIndex()
    return gt


def evaluate_bbox(gt, predictions, image_ids=None):
    COCO, Eval = evaluator_classes()
    # loadRes mutates its list arguments in both implementations.
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = dict(images=copy.deepcopy(gt.dataset['images']),
                          categories=copy.deepcopy(gt.dataset['categories']), annotations=[])
        dt.createIndex()
    evaluator = Eval(gt, dt, 'bbox')
    evaluator.params.imgIds = sorted(gt.getImgIds() if image_ids is None else image_ids)
    evaluator.params.catIds = sorted(gt.getCatIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def faster_bbox_evaluator(native_type):
    """Retain the host's record preparation and use the pinned bbox evaluator."""
    class FasterBBoxEvaluator(native_type):
        def __init__(self, coco_gt, iou_types):
            if list(iou_types) != ['bbox']:
                raise ValueError('This adapter supports bbox evaluation only')
            self.coco_gt = fast_ground_truth(coco_gt.dataset)
            self.iou_types = ['bbox']
            self.coco_eval = {}
            self.img_ids = []
            self._records_by_image = {}

        def update(self, predictions):
            records = self.prepare_for_coco_detection(predictions)
            grouped = {int(image_id): [] for image_id in predictions}
            for record in records:
                grouped[int(record['image_id'])].append(record)
            for image_id, rows in grouped.items():
                self._records_by_image.setdefault(image_id, rows)

        def synchronize_between_processes(self):
            import torch.distributed as dist
            shards = [self._records_by_image]
            if dist.is_available() and dist.is_initialized():
                shards = [None] * dist.get_world_size()
                dist.all_gather_object(shards, self._records_by_image)
            joined = {}
            for shard in shards:
                for image_id, rows in shard.items():
                    # Same first-image policy as the native distributed merger.
                    joined.setdefault(image_id, rows)
            self._records_by_image = joined
            self.img_ids = sorted(joined)

        def records(self):
            return [row for image_id in sorted(self._records_by_image)
                    for row in self._records_by_image[image_id]]

        def accumulate(self):
            self.coco_eval['bbox'] = evaluate_bbox(self.coco_gt, self.records(), self.img_ids)

        def summarize(self):
            self.coco_eval['bbox'].summarize()

    return FasterBBoxEvaluator


def save_predictions(evaluator, output):
    path = Path(output) / 'predictions.json'
    path.write_text(json.dumps(evaluator.records(), separators=(',', ':'), allow_nan=False) + '\n')
    return path


def named_metrics(evaluation):
    import numpy as np
    p = evaluation['params']
    precision, recall = evaluation['precision'], evaluation['recall']
    def mean(values):
        selected = values[values > -1]
        return float(selected.mean()) if selected.size else -1.
    result = {}
    for name, area, threshold in [('AP', 'all', None), ('AP50', 'all', .5), ('AP75', 'all', .75),
                                 ('APs', 'small', None), ('APm', 'medium', None), ('APl', 'large', None),
                                 ('AP80', 'all', .8), ('AP85', 'all', .85), ('AP90', 'all', .9), ('AP95', 'all', .95)]:
        values = precision[:, :, :, p.areaRngLbl.index(area), p.maxDets.index(100)]
        if threshold is not None:
            values = values[np.isclose(p.iouThrs, threshold)]
        result[name] = mean(values)
    for name, area, budget in [('AR1', 'all', 1), ('AR10', 'all', 10), ('AR100', 'all', 100),
                               ('ARs', 'small', 100), ('ARm', 'medium', 100), ('ARl', 'large', 100)]:
        result[name] = mean(recall[:, :, p.areaRngLbl.index(area), p.maxDets.index(budget)])
    return result
