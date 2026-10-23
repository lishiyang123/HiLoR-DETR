"""CPU-only faster-coco-eval Test replay of saved COCO bbox predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from coco_test_evaluation import EVALUATOR_VERSION, evaluate_bbox, fast_ground_truth, named_metrics

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--model', choices=('rtdetr-r18', 'dfine-m', 'yolov8-l', 'yolo11-l', 'yolov9-c', 'yolov10-m', 'yolov12-m'), help='Use the stored SOTA Test predictions')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model:
        if args.predictions or args.annotation or args.seed is None:
            parser.error('--model requires --seed and does not combine with custom input paths')
        folder, filename = {
            'rtdetr-r18': ('02_RT_DETR_R18/results/test_reeval_v1', 'predictions.json'),
            'dfine-m': ('03_D_FINE_M/results/test_reeval_v1_1', 'predictions.json'),
            'yolov8-l': ('01_YOLO_SERIES/YOLOv8_L_TRAINED/results/test_reeval_v1', 'predictions.json'),
            'yolo11-l': ('01_YOLO_SERIES/YOLOv11_L_TRAINED/results/test_reeval_v1', 'predictions.json'),
            'yolov10-m': ('01_YOLO_SERIES/YOLOv10_M_TRAINED/results/test_reeval_v1', 'predictions.json'),
            'yolov12-m': ('01_YOLO_SERIES/YOLOv12_M_TRAINED/results/test_reeval_v1', 'predictions.json'),
            'yolov9-c': ('10_YOLOV9_C/results/test_batch8_v1', 'predictions.coco.json'),
        }[args.model]
        args.predictions = ROOT / 'benchmarks' / folder / f'seed{args.seed}/{filename}'
        args.annotation = ROOT / 'main_host/datasets/CDW1630/annotations/instances_test2017.json'
    elif not args.predictions or not args.annotation:
        parser.error('Provide --model/--seed or both --predictions and --annotation')
    prediction_path = args.predictions.resolve(strict=True)
    annotation_path = args.annotation.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        parser.error('Use a new output directory outside the release')
    predictions = json.loads(prediction_path.read_text())
    dataset = json.loads(annotation_path.read_text())
    ids = {int(row['id']) for row in dataset['images']}
    cats = {int(row['id']) for row in dataset['categories']}
    if any(int(row['image_id']) not in ids or int(row['category_id']) not in cats for row in predictions):
        raise ValueError('Prediction image/category IDs disagree with the annotation')
    evaluator = evaluate_bbox(fast_ground_truth(dataset), predictions)
    evaluator.summarize()
    metrics = named_metrics(evaluator.eval)
    output.mkdir(parents=True)
    (output / 'METRICS.json').write_text(json.dumps(dict(
        metrics=metrics, evaluator='faster-coco-eval', evaluator_version=EVALUATOR_VERSION,
        annotation_sha256=digest(annotation_path), predictions_sha256=digest(prediction_path),
        image_count=len(ids), detection_count=len(predictions),
        scope='saved Test prediction replay; no new inference, fitting or selection',
    ), indent=2) + '\n')
    import numpy as np
    np.savez_compressed(output / 'coco_bbox_arrays.npz',
                        **{key: evaluator.eval[key] for key in ('precision', 'recall', 'scores')})
    print('SAVED TEST REPLAY: COMPLETE; original results unchanged')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
