# YOLOv10-M

Three training seeds (0/1/2), official COCO initialization, 102 epochs,
batch 8, 640 input and FP32. Model-specific SGD and augmentation settings
are in `runs/seedN/args.yaml`. The selected `weights/best_val_ap.pt` maximizes
Val AP, with the earliest epoch retained on an exact tie.

From the release root, in Python 3.11 with Torch 2.5.1 and torchvision 0.20.1:

```bash
pip install -e benchmarks/01_YOLO_SERIES/YOLOv10_M_TRAINED/source/YOLOv10_official
pip install faster-coco-eval==1.7.2
python -B benchmarks/reproduce_comparison.py --model yolov10 --seed 0 --output /your/new/yolov10_train --train
python -B benchmarks/reproduce_comparison.py --model yolov10 --seed 0 --output /your/new/yolov10_test --evaluate
python -B evaluate_test_predictions.py --predictions /your/new/yolov10_test/predictions.json --annotation main_host/datasets/CDW1630/annotations/instances_test2017.json --output /your/new/yolov10_metrics
```

Repeat with seeds 1/2. Omit `--train` for a CPU-only configuration preview.
The saved `results/test_reeval_v1/seedN/predictions.json` can be evaluated by
the same CPU metric entry without running the detector again.

Test inference uses actual batch 8 from a directory, FP32, confidence 0.001,
IoU 0.7, max 300 detections, no TTA and no rectangular batching. Each model
keeps its native postprocessing. Metrics use faster-coco-eval 1.7.2 and
maxDets 1/10/100. `results/THREE_SEED_SUMMARY.json` provides unrounded metrics
and sample standard deviations.

Source: https://github.com/THU-MIG/yolov10/tree/799ff3be47d21173bcf29b351820d4b8e955e0fe

COCO initializer: https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt

The source and derived assets retain the applicable AGPL-3.0 terms;
see `source/YOLOv10_official/LICENSE` and the root third-party license guide.
