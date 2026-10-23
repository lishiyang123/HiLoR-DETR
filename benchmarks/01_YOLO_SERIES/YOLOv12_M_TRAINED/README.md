# YOLOv12-M (turbo)

Three training seeds (0/1/2), official COCO initialization, 102 epochs,
batch 8, 640 input and FP32. Model-specific SGD and augmentation settings
are in `runs/seedN/args.yaml`. The selected `weights/best_val_ap.pt` maximizes
Val AP, with the earliest epoch retained on an exact tie.

From the release root, in Python 3.11 with Torch 2.5.1 and torchvision 0.20.1:

```bash
pip install -e benchmarks/01_YOLO_SERIES/YOLOv12_M_TRAINED/source/YOLOv12_official
pip install faster-coco-eval==1.7.2
python -B benchmarks/reproduce_comparison.py --model yolov12 --seed 0 --output /your/new/yolov12_train --train
python -B benchmarks/reproduce_comparison.py --model yolov12 --seed 0 --output /your/new/yolov12_test --evaluate
python -B evaluate_test_predictions.py --predictions /your/new/yolov12_test/predictions.json --annotation main_host/datasets/CDW1630/annotations/instances_test2017.json --output /your/new/yolov12_metrics
```

Repeat with seeds 1/2. Omit `--train` for a CPU-only configuration preview.
The saved `results/test_reeval_v1/seedN/predictions.json` can be evaluated by
the same CPU metric entry without running the detector again.

Test inference uses actual batch 8 from a directory, FP32, confidence 0.001,
IoU 0.7, max 300 detections, no TTA and no rectangular batching. Each model
keeps its native postprocessing. Metrics use faster-coco-eval 1.7.2 and
maxDets 1/10/100. `results/THREE_SEED_SUMMARY.json` provides unrounded metrics
and sample standard deviations.

Source: https://github.com/sunsmarterjie/yolov12/tree/28a618bb06f38eef5402ae8f7326e86063db93de

COCO initializer: https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt

The source and derived assets retain the applicable AGPL-3.0 terms;
see `source/YOLOv12_official/LICENSE` and the root third-party license guide.
