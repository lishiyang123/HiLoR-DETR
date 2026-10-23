# YOLOv8-L

Three independent training seeds (0/1/2), using Ultralytics 8.3.221 and the
official COCO initializer. Training uses 102 epochs, batch 8, 640 input, FP32
and the recorded SGD/augmentation settings in `runs/seedN/args.yaml`.
`selection.json` identifies the maximum-Val-AP checkpoint (earliest exact tie).
Only that checkpoint is retained in each run.

From the release root, in Python 3.11:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e benchmarks/01_YOLO_SERIES/YOLOv8_L_TRAINED/source/Ultralytics_official
pip install faster-coco-eval==1.7.2
python -B benchmarks/reproduce_comparison.py --model yolov8 --seed 0 --output /your/new/v8_train --train
python -B benchmarks/reproduce_comparison.py --model yolov8 --seed 0 --output /your/new/v8_test --evaluate
python -B evaluate_test_predictions.py --predictions /your/new/v8_test/predictions.json --annotation main_host/datasets/CDW1630/annotations/instances_test2017.json --output /your/new/v8_metrics
```

Repeat with seeds 1/2. Omit `--train` for a CPU-only configuration preview.
Saved predictions can be evaluated without GPU:

```bash
python -B evaluate_test_predictions.py --model yolov8-l --seed 0 --output /your/new/v8_saved_metrics
```

Test inference uses directory-backed batch 8, FP32, 640 input, confidence 0.001,
native NMS IoU 0.7 and at most 300 detections per image, without TTA or rectangular
batching. Metrics use faster-coco-eval 1.7.2, maxDets 1/10/100.
`results/THREE_SEED_SUMMARY.json` contains unrounded metrics and sample SD.

Source: https://github.com/ultralytics/ultralytics/tree/v8.3.221

COCO initialization: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt

The bundled source and derived model assets retain the applicable Ultralytics
AGPL-3.0 terms; see `source/Ultralytics_official/LICENSE` and the release license guide.
