# Official YOLO single-checkpoint Test reproduction

This is the canonical reviewer-facing Test code for YOLOv8-L and YOLO11-L.

- official frozen checkpoint, one checkpoint per model;
- CDW1630 frozen Test split;
- 640x640 FP32 native Ultralytics inference;
- requested and verified effective batch size 8;
- confidence 0.001, NMS IoU 0.7, maximum 300 detections;
- identity class-ID mapping with the frozen display-name aliases
  `PlasticBottle`/`Plastic Bottle` and `Timber`/`timber`;
- `faster-coco-eval==1.7.2`, bbox, maxDets 1/10/100;
- no training, fitting, hyperparameter selection, or Test-time checkpoint selection.

Run from this directory with a new output path:

```bash
bash run_test.sh YOLOv8_L_OFFICIAL /new/output/yolov8
bash run_test.sh YOLOv11_L_OFFICIAL /new/output/yolo11
```

