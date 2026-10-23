# Comparison-model configurations and commands

Run from the aggregate release root. Use each model's recorded `environment/`
stack, not one globally upgraded environment. All new outputs are external.
The code below does not change the published selected weights or results.

| Model | Training evidence | Budget / precision | Initialization and selection |
|---|---|---|---|
| YOLOv8-L | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv8_L_TRAINED/` | 102 epochs, batch 8, FP32, 640 input | [`source/yolov8l.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt) (download separately); maximum Val AP; earliest tie |
| YOLO11-L | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv11_L_TRAINED/` | 102 epochs, batch 8, FP32, 640 input | [`source/yolo11l.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) (download separately); maximum Val AP; earliest tie |
| YOLOv9-C | `10_YOLOV9_C/manifests/TRAINING_PROTOCOL.json` and per-seed `opt.yaml` | 102 epochs, batch 8, FP32, 640 input | [`source/yolov9-c.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) (download separately); maximum Val AP; earliest tie |
| YOLOv10-M | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv10_M_TRAINED/` | 102 epochs, batch 8, FP32, SGD | [`source/yolov10m.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt) (download separately); maximum Val AP |
| YOLOv12-M turbo | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv12_M_TRAINED/` | 102 epochs, batch 8, FP32, SGD | [`source/yolov12m.pt`](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt) (download separately); maximum Val AP |
| D-FINE-M | `03_D_FINE_M/configs/cdw1630_dfine_m_formal.yml` plus its include closure | 102 epochs, batch 8, FP32, EMA | Official [`source/dfine_m_coco.pth`](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth) (download separately); per-seed `VAL_SELECTION.json` |
| RT-DETR-R18 | `../second_host/README.md` | 102 epochs, train batch 4, Val/Test batch 8, FP32, EMA | [`source/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) (download separately); maximum Val AP |
| DEIM / GSA / complete HiLoR | `../main_host/source/FINAL_CODE/README.md` and release quick start | Recorded main-host profiles | Complete HiLoR reuses GSA detector and its own CCVR fits |

## Fresh comparison training

Inspect the fully expanded recorded configuration first (no GPU or writes):

```bash
python -B benchmarks/reproduce_comparison.py --model dfine --seed 0 --output /your/new/dfine_seed0
```

In the matching environment add `--train` to execute. Model choices are
`dfine`, `yolov8`, `yolov11`, `yolov9`, `yolov10`, `yolov12`; repeat for seeds 1/2. Only filesystem
paths, output location and requested seed are materialized; detector training
settings come from the recorded configuration. YOLO training explicitly keeps
AMP off and saves strict improvements in Val AP to `weights/best_val_ap.pt`,
not the upstream composite-fitness `best.pt`. Train/Val images are checked
before execution; Test is not loaded for selection.

```bash
python -B benchmarks/reproduce_comparison.py --model yolov10 --seed 0 --output /your/new/yolov10_seed0 --train
python -B second_host/train.py --method native --seed 0 --workdir /your/new/rtdetr_seed0
```

The planner and configuration checks are CPU-tested. New full comparison
trainings are not automatically executed during package verification.

## Paper checkpoint and saved-result reproduction

Selected YOLOv8/9/10/11/12 weights are `runs/seedN/weights/best_val_ap.pt`.
D-FINE selection may be stage1 or stage2: consult that seed's
`VAL_SELECTION.json`; do not select the last checkpoint or choose by Test AP.
Native RT uses `runs/seedN/best_val.pth` in `02_RT_DETR_R18/`.

Native RT and D-FINE saved predictions can be reevaluated without GPU:

```bash
python -B evaluate_test_predictions.py --model dfine-m --seed 0 --output /your/new/dfine_metrics
python -B evaluate_test_predictions.py --model rtdetr-r18 --seed 0 --output /your/new/rtdetr_metrics
python -B evaluate_test_predictions.py --model yolov9-c --seed 0 --output /your/new/yolov9_metrics
python -B evaluate_test_predictions.py --model yolov8-l --seed 0 --output /your/new/yolov8_metrics
```

For a saved YOLO COCO prediction file, use the same entry's `--predictions`
and `--annotation` arguments. YOLOv9 checkpoint inference additionally has
`10_YOLOV9_C/protocol/frozen_test_batch8_v1/CANONICAL_README.md`.
Formal Test metrics use faster-coco-eval 1.7.2 and standard maxDets 1/10/100.
Model-specific score/NMS preparation and recorded Val environments remain
explicit; a common metric definition does not mean identical postprocessing.

D-FINE and all five YOLO checkpoint entries use the selected per-seed weights:

```bash
python -B benchmarks/reproduce_comparison.py --model dfine --seed 0 --output /your/new/dfine_test --evaluate
python -B benchmarks/reproduce_comparison.py --model yolov9 --seed 0 --output /your/new/yolov9_test --evaluate
python -B benchmarks/reproduce_comparison.py --model yolov8 --seed 0 --output /your/new/yolov8_test --evaluate
python -B benchmarks/reproduce_comparison.py --model yolov11 --seed 0 --output /your/new/yolov11_test --evaluate
python -B benchmarks/reproduce_comparison.py --model yolov10 --seed 0 --output /your/new/yolov10_test --evaluate
python -B benchmarks/reproduce_comparison.py --model yolov12 --seed 0 --output /your/new/yolov12_test --evaluate
```

Use the matching model environment; repeat for seeds 1 and 2. Their current
three-seed Test summaries are stored in their model-specific `results/` folders.
