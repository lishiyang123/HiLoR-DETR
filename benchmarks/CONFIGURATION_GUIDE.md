# Comparison-model configurations and commands

Run from the aggregate release root. Use each model's recorded `environment/`
stack, not one globally upgraded environment. All new outputs are external.
The code below does not change the published selected weights or results.

| Model | Training evidence | Budget / precision | Initialization and selection |
|---|---|---|---|
| YOLOv8-L / YOLO11-L | official single-checkpoint records in their model directories | No CDW three-seed training claimed | Official provided checkpoints; no mean/SD across training seeds |
| YOLOv9-C | `10_YOLOV9_C/manifests/TRAINING_PROTOCOL.json` and per-seed `opt.yaml` | 102 epochs, batch 8, FP32, 640 input | `source/yolov9-c.pt`; maximum Val AP; earliest tie |
| YOLOv10-M | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv10_M_TRAINED/` | 102 epochs, batch 8, FP32, SGD | `source/yolov10m.pt`; maximum Val AP |
| YOLOv12-M turbo | per-seed `args.yaml` in `01_YOLO_SERIES/YOLOv12_M_TRAINED/` | 102 epochs, batch 8, FP32, SGD | `source/yolov12m.pt`; maximum Val AP |
| D-FINE-M | `03_D_FINE_M/configs/cdw1630_dfine_m_formal.yml` plus its include closure | 102 epochs, batch 8, FP32, EMA | Official `source/dfine_m_coco.pth`; per-seed `VAL_SELECTION.json` |
| RT-DETR-R18 | `../second_host/README.md` | 102 epochs, train batch 4, Val/Test batch 8, FP32, EMA | Official COCO initialization; maximum Val AP |
| DEIM / GCCS / complete HiLoR | main `FINAL_CODE/README.md` and release quick start | Recorded main-host profiles | Complete HiLoR reuses GCCS detector and its own CCVR fits |

## Fresh comparison training

Inspect the fully expanded recorded configuration first (no GPU or writes):

```bash
python -B benchmarks/reproduce_comparison.py --model dfine --seed 0 --output /your/new/dfine_seed0
```

In the matching environment add `--train` to execute. Model choices are
`dfine`, `yolov9`, `yolov10`, `yolov12`; repeat for seeds 1/2. Only filesystem
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

Selected YOLOv9/10/12 weights are `runs/seedN/weights/best_val_ap.pt`.
D-FINE selection may be stage1 or stage2: consult that seed's
`VAL_SELECTION.json`; do not select the last checkpoint or choose by Test AP.
Native RT uses `runs/seedN/best_val.pth` in `02_RT_DETR_R18/`.

Native RT and D-FINE saved predictions can be reevaluated without GPU:

```bash
python -B evaluate_test_predictions.py --model dfine-m --seed 0 --output /your/new/dfine_metrics
python -B evaluate_test_predictions.py --model rtdetr-r18 --seed 0 --output /your/new/rtdetr_metrics
```

For a saved YOLO COCO prediction file, use the same entry's `--predictions`
and `--annotation` arguments. YOLOv9 checkpoint inference additionally has
`10_YOLOV9_C/protocol/frozen_test_batch8_v1/CANONICAL_README.md`.
Formal Test metrics use faster-coco-eval 1.7.2 and standard maxDets 1/10/100.
Model-specific score/NMS preparation and recorded Val environments remain
explicit; a common metric definition does not mean identical postprocessing.
