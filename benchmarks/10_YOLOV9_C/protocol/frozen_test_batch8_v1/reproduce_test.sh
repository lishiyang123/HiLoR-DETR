#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${SOTA_PROJECT_ROOT:-$(cd "$ROOT/../../../.." && pwd)}"
MODEL_PYTHON="${YOLOV9_MODEL_PYTHON:-/root/autodl-tmp/envs/yolov9_cdw_train/bin/python}"
EVAL_PYTHON="${YOLOV9_EVAL_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"
SEED="${1:?Usage: bash reproduce_test.sh SEED OUTPUT_DIRECTORY}"
OUTPUT="${2:?Usage: bash reproduce_test.sh SEED OUTPUT_DIRECTORY}"
case "$SEED" in 0|1|2) ;; *) echo "SEED must be 0, 1 or 2" >&2; exit 2 ;; esac

TARGET="$PROJECT/benchmarks/10_YOLOV9_C"
RUN="$TARGET/runs/seed$SEED"
CHECKPOINT="$RUN/weights/best_val_ap.pt"
RESULT="$RUN/TRAINING_RESULT.json"
SOURCE="$TARGET/source/YOLOv9_official"
ANNOTATION="$PROJECT/main_host/datasets/CDW1630/annotations/instances_test2017.json"
IMAGES="$PROJECT/main_host/datasets/CDW1630/test2017"
BEST_EPOCH="$($EVAL_PYTHON -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_epoch_zero_based"])' "$RESULT")"
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 YOLOv5_AUTOINSTALL=False WANDB_MODE=disabled COMET_MODE=DISABLED
"$MODEL_PYTHON" -u -B "$ROOT/predict.py" \
  --source "$SOURCE" --source-commit 5b10b5099005a45332ebe7669f8480d5fc791a65 \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --seed "$SEED" --best-epoch-zero-based "$BEST_EPOCH" \
  --annotation "$ANNOTATION" --images "$IMAGES" \
  --protocol "$ROOT/protocol/EVALUATION_LOCK.json" --output "$OUTPUT"
"$EVAL_PYTHON" -u -B "$ROOT/evaluate.py" \
  --annotation "$ANNOTATION" --predictions "$OUTPUT/predictions.coco.json" \
  --output "$OUTPUT/metrics.json"
echo "YOLOv9-C seed$SEED frozen Test reproduction: COMPLETE / $OUTPUT"
