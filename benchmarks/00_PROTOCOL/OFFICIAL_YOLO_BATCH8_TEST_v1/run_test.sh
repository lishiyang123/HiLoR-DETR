#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_ROOT/../.." && pwd)"
PROJECT_ROOT="${SOTA_PROJECT_ROOT:-$(cd "$PACKAGE_ROOT/.." && pwd)}"
YOLO_PYTHON="${YOLO_SOTA_PYTHON:-/root/autodl-tmp/envs/yolo_sota_eval/bin/python}"
EVAL_PYTHON="${DEIM_EVAL_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"

MODEL_KEY="${1:-}"
OUTPUT="${2:-}"
case "$MODEL_KEY" in
  YOLOv8_L_OFFICIAL)
    WEIGHT="$PACKAGE_ROOT/01_YOLO_SERIES/OFFICIAL_ASSETS_INBOX/YOLOv8l.pt" ;;
  YOLOv11_L_OFFICIAL)
    WEIGHT="$PACKAGE_ROOT/01_YOLO_SERIES/OFFICIAL_ASSETS_INBOX/YOLOv11l.pt" ;;
  *)
    echo "Usage: bash run_test.sh {YOLOv8_L_OFFICIAL|YOLOv11_L_OFFICIAL} NEW_OUTPUT_DIR" >&2
    exit 2 ;;
esac
if [[ -z "$OUTPUT" || -e "$OUTPUT" ]]; then
  echo "Output must be a new, absent directory: $OUTPUT" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export YOLO_AUTOINSTALL=false

"$YOLO_PYTHON" -u -B "$SCRIPT_ROOT/predict.py" \
  --model-key "$MODEL_KEY" \
  --weight "$WEIGHT" \
  --annotation "$PROJECT_ROOT/main_host/datasets/CDW1630/annotations/instances_test2017.json" \
  --images "$PROJECT_ROOT/main_host/datasets/CDW1630/test2017" \
  --protocol "$SCRIPT_ROOT/PROTOCOL.json" \
  --output "$OUTPUT"

"$EVAL_PYTHON" -u -B "$SCRIPT_ROOT/evaluate.py" \
  --annotation "$PROJECT_ROOT/main_host/datasets/CDW1630/annotations/instances_test2017.json" \
  --predictions "$OUTPUT/predictions.coco.json" \
  --output "$OUTPUT/metrics.json"

echo "Official YOLO batch-8 Test reproduction complete: $OUTPUT"
