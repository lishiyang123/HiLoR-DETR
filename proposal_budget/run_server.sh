#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CDW_PROJECT_ROOT:-/root/autodl-tmp/BTC_DEIM_project}"
PYTHON="${CDW_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"
ACTION="${1:---preflight}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: missing executable Python: $PYTHON" >&2
  exit 1
fi

cd "$PACKAGE_DIR"
sha256sum -c PAYLOAD_MANIFEST.sha256
"$PYTHON" -B self_test.py
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "$ACTION" in
  --preflight|--prepare|--run|--verify|--summary|--status)
    exec "$PYTHON" -u -B manager.py "${ACTION#--}" --project "$PROJECT"
    ;;
  *)
    echo "Usage: bash run_server.sh {--preflight|--prepare|--run|--verify|--summary|--status}" >&2
    exit 2
    ;;
esac
