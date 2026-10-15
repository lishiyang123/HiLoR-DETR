#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${REPRO_PROJECT_ROOT:-/root/autodl-tmp/BTC_DEIM_project}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PACKAGE_DIR"
sha256sum --quiet -c PAYLOAD_MANIFEST.sha256
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" -B self_test.py

case "${1:-}" in
  --preflight) ACTION=preflight ;;
  --prepare) ACTION=prepare ;;
  --start) ACTION=start ;;
  --status) ACTION=status ;;
  --verify) ACTION=verify ;;
  --summary) ACTION=summary ;;
  *)
    echo "Usage: bash run_server.sh {--preflight|--prepare|--start|--status|--verify|--summary}" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" -u -B manager.py "$ACTION" --project "$PROJECT"
