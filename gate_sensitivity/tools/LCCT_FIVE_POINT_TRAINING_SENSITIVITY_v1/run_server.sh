#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FINAL_PAPER_RELEASE_ROOT:-$(cd -- "$PACKAGE_DIR/../../.." && pwd)}"
PY="${LCCT_REPRO_PYTHON:-${PYTHON:-$(command -v python3 || command -v python)}}"
if [[ ! -x "$PY" ]]; then
  echo "Python environment missing: $PY" >&2
  exit 1
fi

cd "$PACKAGE_DIR"
sha256sum -c PAYLOAD_MANIFEST.sha256
"$PY" -B self_test.py
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${1:-}" in
  --preflight) exec "$PY" -u -B manager.py preflight --project "$PROJECT" ;;
  --prepare)   exec "$PY" -u -B manager.py prepare --project "$PROJECT" ;;
  --audit)     exec "$PY" -u -B manager.py audit --project "$PROJECT" ;;
  --smoke)     exec "$PY" -u -B manager.py smoke --project "$PROJECT" ;;
  --start)     exec "$PY" -u -B manager.py start --project "$PROJECT" ;;
  --run)       exec "$PY" -u -B manager.py run --project "$PROJECT" ;;
  --resume)    exec "$PY" -u -B manager.py resume --project "$PROJECT" ;;
  --status)    exec "$PY" -u -B manager.py status --project "$PROJECT" ;;
  --summary)   exec "$PY" -u -B manager.py summary --project "$PROJECT" ;;
  --verify)    exec "$PY" -u -B manager.py verify --project "$PROJECT" ;;
  *)
    echo "Usage: bash run_server.sh {--preflight|--prepare|--audit|--smoke|--start|--run|--resume|--status|--summary|--verify}" >&2
    exit 2
    ;;
esac
