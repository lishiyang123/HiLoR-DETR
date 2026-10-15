#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CDW_PROJECT_ROOT:-/root/autodl-tmp/BTC_DEIM_project}"
PYTHON_BIN="${CDW_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cd "$PACKAGE_DIR"
sha256sum -c PAYLOAD_MANIFEST.sha256
"$PYTHON_BIN" -B self_test.py

case "${1:-}" in
  --preflight)     exec "$PYTHON_BIN" -B manager.py preflight --project "$PROJECT" ;;
  --prepare-test)  exec "$PYTHON_BIN" -B manager.py prepare-test --project "$PROJECT" ;;
  --start-test)    exec "$PYTHON_BIN" -B manager.py start-test --project "$PROJECT" ;;
  --test-status)   exec "$PYTHON_BIN" -B manager.py test-status --project "$PROJECT" ;;
  --verify)        exec "$PYTHON_BIN" -B manager.py verify --project "$PROJECT" ;;
  --summary)       exec "$PYTHON_BIN" -B manager.py summary --project "$PROJECT" ;;
  *)
    cat <<'EOF'
Usage: bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh ACTION

  --preflight
  --prepare-test
  --start-test
  --test-status
  --verify
  --summary

This add-on reuses the already frozen B0 checkpoints and QCCR artifacts.
It does not train a detector or fit a postprocessor.
EOF
    exit 2
    ;;
esac
