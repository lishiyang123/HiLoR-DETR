#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CDW_PROJECT_ROOT:-/root/autodl-tmp/BTC_DEIM_project}"
PYTHON_BIN="${CDW_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

action="${1:---help}"
case "$action" in
  --preflight)       exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" preflight --project "$PROJECT" ;;
  --prepare)         exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" prepare --project "$PROJECT" ;;
  --start-fit)       exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" start-fit --project "$PROJECT" ;;
  --fit-status)      exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" fit-status --project "$PROJECT" ;;
  --verify-frozen)   exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" verify-frozen --project "$PROJECT" ;;
  --prepare-test)    exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" prepare-test --project "$PROJECT" ;;
  --start-test)      exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" start-test --project "$PROJECT" ;;
  --test-status)     exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" test-status --project "$PROJECT" ;;
  --verify)          exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" verify --project "$PROJECT" ;;
  --summary)         exec "$PYTHON_BIN" -B "$PACKAGE_DIR/manager.py" summary --project "$PROJECT" ;;
  --self-test)       exec "$PYTHON_BIN" -B "$PACKAGE_DIR/self_test.py" ;;
  *)
    cat <<'EOF'
Usage: bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh ACTION

Fit/freeze stage (Train + Val only):
  --preflight
  --prepare
  --start-fit
  --fit-status
  --verify-frozen

Reporting stage (run only after verify-frozen passes):
  --prepare-test
  --start-test
  --test-status
  --verify
  --summary

Local package check:
  --self-test

Optional environment variables:
  CDW_PROJECT_ROOT   project root (default: /root/autodl-tmp/BTC_DEIM_project)
  CDW_PYTHON         experiment Python interpreter
EOF
    exit 2
    ;;
esac
