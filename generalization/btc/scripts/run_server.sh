#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${BTC_REPRO_PYTHON:-${PYTHON:-$(command -v python3 || command -v python)}}"
exec "$PY" -u -B "$SCRIPT_DIR/manager.py" "$@"
