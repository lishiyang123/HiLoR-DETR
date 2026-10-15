#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CDW_PROJECT_ROOT:-$(cd -- "$PACKAGE_DIR/.." && pwd)}"
PYTHON_BIN="${CDW_DEIM_PYTHON:-/root/autodl-tmp/envs/deim_cvpr2025/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'DEIM Python environment missing: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

cd "$PACKAGE_DIR"
sha256sum -c PAYLOAD_MANIFEST.sha256
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -u -B self_test.py
exec "$PYTHON_BIN" -u -B manager.py --project "$PROJECT" "$@"

