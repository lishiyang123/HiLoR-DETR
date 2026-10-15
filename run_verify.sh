#!/usr/bin/env bash
set -euo pipefail

RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python)"

exec "$PYTHON_BIN" -B "$RELEASE/verify_release.py" verify --release "$RELEASE"
