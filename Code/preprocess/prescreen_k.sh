#!/usr/bin/env bash
# Prescreen k: тонкий раннер поверх prescreen_k.py
# Використання:
#   bash Code/prescreen_k.sh --config ./project.yaml --split auto --k-grid 1.5,2.0,2.5,3.0,3.5 --use-cache auto --target-clip 0.02
# Порада: додай виконувані права: chmod +x Code/prescreen_k.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/prescreen_k.py"

if [[ ! -f "${PY}" ]]; then
  echo "prescreen_k.py not found at ${PY}" >&2
  exit 1
fi

python "${PY}" "$@"
