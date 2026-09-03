#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

cd "$PROJECT_DIR"
mkdir -p logs results

if [[ ! -f data/processed/wikipedia.npz ]]; then
  echo "Missing processed datasets. Run 'git lfs pull' first." >&2
  exit 1
fi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config configs/link_comparison_all.yaml
)
if [[ "$#" -gt 0 ]]; then
  COMMAND+=(--datasets "$@")
fi
"${COMMAND[@]}"
