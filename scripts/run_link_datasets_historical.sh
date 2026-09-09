#!/usr/bin/env bash
# Historical negative sampling run (DyGLib/EdgeBank protocol), kept apart from
# the random-negative run: it reads configs/link_comparison_all_historical.yaml
# (an overlay of link_comparison_all.yaml) and writes results/historical/*.
# Usage mirrors run_link_datasets.sh:
#   bash scripts/run_link_datasets_historical.sh [datasets...]
#   MODELS="rcps_jepa dygformer" SEEDS="0 1" bash scripts/run_link_datasets_historical.sh canparl

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
mkdir -p logs results/historical

"$PYTHON_BIN" - <<'PY'
import os
import torch

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"torch={torch.__version__} runtime_device={device}", flush=True)
if os.environ.get("REQUIRE_CUDA", "0") == "1" and device != "cuda":
    raise SystemExit("REQUIRE_CUDA=1, but this PyTorch runtime cannot use CUDA")
if device == "cuda":
    print(f"cuda_device={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
PY

if [[ ! -f data/processed/wikipedia.npz ]]; then
  echo "Missing processed datasets. Run 'git lfs pull' first." >&2
  exit 1
fi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config configs/link_comparison_all_historical.yaml
)
if [[ "$#" -gt 0 ]]; then
  COMMAND+=(--datasets "$@")
fi
if [[ -n "${MODELS:-}" ]]; then
  read -r -a SELECTED_MODELS <<< "$MODELS"
  COMMAND+=(--models "${SELECTED_MODELS[@]}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  COMMAND+=(--epochs "$EPOCHS")
fi
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SELECTED_SEEDS <<< "$SEEDS"
  COMMAND+=(--seeds "${SELECTED_SEEDS[@]}")
fi
if [[ -n "${MAX_POSITIVE_PAIRS:-}" ]]; then
  COMMAND+=(--max-positive-pairs "$MAX_POSITIVE_PAIRS")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output "$OUTPUT_DIR")
fi
if [[ -n "${OUTPUT_NAME:-}" ]]; then
  COMMAND+=(--output-name "$OUTPUT_NAME")
fi
"${COMMAND[@]}"
