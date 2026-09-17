#!/usr/bin/env bash
# Efficiency run (one seed, every model, one device), kept apart from the main
# run: it reads configs/link_comparison_all_efficiency.yaml (an overlay of
# link_comparison_all.yaml) and writes results/efficiency/*.  Each model's
# result carries an ``efficiency`` block (parameters, seconds per epoch, time to
# the selected checkpoint, test inference time, peak GPU memory).
# Usage mirrors run_link_datasets.sh; datasets are always named explicitly:
#   bash scripts/run_link_efficiency.sh enron wikipedia
#   MODELS="rcps_jepa dygformer" bash scripts/run_link_efficiency.sh enron
#   python scripts/summarize_efficiency.py --results results/efficiency --plot
# Run all models of a dataset in ONE invocation on ONE device so the timings
# are comparable; do not split the models over different GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="configs/link_comparison_all_efficiency.yaml"

if [[ "$#" -eq 0 && "${ALL_DATASETS:-0}" != "1" ]]; then
  echo "name the datasets to time, e.g." >&2
  echo "  bash scripts/run_link_efficiency.sh enron wikipedia" >&2
  echo "or set ALL_DATASETS=1 to run every dataset in link_comparison_all.yaml" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

cd "$PROJECT_DIR"
mkdir -p logs results/efficiency

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
else:
    print("note: peak_memory_mb is only recorded on CUDA", flush=True)
PY

if [[ ! -f data/processed/wikipedia.npz ]]; then
  echo "Missing processed datasets. Run 'git lfs pull' first." >&2
  exit 1
fi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config "$CONFIG"
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
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output "$OUTPUT_DIR")
fi
if [[ -n "${OUTPUT_NAME:-}" ]]; then
  COMMAND+=(--output-name "$OUTPUT_NAME")
fi
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
