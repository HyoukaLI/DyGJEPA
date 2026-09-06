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

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(dblp tmall patent)
fi

CANONICAL_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    tsmall|Tsmall|Tmall) canonical="tmall" ;;
    DBLP) canonical="dblp" ;;
    Patent) canonical="patent" ;;
    *) canonical="$dataset" ;;
  esac
  if [[ ! -f "data/processed/${canonical}.npz" ]]; then
    echo "Missing data/processed/${canonical}.npz; see README node-data preparation." >&2
    exit 1
  fi
  CANONICAL_DATASETS+=("$canonical")
done

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_node_prediction
  --config configs/node_comparison_all.yaml
  --datasets "${CANONICAL_DATASETS[@]}"
)
if [[ -n "${BASELINES+x}" ]]; then
  if [[ -n "$BASELINES" ]]; then
    read -r -a SELECTED_BASELINES <<< "$BASELINES"
    COMMAND+=(--baselines "${SELECTED_BASELINES[@]}")
  else
    COMMAND+=(--baselines)
  fi
fi
if [[ -n "${EPOCHS:-}" ]]; then
  COMMAND+=(--epochs "$EPOCHS")
fi
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SELECTED_SEEDS <<< "$SEEDS"
  COMMAND+=(--seeds "${SELECTED_SEEDS[@]}")
fi
"${COMMAND[@]}"
