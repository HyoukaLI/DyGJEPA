#!/usr/bin/env bash
# DyGJEPA ablation run, kept apart from the main (random-negative) run: it
# reads configs/ablation/<variant>.yaml (an overlay of link_comparison_all.yaml
# that removes one module) and writes results/ablation/<variant>/*.
# Usage mirrors run_link_datasets.sh with the variant as the first argument:
#   bash scripts/run_link_ablation.sh no_history [datasets...]
#   SEEDS="42" EPOCHS=2 bash scripts/run_link_ablation.sh no_signature canparl
#   bash scripts/run_link_ablation.sh --list          # available variants

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ABLATION_DIR="$PROJECT_DIR/configs/ablation"

if [[ "$#" -lt 1 || "$1" == "--list" || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "usage: bash scripts/run_link_ablation.sh <variant> [datasets...]" >&2
  echo "variants (configs/ablation/*.yaml):" >&2
  for file in "$ABLATION_DIR"/*.yaml; do
    echo "  $(basename "${file%.yaml}")" >&2
  done
  exit 1
fi
VARIANT="$1"
shift
CONFIG="configs/ablation/$VARIANT.yaml"
if [[ ! -f "$PROJECT_DIR/$CONFIG" ]]; then
  echo "unknown ablation variant '$VARIANT' ($CONFIG not found)" >&2
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
mkdir -p logs "results/ablation/$VARIANT"

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

# The overlay inherits every dataset entry of link_comparison_all.yaml (11+,
# several of which are not in the main table), so the datasets are always
# named explicitly; ALL_DATASETS=1 runs the whole list on purpose.
if [[ "$#" -eq 0 && "${ALL_DATASETS:-0}" != "1" ]]; then
  echo "name the datasets to ablate, e.g." >&2
  echo "  bash scripts/run_link_ablation.sh $VARIANT wikipedia uci canparl" >&2
  echo "or set ALL_DATASETS=1 to run every dataset in link_comparison_all.yaml" >&2
  exit 1
fi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config "$CONFIG"
)
if [[ "$#" -gt 0 ]]; then
  COMMAND+=(--datasets "$@")
fi
# The overlay already narrows the run to rcps_jepa; MODELS is honoured for
# symmetry with the other run scripts.
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
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
