#!/usr/bin/env bash
# Noise-robustness run (DyG-Mamba §5.5), kept apart from the main run: it reads
# configs/link_comparison_all_robustness.yaml (an overlay of
# link_comparison_all.yaml) and writes results/robustness/link_robustness_<ds>.json,
# .csv and figures/<ds>_noise_ap.{pdf,png}.  Datasets are named explicitly.
#   bash scripts/run_link_robustness.sh wikipedia
#   MODELS="rcps_jepa tgn" RATES="0 0.3 0.6" bash scripts/run_link_robustness.sh uci
#   EPOCHS=1 MODELS="rcps_jepa dvgmae" bash scripts/run_link_robustness.sh uci   # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="configs/link_comparison_all_robustness.yaml"

if [[ "$#" -eq 0 ]]; then
  echo "name the datasets, e.g. bash scripts/run_link_robustness.sh wikipedia" >&2
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
mkdir -p logs results/robustness

"$PYTHON_BIN" - <<'PY'
import os
import torch

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"torch={torch.__version__} runtime_device={device}", flush=True)
if os.environ.get("REQUIRE_CUDA", "0") == "1" and device != "cuda":
    raise SystemExit("REQUIRE_CUDA=1, but this PyTorch runtime cannot use CUDA")
PY

if [[ ! -f data/processed/wikipedia.npz ]]; then
  echo "Missing processed datasets. Run 'git lfs pull' first." >&2
  exit 1
fi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_robustness
  --config "$CONFIG" --datasets "$@"
)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a SELECTED_MODELS <<< "$MODELS"
  COMMAND+=(--models "${SELECTED_MODELS[@]}")
fi
if [[ -n "${RATES:-}" ]]; then
  read -r -a SELECTED_RATES <<< "$RATES"
  COMMAND+=(--rates "${SELECTED_RATES[@]}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  COMMAND+=(--epochs "$EPOCHS")
fi
if [[ -n "${SEED:-}" ]]; then
  COMMAND+=(--seed "$SEED")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output "$OUTPUT_DIR")
fi
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
