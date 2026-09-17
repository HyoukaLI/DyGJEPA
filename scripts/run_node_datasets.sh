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
    if [[ "${AUTO_PREPARE:-1}" == "1" ]]; then
      echo "== data/processed/${canonical}.npz missing: preparing it (scripts/prepare_node_dataset.sh)"
      bash scripts/prepare_node_dataset.sh "$canonical"
    else
      echo "Missing data/processed/${canonical}.npz; run bash scripts/prepare_node_dataset.sh ${canonical}." >&2
      exit 1
    fi
  fi
  CANONICAL_DATASETS+=("$canonical")
done

# Refuse to launch on the 4-D structural fallback archives: every model then
# collapses to the majority class and the numbers are not comparable with the
# SG-JEPA / SpikeNet protocol.  With AUTO_PREPARE=1 (default) such archives are
# rebuilt with the DeepWalk features first; DYGJEPA_ALLOW_STRUCTURAL_FALLBACK=1
# runs the fallback protocol on purpose.
check_archives() {
"$PYTHON_BIN" - "${CANONICAL_DATASETS[@]}" <<'GUARD'
import os
import sys
import zipfile

import numpy as np

bad = []
for name in sys.argv[1:]:
    path = f"data/processed/{name}.npz"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        source = (
            str(np.load(archive.open("feature_source.npy"), allow_pickle=True))
            if "feature_source.npy" in members
            else "unknown"
        )
        with archive.open("features.npy") as handle:
            version = np.lib.format.read_magic(handle)
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_header(handle)
            first = np.frombuffer(
                handle.read(int(np.prod(shape[1:])) * dtype.itemsize), dtype=dtype
            )
    empty = not bool(np.any(first))
    print(f"{path}: features {shape}, feature_source={source}"
          + (", FIRST SNAPSHOT ALL ZERO" if empty else ""), flush=True)
    if source == "structural-fallback" or empty:
        bad.append(path)
if bad and os.environ.get("DYGJEPA_ALLOW_STRUCTURAL_FALLBACK", "0") != "1":
    print(
        "structural-fallback archives: " + ", ".join(bad) + "\n"
        "rebuild them with the DeepWalk features (python scripts/prepare_spikenet_node.py "
        "--dataset <name>; scripts/prepare_dblp.py for DBLP) or set "
        "DYGJEPA_ALLOW_STRUCTURAL_FALLBACK=1 to run the fallback protocol on purpose.",
        file=sys.stderr,
    )
    raise SystemExit(1)
GUARD
}
if ! check_archives; then
  if [[ "${AUTO_PREPARE:-1}" == "1" && "${DYGJEPA_ALLOW_STRUCTURAL_FALLBACK:-0}" != "1" ]]; then
    for canonical in "${CANONICAL_DATASETS[@]}"; do
      bash scripts/prepare_node_dataset.sh "$canonical"
    done
    check_archives || exit 1
  else
    exit 1
  fi
fi

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
# Per-job knobs, same spelling as the link launcher:
#   MODELS="cawn"                -> --models (sg_jepa / rcps_jepa / one or more baselines)
#   SEED=42 or SEEDS="42 44"     -> --seeds
#   TRAIN_RATIO=0.6              -> --train-ratio (probe split; result stem gets _ratio0.6)
#   OUTPUT_DIR=results/tmall     -> --output      (directory of the result files)
#   OUTPUT_NAME=cawn             -> --output-name (file stem; each seed writes
#                                   <OUTPUT_DIR>/<OUTPUT_NAME>[_ratio<r>]_seed<s>.json)
#   e.g. MODELS=cawn SEED=42 TRAIN_RATIO=0.4 OUTPUT_DIR=results/tmall OUTPUT_NAME=cawn \
#          bash scripts/run_node_datasets.sh tmall
if [[ -n "${SEEDS:-}" || -n "${SEED:-}" ]]; then
  read -r -a SELECTED_SEEDS <<< "${SEEDS:-$SEED}"
  COMMAND+=(--seeds "${SELECTED_SEEDS[@]}")
fi
if [[ -n "${MODELS:-}" ]]; then
  read -r -a SELECTED_MODELS <<< "$MODELS"
  COMMAND+=(--models "${SELECTED_MODELS[@]}")
fi
if [[ -n "${TRAIN_RATIO:-}" ]]; then
  COMMAND+=(--train-ratio "$TRAIN_RATIO")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output "$OUTPUT_DIR")
fi
if [[ -n "${OUTPUT_NAME:-}" ]]; then
  COMMAND+=(--output-name "$OUTPUT_NAME")
fi

printf '[%s] command:' "$(date +%Y-%m-%dT%H:%M:%S)"
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
