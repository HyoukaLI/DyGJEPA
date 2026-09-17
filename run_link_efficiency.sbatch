#!/bin/bash
#SBATCH --job-name=jepa_link_eff
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:a100_1g.20gb:1
#SBATCH --qos=work
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Efficiency run: one seed, EVERY model of a dataset trained and evaluated
# sequentially in this one job on this one GPU, so the recorded wall-clock and
# peak-memory numbers are comparable.  Same environment and knobs as
# run_link_datasets.sbatch, but its own job name, log files, config
# (configs/link_comparison_all_efficiency.yaml, an overlay of the random
# config) and result tree (results/efficiency/), so it never touches the main
# results.  DATASET_NAMES is required; submit one job per dataset and keep the
# same --gres for all of them:
#
#   DATASET_NAMES="enron" sbatch run_link_efficiency.sbatch
#   for d in enron wikipedia mooc; do DATASET_NAMES=$d sbatch run_link_efficiency.sbatch; done
#   python scripts/summarize_efficiency.py --results results/efficiency --plot

set -euo pipefail
module purge
module load anaconda3/2024.02

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REQUIRE_CUDA=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/export/home/s5304639/.conda/envs/trg/bin/python}"
CONFIG="configs/link_comparison_all_efficiency.yaml"

cd "$PROJECT_DIR"
mkdir -p logs results/efficiency
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Weights & Biases (optional).  Submit with DYGJEPA_WANDB=1 to enable; see
# run_link_datasets.sbatch for the offline-sync workflow.
export WANDB_PROJECT="${WANDB_PROJECT:-dygjepa}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local} efficiency run"
"$PYTHON_BIN" -V
nvidia-smi

"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the link-dataset Slurm job")
print("cuda_device", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config "$CONFIG"
)
# The overlay inherits every dataset entry of link_comparison_all.yaml, so
# DATASET_NAMES is required unless ALL_DATASETS=1 is set on purpose.
#   DATASET_NAMES="canparl" EPOCHS=2 sbatch run_link_efficiency.sbatch   # quick check
# MODELS is accepted but defeats the purpose unless every model of the dataset
# still ends up timed on the same device.
if [[ -z "${DATASET_NAMES:-}" && "${ALL_DATASETS:-0}" != "1" ]]; then
  echo "DATASET_NAMES is required, e.g. DATASET_NAMES=\"enron\" sbatch run_link_efficiency.sbatch" >&2
  echo "(set ALL_DATASETS=1 to run every dataset in link_comparison_all.yaml)" >&2
  exit 1
fi
if [[ -n "${DATASET_NAMES:-}" ]]; then
  read -r -a DATASETS <<< "$DATASET_NAMES"
  COMMAND+=(--datasets "${DATASETS[@]}")
fi
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_NAMES <<< "$MODELS"
  COMMAND+=(--models "${MODEL_NAMES[@]}")
fi
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_VALUES <<< "$SEEDS"
  COMMAND+=(--seeds "${SEED_VALUES[@]}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  COMMAND+=(--epochs "$EPOCHS")
fi

printf '[%s] command:' "$(date --iso-8601=seconds)"
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
