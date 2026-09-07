#!/bin/bash
#SBATCH --job-name=jepa_link_all
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:a100_1g.20gb:1
#SBATCH --qos=work
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

cd "$PROJECT_DIR"
mkdir -p logs results
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Weights & Biases (optional).  Submit with DYGJEPA_WANDB=1 to enable:
#   DYGJEPA_WANDB=1 sbatch run_link_datasets.sbatch
# mode auto tries an online run and falls back to offline on a network-less
# node; upload those later from the login node with:
#   wandb sync wandb/offline-run-*
# (Slurm forwards the submitting environment, so DYGJEPA_WANDB=1 reaches the job.
#  It is deliberately not defaulted here, so wandb.enabled in the YAML still works.)
export WANDB_PROJECT="${WANDB_PROJECT:-dygjepa}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
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
  --config configs/link_comparison_all.yaml
)
# Narrow the run at submission time.  Each variable takes a space-separated
# list and is passed straight through to the comparison entry point:
#
#   DATASET_NAMES="mooc lastfm" sbatch run_link_datasets.sbatch
#   DATASET_NAMES="wikipedia" MODELS="rcps_jepa" sbatch run_link_datasets.sbatch
#   DATASET_NAMES="wikipedia" MODELS="rcps_jepa" SEEDS="0" EPOCHS=2 \
#     sbatch run_link_datasets.sbatch      # quick end-to-end check
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
