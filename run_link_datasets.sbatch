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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/export/home/s5304639/.conda/envs/trg/bin/python}"

cd "$PROJECT_DIR"
mkdir -p logs results
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
"$PYTHON_BIN" -V
nvidia-smi

COMMAND=(
  "$PYTHON_BIN" -m jepa_compare.compare_link_prediction
  --config configs/link_comparison_all.yaml
)
# Override at submission time when only a subset is needed, for example:
# DATASET_NAMES="mooc lastfm" sbatch run_link_datasets.sbatch
if [[ -n "${DATASET_NAMES:-}" ]]; then
  read -r -a DATASETS <<< "$DATASET_NAMES"
  COMMAND+=(--datasets "${DATASETS[@]}")
fi
"${COMMAND[@]}"
