#!/bin/bash
#SBATCH --job-name=jepa_node_all
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=work
#SBATCH --time=48:00:00
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

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
"$PYTHON_BIN" -V
nvidia-smi

"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the node-dataset Slurm job")
print("cuda_device", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

COMMAND=(bash scripts/run_node_datasets.sh)
if [[ -n "${DATASET_NAMES:-}" ]]; then
  read -r -a DATASETS <<< "$DATASET_NAMES"
  COMMAND+=("${DATASETS[@]}")
fi
"${COMMAND[@]}"
