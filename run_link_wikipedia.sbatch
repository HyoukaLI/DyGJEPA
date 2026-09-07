#!/bin/bash
#
#SBATCH --job-name=jepa_link_wiki
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128G
##SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --gres=gpu:a100_1g.20gb:1
##SBATCH --gres=gpu:a100_3g.40gb:1
##SBATCH --gres=gpu:a100:1
#SBATCH --qos=work
#SBATCH --time=48:00:00
#SBATCH --output=/export/home/s5304639/slurm/DYGGPEA/logs/%x-%j.out
#SBATCH --error=/export/home/s5304639/slurm/DYGGPEA/logs/%x-%j.err

set -euo pipefail
module purge
module load anaconda3/2024.02

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/export/home/s5304639/slurm/DYGGPEA
ENV_PY=/export/home/s5304639/.conda/envs/trg/bin/python
CONFIG="$PROJECT_DIR/configs/link_comparison_wikipedia.yaml"
DATA="$PROJECT_DIR/data/processed/wikipedia.npz"
OUTPUT="$PROJECT_DIR/results/link_comparison_wikipedia.json"

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
echo "config=$CONFIG"
echo "data=$DATA"
echo "output=$OUTPUT"
nvidia-smi
"$ENV_PY" -V
"$ENV_PY" - <<'PY'
import torch
import yaml

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("yaml", yaml.__version__)
with open("configs/link_comparison_wikipedia.yaml") as handle:
    config = yaml.safe_load(handle)
print(
    "protocol",
    config["split"],
    config["link"],
    "jodie",
    config["jodie"],
    config["jodie_training"],
    "dyrep",
    config["dyrep"],
    config["dyrep_training"],
    "tgat",
    config["tgat"],
    config["tgat_training"],
    "additional",
    config["additional_baselines"],
    {name: config[name] for name in config["additional_baselines"]["enabled"] if name != "edgebank"},
    "rcps",
    config["rcps_training"],
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Wikipedia link comparison")
PY

for required in "$CONFIG" "$DATA"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required file: $required"
    exit 1
  fi
done

"$ENV_PY" -m jepa_compare.compare_link_prediction --config "$CONFIG"

echo "[$(date --iso-8601=seconds)] completed"
cat "$OUTPUT"
