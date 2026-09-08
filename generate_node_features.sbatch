#!/bin/bash
#SBATCH --job-name=jepa_deepwalk
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=work
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generate the SpikeNet DeepWalk node features data/raw/<dataset>/<dataset>.npy
# ([T, N, 80]) that the SG-JEPA paper protocol requires, then rebuild the NPZ.
#
#   DATASET=tmall  sbatch generate_node_features.sbatch
#   DATASET=patent sbatch generate_node_features.sbatch
#   DATASET=dblp   sbatch generate_node_features.sbatch
#
# The job is resumable: finished snapshots are recorded next to the .npy, so
# re-submitting after a time-out continues with the missing ones.  Walk corpora
# are streamed to SCRATCH_DIR (Patent: ~25 GB per snapshot, deleted after use).
# Requires the feature extras: pip install -e '.[features]'  (gensim numba scipy tqdm)

set -euo pipefail
module purge
module load anaconda3/2024.02

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/export/home/s5304639/.conda/envs/trg/bin/python}"
DATASET="${DATASET:-tmall}"
SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-$PROJECT_DIR/data/raw/$DATASET/walks}}"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local} dataset=$DATASET scratch=$SCRATCH_DIR"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -c "import gensim, numba, scipy; print('gensim', gensim.__version__, 'numba', numba.__version__)"
df -h "$SCRATCH_DIR" | tail -1

"$PYTHON_BIN" scripts/generate_spikenet_deepwalk.py \
  --dataset "$DATASET" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --scratch-dir "$SCRATCH_DIR" \
  ${EXTRA_ARGS:-}

# Rebuild the processed archive with the features (auto-detected next to the raw data).
if [[ "$DATASET" == "dblp" ]]; then
  "$PYTHON_BIN" scripts/prepare_dblp.py
else
  "$PYTHON_BIN" scripts/prepare_spikenet_node.py --dataset "$DATASET"
fi
"$PYTHON_BIN" - "$DATASET" <<'PY'
import sys
import numpy as np

archive = np.load(f"data/processed/{sys.argv[1]}.npz", allow_pickle=True)
print("features", archive["features"].shape, "feature_source", archive["feature_source"])
PY
