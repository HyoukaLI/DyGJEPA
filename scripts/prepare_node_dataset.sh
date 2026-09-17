#!/usr/bin/env bash
# One-shot preparation of a node-classification dataset for a fresh clone:
#   1. raw files      - Tmall ships in the repository (tmall.txt.gz + node2label);
#                       DBLP / Patent are fetched from the GitHub release;
#   2. DeepWalk feats - <dataset>.npy is fetched from the release (split parts are
#                       reassembled); if the release has none, GENERATE=1 runs the
#                       official recipe locally (hours for Tmall, ~1 day Patent);
#   3. archive        - data/processed/<dataset>.npz is (re)built whenever it is
#                       missing or still carries the 4-D structural fallback.
# Afterwards `bash scripts/run_node_datasets.sh <dataset>` runs on the local GPU.
#
#   bash scripts/prepare_node_dataset.sh tmall
#   GENERATE=1 bash scripts/prepare_node_dataset.sh tmall     # no .npy on the release
#   RELEASE=node-data-v2 bash scripts/prepare_node_dataset.sh patent
# Needs network for the release downloads: run it on a login node, not inside
# a Slurm job on a network-less compute node.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else PYTHON_BIN="$(command -v python3 || command -v python)"; fi
fi
export PYTHON_BIN

if [[ "$#" -ne 1 ]]; then echo "usage: bash scripts/prepare_node_dataset.sh dblp|tmall|patent" >&2; exit 1; fi
case "$1" in
  dblp|DBLP) dataset=dblp ;;
  tmall|tsmall|Tmall|Tsmall) dataset=tmall ;;
  patent|Patent) dataset=patent ;;
  *) echo "unknown dataset $1 (dblp|tmall|patent)" >&2; exit 1 ;;
esac
raw_dir="data/raw/$dataset"
npy="$raw_dir/$dataset.npy"
npz="data/processed/$dataset.npz"

have_raw() {
  case "$dataset" in
    dblp)   [[ -s $raw_dir/dblp.txt && -s $raw_dir/node2label.txt ]] ;;
    tmall)  { [[ -s $raw_dir/tmall.txt ]] || [[ -s $raw_dir/tmall.txt.gz ]]; } && [[ -s $raw_dir/node2label.txt ]] ;;
    patent) [[ -s $raw_dir/patent_edges.json && -s $raw_dir/patent_nodes.json ]] ;;
  esac
}

# 1 + 2: raw files and DeepWalk features from the release (skips what exists).
if ! have_raw || [[ ! -s "$npy" ]]; then
  echo "== fetching $dataset inputs from the GitHub release"
  bash scripts/download_node_data.sh "$dataset"
fi
have_raw || { echo "raw files for $dataset are missing under $raw_dir" >&2; exit 1; }

if [[ ! -s "$npy" ]]; then
  if [[ "${GENERATE:-0}" == "1" ]]; then
    echo "== no $dataset.npy on the release; generating the DeepWalk features locally (official recipe)"
    "$PYTHON_BIN" scripts/generate_spikenet_deepwalk.py --dataset "$dataset"
  else
    echo "no $npy: the release carries no DeepWalk features for $dataset." >&2
    echo "Re-run with GENERATE=1 (hours for Tmall, ~1 day for Patent; needs pip install -e '.[features]')," >&2
    echo "or DATASET=$dataset sbatch generate_node_features.sbatch on the cluster." >&2
    exit 1
  fi
fi

# 3: the archive, rebuilt unless it already carries DeepWalk features.
needs_build=1
if [[ -s "$npz" ]]; then
  "$PYTHON_BIN" - "$npz" <<'PY' && needs_build=0 || true
import sys, zipfile, numpy as np
with zipfile.ZipFile(sys.argv[1]) as z:
    ok = "feature_source.npy" in z.namelist() and str(np.load(z.open("feature_source.npy"), allow_pickle=True)) != "structural-fallback"
    if ok:
        # An archive built from an unfinished (pre-allocated) .npy has all-zero
        # features; the first snapshot is enough to tell, and only its bytes
        # are decompressed.
        with z.open("features.npy") as handle:
            version = np.lib.format.read_magic(handle)
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_header(handle)
            first = np.frombuffer(handle.read(int(np.prod(shape[1:])) * dtype.itemsize), dtype=dtype)
        ok = bool(np.any(first))
raise SystemExit(0 if ok else 1)
PY
fi
if [[ "$needs_build" == "1" ]]; then
  echo "== building $npz from $npy"
  if [[ "$dataset" == "dblp" ]]; then
    "$PYTHON_BIN" scripts/prepare_dblp.py
  else
    "$PYTHON_BIN" scripts/prepare_spikenet_node.py --dataset "$dataset"
  fi
else
  echo "== $npz already carries DeepWalk features"
fi
echo "ready: bash scripts/run_node_datasets.sh $dataset"
