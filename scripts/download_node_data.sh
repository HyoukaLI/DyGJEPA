#!/usr/bin/env bash
# Fetch the node-classification inputs (DBLP / Tmall / Patent) from the GitHub
# release so anyone can reproduce the SG-JEPA-protocol runs without the
# original machine.
#
#   scripts/download_node_data.sh                 # all three
#   scripts/download_node_data.sh tmall patent    # subset
#   RELEASE=node-data-v2 scripts/download_node_data.sh
#
# For every dataset it downloads the raw SpikeNet files into data/raw/<dataset>/
# and, when the release also carries pre-computed DeepWalk features
# (<dataset>.npy, possibly split into <dataset>.npy.part_aa, part_ab, ... to
# stay under GitHub's 2 GB asset limit), reassembles them next to the raw
# files.  Without the .npy you must generate it (hours for Tmall, ~1 day for
# Patent):  DATASET=patent sbatch generate_node_features.sbatch
# Afterwards build the archives:
#   python scripts/prepare_dblp.py
#   python scripts/prepare_spikenet_node.py --dataset tmall
#   python scripts/prepare_spikenet_node.py --dataset patent

set -euo pipefail
REPO="${REPO:-HyoukaLI/DyGJEPA}"
RELEASE="${RELEASE:-node-data-v1}"
BASE="https://github.com/$REPO/releases/download/$RELEASE"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

declare -A RAW=(
  [dblp]="dblp.txt node2label.txt"
  [tmall]="tmall.txt node2label.txt"
  [patent]="patent_edges.json patent_nodes.json"
)
# Release assets are flat, so per-dataset files are prefixed on upload:
#   dblp_node2label.txt, tmall_node2label.txt, tmall.txt, patent_edges.json, ...
asset_name() {  # <dataset> <local file>
  case "$2" in
    node2label.txt) echo "$1_node2label.txt" ;;
    *) echo "$2" ;;
  esac
}

fetch() {  # <url> <dest>
  if [[ -s "$2" ]]; then echo "  have $2"; return 0; fi
  if [[ -s "$2.gz" ]]; then echo "  have $2.gz (shipped in the repository)"; return 0; fi
  echo "  get  $1"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$2.partial" "$1" && mv "$2.partial" "$2"
}

DATASETS=("$@"); [[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(dblp tmall patent)
for dataset in "${DATASETS[@]}"; do
  [[ -n "${RAW[$dataset]:-}" ]] || { echo "unknown dataset $dataset"; exit 1; }
  dir="data/raw/$dataset"; mkdir -p "$dir"
  echo "== $dataset -> $dir"
  for file in ${RAW[$dataset]}; do
    fetch "$BASE/$(asset_name "$dataset" "$file")" "$dir/$file"
  done
  # Pre-computed features: whole file, or split parts.
  npy="$dir/$dataset.npy"
  if [[ -s "$npy" ]]; then
    echo "  have $npy"
  elif curl -fsIL "$BASE/$dataset.npy" >/dev/null 2>&1; then
    fetch "$BASE/$dataset.npy" "$npy"
  else
    parts=()
    for suffix in aa ab ac ad ae af ag ah ai aj ak al am an ao ap; do
      url="$BASE/$dataset.npy.part_$suffix"
      curl -fsIL "$url" >/dev/null 2>&1 || break
      fetch "$url" "$npy.part_$suffix"; parts+=("$npy.part_$suffix")
    done
    if [[ ${#parts[@]} -gt 0 ]]; then
      echo "  cat ${#parts[@]} parts -> $npy"; cat "${parts[@]}" > "$npy" && rm -f "${parts[@]}"
    else
      echo "  no $dataset.npy on release $RELEASE: generate it with" \
           "DATASET=$dataset sbatch generate_node_features.sbatch"
    fi
  fi
  if [[ -s "$npy" ]]; then
    python3 - "$npy" <<'PY'
import sys, numpy as np
a = np.load(sys.argv[1], mmap_mode="r"); print(f"  {sys.argv[1]}: shape={a.shape} dtype={a.dtype}")
PY
  fi
done
echo "done. Next: python scripts/prepare_dblp.py ; python scripts/prepare_spikenet_node.py --dataset tmall|patent"
