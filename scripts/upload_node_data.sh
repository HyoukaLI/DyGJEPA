#!/usr/bin/env bash
# Publish the node-classification inputs to a GitHub release so other people
# can run scripts/download_node_data.sh.  Needs the GitHub CLI, logged in
# (gh auth login).  Run from any machine that has data/raw/<dataset>/.
#
#   scripts/upload_node_data.sh                 # raw files for all three
#   WITH_NPY=1 scripts/upload_node_data.sh      # also the DeepWalk features, split into <2 GB parts
#   RELEASE=node-data-v2 scripts/upload_node_data.sh tmall
set -euo pipefail
REPO="${REPO:-HyoukaLI/DyGJEPA}"
RELEASE="${RELEASE:-node-data-v1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v gh >/dev/null || { echo "gh CLI not found (https://cli.github.com)"; exit 1; }
gh release view "$RELEASE" -R "$REPO" >/dev/null 2>&1 \
  || gh release create "$RELEASE" -R "$REPO" -t "$RELEASE" -n "Node-classification inputs (SpikeNet raw data + DeepWalk features)"

declare -A RAW=(
  [dblp]="dblp.txt node2label.txt"
  [tmall]="tmall.txt node2label.txt"
  [patent]="patent_edges.json patent_nodes.json"
)
DATASETS=("$@"); [[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(dblp tmall patent)
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
for dataset in "${DATASETS[@]}"; do
  dir="data/raw/$dataset"
  for file in ${RAW[$dataset]}; do
    asset="$file"; [[ "$file" == node2label.txt ]] && asset="${dataset}_node2label.txt"
    ln -sf "$(pwd)/$dir/$file" "$tmp/$asset"
    echo "== upload $asset"; gh release upload "$RELEASE" -R "$REPO" "$tmp/$asset" --clobber
  done
  if [[ "${WITH_NPY:-0}" == "1" && -s "$dir/$dataset.npy" ]]; then
    size=$(stat -c %s "$dir/$dataset.npy" 2>/dev/null || stat -f %z "$dir/$dataset.npy")
    if (( size < 2000000000 )); then
      echo "== upload $dataset.npy"; gh release upload "$RELEASE" -R "$REPO" "$dir/$dataset.npy" --clobber
    else
      echo "== split $dataset.npy ($size bytes) into 1900 MB parts"
      split -b 1900m -a 2 "$dir/$dataset.npy" "$tmp/$dataset.npy.part_"
      for part in "$tmp/$dataset.npy.part_"*; do
        echo "== upload $(basename "$part")"; gh release upload "$RELEASE" -R "$REPO" "$part" --clobber
      done
    fi
  fi
done
echo "done: https://github.com/$REPO/releases/tag/$RELEASE"
