#!/usr/bin/env bash
# Regression check for the DyGJEPA module-ablation switches.
#
#   1. unit tests (ablation switches, overlay configs, DyGJEPA/link protocol);
#   2. A/B: the random-negative main protocol must be bit-identical between
#      git HEAD and the working tree (all switches default to the full model);
#   3. smoke: two ablation overlays run end to end and change the result.
#
# Everything runs on CPU so the comparison is deterministic.  Usage:
#   bash scripts/check_rcps_ablation.sh              # uci, seed 42, 1 epoch
#   DATASET=canparl bash scripts/check_rcps_ablation.sh
# Takes a few minutes; results go to a temporary directory that is printed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-uci}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rcps_ablation_check.XXXXXX")"
HEAD_TREE="$WORK/head"
echo "work dir: $WORK"
cleanup() {
  cd "$PROJECT_DIR"
  git worktree remove --force "$HEAD_TREE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"
echo "== 1/3 unit tests"
"$PYTHON_BIN" -m pytest -q \
  tests/test_rcps_ablation.py \
  tests/test_rcps_jepa.py \
  tests/test_historical_negatives.py \
  tests/test_multi_dataset_config.py

echo "== 2/3 A/B: HEAD vs working tree on the random protocol ($DATASET, seed $SEED, $EPOCHS epoch(s), cpu)"
git worktree add --detach "$HEAD_TREE" HEAD >/dev/null
ln -s "$PROJECT_DIR/data" "$HEAD_TREE/data"

run_tree() {
  local tree="$1" config="$2" out="$3"
  (
    cd "$tree"
    PYTHONHASHSEED=42 "$PYTHON_BIN" -m jepa_compare.compare_link_prediction \
      --config "$config" --datasets "$DATASET" --models rcps_jepa \
      --seeds "$SEED" --epochs "$EPOCHS" --output "$out" > "$out.log" 2>&1
  )
}

cat > "$WORK/main_cpu.yaml" <<EOF
base_config: $PROJECT_DIR/configs/link_comparison_all.yaml
device: cpu
EOF
mkdir -p "$WORK/head_out" "$WORK/work_out"
run_tree "$HEAD_TREE" "$WORK/main_cpu.yaml" "$WORK/head_out"
run_tree "$PROJECT_DIR" "$WORK/main_cpu.yaml" "$WORK/work_out"

"$PYTHON_BIN" - "$WORK/head_out/link_comparison_$DATASET.json" "$WORK/work_out/link_comparison_$DATASET.json" <<'PY'
import json
import sys

head, work = (json.load(open(path)) for path in sys.argv[1:3])
# Wall-clock / memory records differ by construction; compare the metrics.
head = {m: {k: v for k, v in r.items() if k != "efficiency"} for m, r in head.items()}
work = {m: {k: v for k, v in r.items() if k != "efficiency"} for m, r in work.items()}
if head != work:
    print("MISMATCH between HEAD and working tree:")
    print(json.dumps(head, indent=1)[:2000])
    print(json.dumps(work, indent=1)[:2000])
    raise SystemExit(1)
print("bit-identical:", json.dumps(work["rcps_jepa"]["test"]))
PY

echo "== 3/3 smoke: ablation overlays change the result"
for variant in no_history prior_only; do
  cat > "$WORK/${variant}_cpu.yaml" <<EOF
base_config: $PROJECT_DIR/configs/ablation/$variant.yaml
device: cpu
EOF
  mkdir -p "$WORK/${variant}_out"
  run_tree "$PROJECT_DIR" "$WORK/${variant}_cpu.yaml" "$WORK/${variant}_out"
  grep -m1 '"ablation"' "$WORK/${variant}_out.log" || true
done

"$PYTHON_BIN" - "$WORK" "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

work, dataset = Path(sys.argv[1]), sys.argv[2]
full = json.load(open(work / "work_out" / f"link_comparison_{dataset}.json"))["rcps_jepa"]
for variant in ("no_history", "prior_only"):
    result = json.load(open(work / f"{variant}_out" / f"link_comparison_{dataset}.json"))["rcps_jepa"]
    assert result["test"] != full["test"], f"{variant} did not change the test result"
    if variant == "prior_only":
        assert result["test"]["best_epoch"] == 0.0, result["test"]
    print(variant, json.dumps(result["test"]))
print("ablation smoke OK")
PY
echo "all checks passed"
