#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

cd "$PROJECT_DIR"
for specification in \
  "CanParl:canparl" \
  "Contacts:contacts" \
  "Flights:flights" \
  "UNtrade:untrade" \
  "UNvote:unvote" \
  "USLegis:uslegis" \
  "enron:enron" \
  "uci:uci"
do
  raw_name="${specification%%:*}"
  output_name="${specification##*:}"
  "$PYTHON_BIN" scripts/prepare_dyglib_homogeneous.py \
    --dataset-dir "data/raw/$raw_name" \
    --output "data/processed/$output_name.npz"
done
