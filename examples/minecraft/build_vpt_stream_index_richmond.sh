#!/bin/bash
set -euo pipefail

VLA_REPO_ROOT=${VLA_REPO_ROOT:-$PWD}
VLA_SOURCE=${VLA_SOURCE:-/s3/profiles/data-foundations-use2-1/minecraft_vpt}
VLA_INDEX_OUTPUT=${VLA_INDEX_OUTPUT:-$VLA_SOURCE/vla_foundry_stream_index}

cd "$VLA_REPO_ROOT"
VLA_PYTHON="$VLA_REPO_ROOT/.venv/bin/python"
test -x "$VLA_PYTHON"
test -d "$VLA_SOURCE/data"
test -d "$VLA_SOURCE/splits"
test -s "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl"
test -s "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl"
export PYTHONPATH="$VLA_REPO_ROOT"

if test -s "$VLA_INDEX_OUTPUT/dataset.json"; then
  echo "Streaming index already exists: $VLA_INDEX_OUTPUT"
  exit 0
fi
"$VLA_PYTHON" -u -m vla_foundry.data.minecraft.build_stream_index \
  --data-root "$VLA_SOURCE" \
  --sources \
    "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl" \
    "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl" \
  --output "$VLA_INDEX_OUTPUT" \
  --recordings-per-shard 2
