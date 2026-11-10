#!/bin/bash
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_SOURCE=/s3/profiles/jepa--use2-az3--x-s3/data/minecraft_vpt
VLA_OUTPUT=/storage/home/ericli11/minecraft-vla-poc
"$VLA_PYTHON" -u -m vla_foundry.data.minecraft.prepare_vla \
  --data-root "$VLA_SOURCE" \
  --sources "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl" "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl" \
  --output "$VLA_OUTPUT/dataset-v2-smoke-r2" --max-clips 32 8 8 --workers 2
"$VLA_PYTHON" -u -m vla_foundry.data.minecraft.prepare_vla \
  --data-root "$VLA_SOURCE" \
  --sources "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl" "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl" \
  --output "$VLA_OUTPUT/dataset-v2" --workers 16
