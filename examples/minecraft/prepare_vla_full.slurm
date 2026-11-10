#!/bin/bash
#SBATCH --job-name=minecraft-vla-full-prepare
#SBATCH --account=data
#SBATCH --qos=cpu_lowest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/prepare-vla-full-%j.log
set -euo pipefail
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_SOURCE=/s3/profiles/jepa--use2-az3--x-s3/data/minecraft_vpt
VLA_OUTPUT=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-v2-full
"$VLA_PYTHON" -u -m vla_foundry.data.minecraft.prepare_vla \
  --data-root "$VLA_SOURCE" \
  --sources \
    "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl" \
    "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl" \
  --output "$VLA_OUTPUT" \
  --all-clips \
  --resume \
  --workers 16 \
  --max-tasks-per-child 25
"$VLA_PYTHON" -u examples/minecraft/audit_discrete.py \
  --dataset "$VLA_OUTPUT" \
  --output /storage/home/ericli11/minecraft-vla-poc/d29-label-audit-full.json \
  --workers 24
