#!/bin/bash
#SBATCH --job-name=minecraft-craftium-d29-data
#SBATCH --account=data
#SBATCH --qos=cpu_lowest
#SBATCH --partition=all
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/prepare-craftium-d29-%j.log
set -euo pipefail

export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VLA_SOURCE=/s3/profiles/scale-ai-gaming-data--use2-az3--x-s3/koustuvs/game_data/craftium_ppo_d29_12step_chunks_her
VLA_OUTPUT=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-craftium-d29-balanced

test ! -e "$VLA_OUTPUT"
exec "$PWD/.venv/bin/python" -u -m vla_foundry.data.minecraft.prepare_craftium_d29 \
  --sources \
    "$VLA_SOURCE/training_data_train.jsonl" \
    "$VLA_SOURCE/training_data_val.jsonl" \
  --output "$VLA_OUTPUT" \
  --per-class 5000 500 500 \
  --workers 32 \
  --horizon 15 \
  --stride 1 \
  --shard-samples 512
