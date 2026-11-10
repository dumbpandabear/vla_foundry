#!/bin/bash
#SBATCH --job-name=minecraft-qwen3-d29-balanced
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/train-d29-balanced-%j.log
set -euo pipefail
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
"$VLA_PYTHON" -u -m vla_foundry.main --config_path examples/minecraft/qwen3_discrete29_balanced.yaml
"$VLA_PYTHON" examples/minecraft/export_discrete.py \
 --run "$VLA_ROOT/runs/minecraft-qwen3-d29-balanced-poc" --audit "$VLA_ROOT/d29-label-audit.json" \
 --output "$VLA_ROOT/policy-qwen3-d29-balanced-poc"
"$VLA_PYTHON" -u examples/minecraft/evaluate_discrete.py \
 --bundle "$VLA_ROOT/policy-qwen3-d29-balanced-poc" --manifest "$VLA_ROOT/dataset-v2/test/manifest.jsonl" \
 --output "$VLA_ROOT/test-qwen3-d29-balanced-poc.json"
