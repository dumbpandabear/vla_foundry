#!/bin/bash
# Submit with sbatch --qos=h200_dev --partition=h200 --gpus=1 --cpus-per-task=8 --mem=32G --time=00:30:00.
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_ARTIFACTS=/storage/home/ericli11/minecraft-vla-poc
"$VLA_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "A GPU allocation is required"; print(torch.cuda.get_device_name())'
"$VLA_PYTHON" -u -m vla_foundry.main --config_path examples/minecraft/poc.yaml
"$VLA_PYTHON" -m vla_foundry.inference.minecraft.export \
  --run "$VLA_ARTIFACTS/runs/minecraft-vla-poc-v1" --dataset "$VLA_ARTIFACTS/dataset-v1" --output "$VLA_ARTIFACTS/policy-v1"
"$VLA_PYTHON" -m vla_foundry.inference.minecraft.evaluate \
  --bundle "$VLA_ARTIFACTS/policy-v1" --manifest "$VLA_ARTIFACTS/dataset-v1/test/manifest.jsonl" \
  --output "$VLA_ARTIFACTS/test-v1.json" --device cuda
