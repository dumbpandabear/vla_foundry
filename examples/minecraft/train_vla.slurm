#!/bin/bash
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_ARTIFACTS=/storage/home/ericli11/minecraft-vla-poc
"$VLA_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
"$VLA_PYTHON" -u -m vla_foundry.main --config_path examples/minecraft/vla_v2.yaml
"$VLA_PYTHON" -m vla_foundry.inference.minecraft.export \
  --run "$VLA_ARTIFACTS/runs/minecraft-vla-v2" --dataset "$VLA_ARTIFACTS/dataset-v2" --output "$VLA_ARTIFACTS/policy-v2"
"$VLA_PYTHON" -u -m vla_foundry.inference.minecraft.evaluate_vla \
  --bundle "$VLA_ARTIFACTS/policy-v2" --manifest "$VLA_ARTIFACTS/dataset-v2/test/manifest.jsonl" \
  --output "$VLA_ARTIFACTS/test-v2.json" --device cuda
