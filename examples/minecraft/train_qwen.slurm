#!/bin/bash
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
VLA_PYTHON="$PWD/.venv/bin/python"
VLA_ARTIFACTS=/storage/home/ericli11/minecraft-vla-poc
VLA_EFFECTIVE_CONFIG="$VLA_ARTIFACTS/qwen-config-${SLURM_JOB_ID}.yaml"
"$VLA_PYTHON" examples/minecraft/resume_qwen.py examples/minecraft/qwen3_lora.yaml "$VLA_EFFECTIVE_CONFIG"
"$VLA_PYTHON" -u -m vla_foundry.main --config_path "$VLA_EFFECTIVE_CONFIG"
if [ ! -f "$VLA_ARTIFACTS/policy-qwen3-lora-poc/policy.json" ]; then
"$VLA_PYTHON" -m vla_foundry.inference.minecraft.export \
 --run "$VLA_ARTIFACTS/runs/minecraft-qwen3-lora-poc" --dataset "$VLA_ARTIFACTS/dataset-v2" \
 --output "$VLA_ARTIFACTS/policy-qwen3-lora-poc"
fi
if [ ! -f "$VLA_ARTIFACTS/test-qwen3-lora-poc.json" ]; then
"$VLA_PYTHON" -u -m vla_foundry.inference.minecraft.evaluate_vla \
 --bundle "$VLA_ARTIFACTS/policy-qwen3-lora-poc" --manifest "$VLA_ARTIFACTS/dataset-v2/test/manifest.jsonl" \
 --output "$VLA_ARTIFACTS/test-qwen3-lora-poc.json" --device cuda --max-samples 512 --batch-size 2
fi
# Render on the CPU while policy inference remains on the allocated H200.
export MKL_NUM_THREADS=4
export LP_NUM_THREADS=4
export SDL_VIDEODRIVER=offscreen
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
VLA_MESA_JSON="${SLURM_TMPDIR:-/tmp}/craftium-mesa-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"}}' > "$VLA_MESA_JSON"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_MESA_JSON"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
for VLA_SEED in 42 43 44; do
 if [ -f "$VLA_ARTIFACTS/rollout-qwen3-${SLURM_JOB_ID}-seed${VLA_SEED}/result.json" ]; then continue; fi
 "$VLA_PYTHON" -u /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/run_minecraft_vla.py \
  --bundle "$VLA_ARTIFACTS/policy-qwen3-lora-poc" --device cuda --steps 2000 --seed "$VLA_SEED" \
  --action-mode continuous --replan-ticks 8 \
  --output "$VLA_ARTIFACTS/rollout-qwen3-${SLURM_JOB_ID}-seed${VLA_SEED}"
done
