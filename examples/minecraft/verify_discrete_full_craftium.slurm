#!/bin/bash
#SBATCH --job-name=minecraft-d29-full-craftium
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/craftium-d29-full-%j.log
set -euo pipefail
export PYTHONPATH="$PWD"
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-full"

"$PWD/.venv/bin/python" -u examples/minecraft/gate_discrete_full.py \
  --evaluation "$VLA_ROOT/test-qwen3-d29-full.json" \
  --policy "$VLA_BUNDLE/policy.json" \
  --training-log "$VLA_ROOT/runs/minecraft-qwen3-d29-full/out.log" \
  --output "$VLA_ROOT/d29-full-quality-gate.json"

for VLA_SEED in 42 43 44; do
  VLA_OUTPUT="$VLA_ROOT/rollout-d29-full-seed${VLA_SEED}"
  if test -s "$VLA_OUTPUT/result.json"; then
    echo "Verified rollout already exists: $VLA_OUTPUT/result.json"
    continue
  fi
  if test -e "$VLA_OUTPUT"; then
    VLA_INCOMPLETE="${VLA_OUTPUT}-incomplete-${SLURM_JOB_ID}"
    test ! -e "$VLA_INCOMPLETE"
    mv "$VLA_OUTPUT" "$VLA_INCOMPLETE"
    echo "Moved incomplete rollout to $VLA_INCOMPLETE"
  fi
  srun examples/minecraft/run_craftium.slurm \
    --bundle "$VLA_BUNDLE" \
    --goal "Mine stone blocks using a pickaxe." \
    --task mine_cobblestone \
    --steps 2000 \
    --seed "$VLA_SEED" \
    --action-mode discrete_29 \
    --output "$VLA_OUTPUT"
done

"$PWD/.venv/bin/python" -u examples/minecraft/summarize_craftium_rollouts.py \
  --rollouts \
    "$VLA_ROOT/rollout-d29-full-seed42" \
    "$VLA_ROOT/rollout-d29-full-seed43" \
    "$VLA_ROOT/rollout-d29-full-seed44" \
  --expected-steps 2000 \
  --output "$VLA_ROOT/craftium-d29-full-summary.json"
