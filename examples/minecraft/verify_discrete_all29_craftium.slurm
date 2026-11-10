#!/bin/bash
#SBATCH --job-name=minecraft-d29-all29-craftium
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/craftium-d29-all29-%j.log
set -euo pipefail

export PYTHONPATH="$PWD"
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-all29"
VLA_DATA=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-craftium-d29-balanced

"$PWD/.venv/bin/python" -u examples/minecraft/gate_discrete_all29.py \
  --evaluation "$VLA_ROOT/test-qwen3-d29-all29.json" \
  --policy "$VLA_BUNDLE/policy.json" \
  --audit "$VLA_DATA/d29_label_audit.json" \
  --training-log "$VLA_ROOT/runs/minecraft-qwen3-d29-all29/out.log" \
  --output "$VLA_ROOT/d29-all29-quality-gate.json"

VLA_ROLLOUTS=()
while IFS='|' read -r VLA_TASK VLA_GOAL; do
  VLA_OUTPUT="$VLA_ROOT/rollout-d29-all29-${VLA_TASK}"
  VLA_ROLLOUTS+=("$VLA_OUTPUT")
  if test -e "$VLA_OUTPUT"; then
    VLA_INCOMPLETE="${VLA_OUTPUT}-incomplete-${SLURM_JOB_ID}"
    test ! -e "$VLA_INCOMPLETE"
    mv "$VLA_OUTPUT" "$VLA_INCOMPLETE"
  fi
  srun examples/minecraft/run_craftium.slurm \
    --bundle "$VLA_BUNDLE" \
    --goal "$VLA_GOAL" \
    --task "$VLA_TASK" \
    --steps 512 \
    --seed 42 \
    --action-mode discrete_29 \
    --replan-ticks 4 \
    --output "$VLA_OUTPUT"
done <<'EOF'
craft_planks_from_log|Craft wooden planks from the log in inventory.
craft_crafting_table_from_planks|Craft a crafting table from wooden planks.
craft_stick_from_crafting_table|Craft sticks using the available materials.
craft_wooden_pickaxe_from_stick|Craft a wooden pickaxe.
craft_stone_pickaxe_from_cobblestone|Craft a stone pickaxe.
craft_furnace_from_iron_ore|Craft a furnace.
craft_iron_ingot_from_furnace|Smelt an iron ingot in the furnace.
craft_iron_pickaxe_from_iron_ingot|Craft an iron pickaxe.
EOF

"$PWD/.venv/bin/python" -u examples/minecraft/summarize_all29_rollouts.py \
  --rollouts "${VLA_ROLLOUTS[@]}" \
  --output "$VLA_ROOT/craftium-d29-all29-summary.json"
