#!/bin/bash
#SBATCH --job-name=minecraft-qwen3-d29-full
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/train-d29-full-%j.log
set -euo pipefail
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_RUN="$VLA_ROOT/runs/minecraft-qwen3-d29-full"
VLA_DATA=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-v2-full
VLA_FINAL="$VLA_ROOT/test-qwen3-d29-full.json"
VLA_AUDIT="$VLA_ROOT/d29-label-audit-full.json"
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-full"

# Dependency-chained stages become harmless no-ops once a prior stage finishes.
if test -s "$VLA_FINAL"; then
  echo "Full evaluation already exists: $VLA_FINAL"
  exit 0
fi
test -s "$VLA_DATA/dataset.json"
test -s "$VLA_AUDIT"
VLA_TOTAL_SAMPLES=$(jq -er '.samples.train | select(. > 0)' "$VLA_DATA/dataset.json")
VLA_CLASS_WEIGHTS=$("$PWD/.venv/bin/python" examples/minecraft/compute_class_weights.py --audit "$VLA_AUDIT")

VLA_RESUME=$(find "$VLA_RUN/checkpoints" -maxdepth 1 -name 'checkpoint_*.pt' -printf '%f\n' 2>/dev/null \
  | sort -V | tail -n 1 || true)
if test -n "$VLA_RESUME"; then
  VLA_RESUME="$VLA_RUN/checkpoints/$VLA_RESUME"
  VLA_WEIGHTS_ONLY=false
else
  VLA_RESUME=$(jq -er '.checkpoint' "$VLA_ROOT/policy-qwen3-d29-balanced-poc/policy.json")
  VLA_WEIGHTS_ONLY=true
fi

"$PWD/.venv/bin/python" -m torch.distributed.run --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
  --config_path examples/minecraft/qwen3_discrete29_full.yaml \
  --total_train_samples "$VLA_TOTAL_SAMPLES" \
  --model.resume_from_checkpoint "$VLA_RESUME" \
  --model.resume_weights_only "$VLA_WEIGHTS_ONLY" \
  --hparams.class_weights "$VLA_CLASS_WEIGHTS"

if test -s "$VLA_BUNDLE/policy.json" && test -s "$VLA_BUNDLE/policy.pt" \
  && test -s "$VLA_BUNDLE/config_model.yaml" && test -s "$VLA_BUNDLE/config_processor.yaml" \
  && test -s "$VLA_BUNDLE/label_audit.json"; then
  echo "Reusing completed policy bundle: $VLA_BUNDLE"
else
  if test -e "$VLA_BUNDLE"; then
    # Preserve an interrupted export for diagnosis while allowing a clean retry.
    VLA_INCOMPLETE="${VLA_BUNDLE}-incomplete-${SLURM_JOB_ID}"
    test ! -e "$VLA_INCOMPLETE"
    mv "$VLA_BUNDLE" "$VLA_INCOMPLETE"
    echo "Moved incomplete policy bundle to $VLA_INCOMPLETE"
  fi
  "$PWD/.venv/bin/python" examples/minecraft/export_discrete.py \
    --run "$VLA_RUN" --audit "$VLA_AUDIT" \
    --output "$VLA_BUNDLE"
fi
"$PWD/.venv/bin/python" -u examples/minecraft/evaluate_discrete.py \
  --bundle "$VLA_BUNDLE" --manifest "$VLA_DATA/test/manifest.jsonl" \
  --output "$VLA_FINAL"
