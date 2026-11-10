#!/bin/bash
#SBATCH --job-name=minecraft-qwen3-d29-all29
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/train-d29-all29-%j.log
set -euo pipefail

export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_FOUNDATION="$VLA_ROOT/runs/minecraft-qwen3-d29-full"
VLA_RUN="$VLA_ROOT/runs/minecraft-qwen3-d29-all29"
VLA_DATA=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-craftium-d29-balanced
VLA_AUDIT="$VLA_DATA/d29_label_audit.json"
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-all29"
VLA_EVALUATION="$VLA_ROOT/test-qwen3-d29-all29.json"

test -s "$VLA_DATA/dataset.json"
test -s "$VLA_AUDIT"
VLA_TRAIN_SAMPLES=$(jq -er '.samples.train | select(. > 0)' "$VLA_DATA/dataset.json")
VLA_TOTAL_SAMPLES=$((VLA_TRAIN_SAMPLES * 10))
VLA_CLASS_WEIGHTS=$("$PWD/.venv/bin/python" examples/minecraft/compute_class_weights.py --audit "$VLA_AUDIT")
VLA_RESUME=$(find "$VLA_RUN/checkpoints" -maxdepth 1 -name 'checkpoint_*.pt' -printf '%f\n' 2>/dev/null \
  | sort -V | tail -n 1 || true)
if test -n "$VLA_RESUME"; then
  VLA_RESUME="$VLA_RUN/checkpoints/$VLA_RESUME"
  VLA_WEIGHTS_ONLY=false
else
  VLA_RESUME=$("$PWD/.venv/bin/python" -c '
import re, pathlib
run = pathlib.Path("/storage/home/ericli11/minecraft-vla-poc/runs/minecraft-qwen3-d29-full")
rows = re.findall(r"Validation after checkpoint (\d+): avg_loss=([\d.eE+-]+)", (run / "out.log").read_text())
if not rows: raise SystemExit("No validated foundation checkpoint")
number, _ = min(rows, key=lambda row: float(row[1]))
print(run / "checkpoints" / f"checkpoint_{number}.pt")
')
  VLA_WEIGHTS_ONLY=true
fi
test -s "$VLA_RESUME"

"$PWD/.venv/bin/python" -m torch.distributed.run --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
  --config_path examples/minecraft/qwen3_discrete29_all29.yaml \
  --total_train_samples "$VLA_TOTAL_SAMPLES" \
  --model.resume_from_checkpoint "$VLA_RESUME" \
  --model.resume_weights_only "$VLA_WEIGHTS_ONLY" \
  --hparams.class_weights "$VLA_CLASS_WEIGHTS"

if ! test -s "$VLA_BUNDLE/policy.json"; then
  "$PWD/.venv/bin/python" examples/minecraft/export_discrete.py \
    --run "$VLA_RUN" --audit "$VLA_AUDIT" --output "$VLA_BUNDLE"
fi
"$PWD/.venv/bin/python" -u examples/minecraft/evaluate_discrete.py \
  --bundle "$VLA_BUNDLE" --manifest "$VLA_DATA/test/manifest.jsonl" --output "$VLA_EVALUATION"
