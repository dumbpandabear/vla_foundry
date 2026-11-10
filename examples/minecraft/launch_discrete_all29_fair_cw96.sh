#!/bin/bash
set -euo pipefail

# Fine-tune all 29 native Craftium actions after the full VPT foundation run.
VLA_REPO_ROOT=${VLA_REPO_ROOT:-$PWD}
VLA_ROOT=${VLA_ROOT:-/storage/home/ericli11/minecraft-vla-poc}
VLA_DATA=${VLA_ALL29_DATA:-/checkpoint/data/ericli11/minecraft-vla-poc/dataset-craftium-d29-balanced}
VLA_BASE_MODEL=${VLA_BASE_MODEL:-/checkpoint/data/ericli11/models/Qwen3-VL-30B-A3B-Thinking}
VLA_RUNS=${VLA_RUNS:-$VLA_ROOT/runs}
VLA_FOUNDATION_RUN=${VLA_FOUNDATION_RUN:-$VLA_RUNS/minecraft-qwen3-d29-full-fair-cw96}
VLA_NAME=${VLA_ALL29_NAME:-minecraft-qwen3-d29-all29-fair-cw96}
VLA_GPUS_PER_NODE=${VLA_GPUS_PER_NODE:-4}
VLA_EXPECTED_WORLD_SIZE=${VLA_EXPECTED_WORLD_SIZE:-96}
VLA_PER_GPU_BATCH_SIZE=${VLA_PER_GPU_BATCH_SIZE:-4}
VLA_GLOBAL_BATCH_SIZE=${VLA_GLOBAL_BATCH_SIZE:-384}
VLA_LR=${VLA_ALL29_LR:-0.00001}
VLA_WARMUP=${VLA_ALL29_WARMUP:-200}
VLA_EPOCHS=${VLA_ALL29_EPOCHS:-10}

: "${SLURM_JOB_ID:?Run this launcher inside a Slurm allocation}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
cd "$VLA_REPO_ROOT"

VLA_WORLD_SIZE=$((SLURM_NNODES * VLA_GPUS_PER_NODE))
if test "$VLA_WORLD_SIZE" -ne "$VLA_EXPECTED_WORLD_SIZE"; then
  echo "Expected $VLA_EXPECTED_WORLD_SIZE GPUs, got $VLA_WORLD_SIZE" >&2
  exit 2
fi
if test $((VLA_GLOBAL_BATCH_SIZE % (VLA_WORLD_SIZE * VLA_PER_GPU_BATCH_SIZE))) -ne 0; then
  echo "Global batch must be divisible by world size x per-GPU microbatch" >&2
  exit 2
fi

VLA_PYTHON="$VLA_REPO_ROOT/.venv/bin/python"
VLA_AUDIT="$VLA_DATA/d29_label_audit.json"
test -x "$VLA_PYTHON"
test -s "$VLA_DATA/dataset.json"
test -s "$VLA_DATA/train/manifest.jsonl"
test -s "$VLA_DATA/val/manifest.jsonl"
test -s "$VLA_DATA/test/manifest.jsonl"
test -s "$VLA_BASE_MODEL/config.json"
test -s "$VLA_AUDIT"
test -s "$VLA_FOUNDATION_RUN/out.log"

if test -z "${VLA_RESUME:-}"; then
  VLA_RESUME=$(VLA_FOUNDATION_RUN="$VLA_FOUNDATION_RUN" "$VLA_PYTHON" -c '
import os
import pathlib
import re

run = pathlib.Path(os.environ["VLA_FOUNDATION_RUN"])
rows = re.findall(r"Validation after checkpoint (\d+): avg_loss=([\d.eE+-]+)", (run / "out.log").read_text())
if not rows:
    raise SystemExit("No validated foundation checkpoint")
number, _ = min(rows, key=lambda row: float(row[1]))
print(run / "checkpoints" / f"checkpoint_{number}.pt")
')
fi
test -s "$VLA_RESUME"

VLA_TRAIN_SAMPLES=$(jq -er '.samples.train | select(. > 0)' "$VLA_DATA/dataset.json")
VLA_TOTAL_SAMPLES=$((VLA_TRAIN_SAMPLES * VLA_EPOCHS))
VLA_CLASS_WEIGHTS=$(PYTHONPATH="$VLA_REPO_ROOT" "$VLA_PYTHON" \
  examples/minecraft/compute_class_weights.py --audit "$VLA_AUDIT")

export PYTHONPATH="$VLA_REPO_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export VLA_NNODES="$SLURM_NNODES"
export VLA_NPROC_PER_NODE="$VLA_GPUS_PER_NODE"
export VLA_MASTER_ADDR
VLA_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export VLA_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export VLA_TRAIN_ARGS="--config_path examples/minecraft/qwen3_discrete29_all29.yaml \
  --name $VLA_NAME \
  --save_path $VLA_RUNS \
  --total_train_samples $VLA_TOTAL_SAMPLES \
  --model.vision_language_backbone.hf_pretrained $VLA_BASE_MODEL \
  --model.resume_from_checkpoint $VLA_RESUME \
  --model.resume_weights_only true \
  --data.processor $VLA_BASE_MODEL \
  --data.dataset_manifest [$VLA_DATA/train/manifest.jsonl] \
  --data.dataset_statistics [$VLA_DATA/train/stats.json] \
  --data.val_dataset_manifest [$VLA_DATA/val/manifest.jsonl] \
  --data.val_dataset_statistics [$VLA_DATA/val/stats.json] \
  --hparams.global_batch_size $VLA_GLOBAL_BATCH_SIZE \
  --hparams.per_gpu_batch_size $VLA_PER_GPU_BATCH_SIZE \
  --hparams.lr $VLA_LR \
  --hparams.warmup $VLA_WARMUP \
  --hparams.class_weights $VLA_CLASS_WEIGHTS"

printf 'nodes=%s world_size=%s global_batch=%s microbatch=%s epochs=%s total_samples=%s\n' \
  "$SLURM_NNODES" "$VLA_WORLD_SIZE" "$VLA_GLOBAL_BATCH_SIZE" \
  "$VLA_PER_GPU_BATCH_SIZE" "$VLA_EPOCHS" "$VLA_TOTAL_SAMPLES"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  examples/minecraft/run_multinode_torchrun.sh

VLA_RUN="$VLA_RUNS/$VLA_NAME"
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-all29-fair-cw96"
VLA_EVALUATION="$VLA_ROOT/test-qwen3-d29-all29-fair-cw96.json"
if test -e "$VLA_BUNDLE"; then
  echo "Refusing to overwrite existing bundle: $VLA_BUNDLE" >&2
  exit 2
fi
"$VLA_PYTHON" examples/minecraft/export_discrete.py \
  --run "$VLA_RUN" --audit "$VLA_AUDIT" --output "$VLA_BUNDLE"
"$VLA_PYTHON" -u examples/minecraft/evaluate_discrete.py \
  --bundle "$VLA_BUNDLE" --manifest "$VLA_DATA/test/manifest.jsonl" --output "$VLA_EVALUATION"
"$VLA_PYTHON" -u examples/minecraft/gate_discrete_all29.py \
  --evaluation "$VLA_EVALUATION" \
  --policy "$VLA_BUNDLE/policy.json" \
  --audit "$VLA_AUDIT" \
  --training-log "$VLA_RUN/out.log" \
  --output "$VLA_ROOT/d29-all29-quality-gate-fair-cw96.json"
