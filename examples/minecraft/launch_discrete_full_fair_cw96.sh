#!/bin/bash
set -euo pipefail

# Run this inside a 24-node Richmond GB300 allocation with 4 GPUs per node. Paths may
# be overridden for clusters that do not share /storage/home and /checkpoint.
VLA_REPO_ROOT=${VLA_REPO_ROOT:-$PWD}
VLA_ROOT=${VLA_ROOT:-/storage/home/ericli11/minecraft-vla-poc}
VLA_SOURCE=${VLA_SOURCE:-/s3/profiles/data-foundations-use2-1/minecraft_vpt}
VLA_DATA=${VLA_DATA:-/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index}
VLA_BASE_MODEL=${VLA_BASE_MODEL:-/checkpoint/data/ericli11/models/Qwen3-VL-30B-A3B-Thinking}
VLA_AUDIT=${VLA_AUDIT:-$VLA_DATA/d29_label_audit.json}
VLA_RESUME=${VLA_RESUME:-$VLA_ROOT/transfer/checkpoint_8.pt}
VLA_RUNS=${VLA_RUNS:-$VLA_ROOT/runs}
VLA_NAME=${VLA_NAME:-minecraft-qwen3-d29-full-fair-cw96}
VLA_GPUS_PER_NODE=${VLA_GPUS_PER_NODE:-4}
VLA_EXPECTED_WORLD_SIZE=${VLA_EXPECTED_WORLD_SIZE:-96}
VLA_PER_GPU_BATCH_SIZE=${VLA_PER_GPU_BATCH_SIZE:-4}
VLA_GLOBAL_BATCH_SIZE=${VLA_GLOBAL_BATCH_SIZE:-384}
VLA_LR=${VLA_LR:-0.00005}
VLA_WARMUP=${VLA_WARMUP:-200}
VLA_NUM_CHECKPOINTS=${VLA_NUM_CHECKPOINTS:-80}
VLA_TOTAL_VAL_SAMPLES=${VLA_TOTAL_VAL_SAMPLES:-32768}

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
test -x "$VLA_PYTHON"
test -s "$VLA_DATA/dataset.json"
test -s "$VLA_DATA/train/manifest.jsonl"
test -s "$VLA_DATA/val/manifest.jsonl"
test -s "$VLA_DATA/test/manifest.jsonl"
test -s "$VLA_BASE_MODEL/config.json"
test -s "$VLA_AUDIT"
test -s "$VLA_RESUME"

VLA_TOTAL_SAMPLES=${VLA_TOTAL_SAMPLES:-$(jq -er '.samples.train | select(. > 0)' "$VLA_DATA/dataset.json")}
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
export VLA_TRAIN_ARGS="--config_path examples/minecraft/qwen3_discrete29_full.yaml \
  --name $VLA_NAME \
  --save_path $VLA_RUNS \
  --total_train_samples $VLA_TOTAL_SAMPLES \
  --num_checkpoints $VLA_NUM_CHECKPOINTS \
  --total_val_samples $VLA_TOTAL_VAL_SAMPLES \
  --model.vision_language_backbone.hf_pretrained $VLA_BASE_MODEL \
  --model.resume_from_checkpoint $VLA_RESUME \
  --model.resume_weights_only true \
  --data.type minecraft_vpt_stream \
  --data.processor $VLA_BASE_MODEL \
  --data.vpt_data_root $VLA_SOURCE \
  --data.dataset_modality [minecraft_vpt_stream] \
  --data.dataset_manifest [$VLA_DATA/train/manifest.jsonl] \
  --data.dataset_statistics [$VLA_DATA/train/stats.json] \
  --data.val_dataset_manifest [$VLA_DATA/val/manifest.jsonl] \
  --data.val_dataset_statistics [$VLA_DATA/val/stats.json] \
  --hparams.global_batch_size $VLA_GLOBAL_BATCH_SIZE \
  --hparams.per_gpu_batch_size $VLA_PER_GPU_BATCH_SIZE \
  --hparams.lr $VLA_LR \
  --hparams.warmup $VLA_WARMUP \
  --hparams.class_weights $VLA_CLASS_WEIGHTS"

printf 'nodes=%s world_size=%s global_batch=%s microbatch=%s total_samples=%s\n' \
  "$SLURM_NNODES" "$VLA_WORLD_SIZE" "$VLA_GLOBAL_BATCH_SIZE" \
  "$VLA_PER_GPU_BATCH_SIZE" "$VLA_TOTAL_SAMPLES"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  examples/minecraft/run_multinode_torchrun.sh

VLA_RUN="$VLA_RUNS/$VLA_NAME"
VLA_BUNDLE="$VLA_ROOT/policy-qwen3-d29-full-fair-cw96"
VLA_EVALUATION="$VLA_ROOT/test-qwen3-d29-full-fair-cw96.json"
if test -e "$VLA_BUNDLE"; then
  echo "Refusing to overwrite existing bundle: $VLA_BUNDLE" >&2
  exit 2
fi
"$VLA_PYTHON" examples/minecraft/export_discrete.py \
  --run "$VLA_RUN" --audit "$VLA_AUDIT" --output "$VLA_BUNDLE"
"$VLA_PYTHON" -u examples/minecraft/evaluate_discrete.py \
  --bundle "$VLA_BUNDLE" \
  --manifest "$VLA_DATA/test/manifest.jsonl" \
  --vpt-data-root "$VLA_SOURCE" \
  --output "$VLA_EVALUATION"
