#!/bin/bash
#SBATCH --job-name=minecraft-d29-scale16
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=00:35:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/scale16-smoke-%j.log
set -euo pipefail

export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_RUN="$VLA_ROOT/runs/minecraft-qwen3-d29-full"
VLA_SMOKE_RUN="$VLA_ROOT/runs/minecraft-qwen3-d29-scale16-smoke"
VLA_AUDIT="$VLA_ROOT/d29-label-audit-full.json"
VLA_DATA=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-v2-full

test ! -e "$VLA_SMOKE_RUN"
VLA_RESUME=$(find "$VLA_RUN/checkpoints" -maxdepth 1 -name 'checkpoint_*.pt' -printf '%f\n' \
  | sort -V | tail -n 1)
test -n "$VLA_RESUME"
VLA_RESUME="$VLA_RUN/checkpoints/$VLA_RESUME"
VLA_TOTAL_SAMPLES=$(jq -er '.samples.train | select(. > 0)' "$VLA_DATA/dataset.json")
VLA_CLASS_WEIGHTS=$("$PWD/.venv/bin/python" examples/minecraft/compute_class_weights.py --audit "$VLA_AUDIT")

export VLA_NNODES="$SLURM_NNODES"
export VLA_NPROC_PER_NODE=8
export VLA_MASTER_ADDR
VLA_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export VLA_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export VLA_TRAIN_ARGS="--config_path examples/minecraft/qwen3_discrete29_full.yaml \
  --name minecraft-qwen3-d29-scale16-smoke \
  --total_train_samples $VLA_TOTAL_SAMPLES \
  --model.resume_from_checkpoint $VLA_RESUME \
  --model.resume_weights_only false \
  --hparams.per_gpu_batch_size 16 \
  --hparams.class_weights $VLA_CLASS_WEIGHTS \
  --validate_on_resume false \
  --wandb false \
  --log_every_n_steps 1"

set +e
timeout --signal=TERM 30m srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 examples/minecraft/run_multinode_torchrun.sh
VLA_STATUS=$?
set -e
if test "$VLA_STATUS" -ne 0 && test "$VLA_STATUS" -ne 124; then
  exit "$VLA_STATUS"
fi
echo "16-GPU scaling smoke completed with status $VLA_STATUS"
