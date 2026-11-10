#!/bin/bash
#SBATCH --job-name=minecraft-d29-ddp8-smoke
#SBATCH --account=data
#SBATCH --qos=h200_dev
#SBATCH --partition=h200
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=00:45:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/ddp8-smoke-%j.log
set -euo pipefail
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PWD/.venv/bin/python" -m torch.distributed.run --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
  --config_path examples/minecraft/qwen3_discrete29_balanced.yaml \
  --name minecraft-qwen3-d29-ddp8-smoke-v3 \
  --total_train_samples 4096 \
  --num_checkpoints 1 \
  --total_val_samples 4096 \
  --log_every_n_steps 1 \
  --model.resume_from_checkpoint \
    /storage/home/ericli11/minecraft-vla-poc/runs/minecraft-qwen3-d29-balanced-poc/checkpoints/checkpoint_4.pt \
  --model.resume_weights_only true \
  --hparams.global_batch_size 256 \
  --hparams.warmup 2
