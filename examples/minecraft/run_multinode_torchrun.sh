#!/bin/bash
set -euo pipefail

: "${VLA_NNODES:?VLA_NNODES is required}"
: "${VLA_NPROC_PER_NODE:?VLA_NPROC_PER_NODE is required}"
: "${VLA_MASTER_ADDR:?VLA_MASTER_ADDR is required}"
: "${VLA_MASTER_PORT:?VLA_MASTER_PORT is required}"
: "${VLA_TRAIN_ARGS:?VLA_TRAIN_ARGS is required}"

exec "$PWD/.venv/bin/python" -m torch.distributed.run \
  --nnodes="$VLA_NNODES" \
  --nproc_per_node="$VLA_NPROC_PER_NODE" \
  --node_rank="${SLURM_NODEID:?SLURM_NODEID is required}" \
  --master_addr="$VLA_MASTER_ADDR" \
  --master_port="$VLA_MASTER_PORT" \
  vla_foundry/main.py $VLA_TRAIN_ARGS
