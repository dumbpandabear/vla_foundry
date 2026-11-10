#!/bin/bash
set -euo pipefail

VLA_REPO_ROOT=${VLA_REPO_ROOT:-$PWD}
VLA_ROOT=${VLA_ROOT:-/storage/home/ericli11/minecraft-vla-poc}
VLA_SOURCE=${VLA_SOURCE:-/s3/profiles/data-foundations-use2-1/minecraft_vpt}
VLA_DATA=${VLA_DATA:-/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index}
VLA_BASE_MODEL=${VLA_BASE_MODEL:-/checkpoint/data/ericli11/models/Qwen3-VL-30B-A3B-Thinking}
VLA_RESUME=${VLA_RESUME:-$VLA_ROOT/transfer/checkpoint_8.pt}
VLA_GPUS_PER_NODE=${VLA_GPUS_PER_NODE:-4}
VLA_EXPECTED_WORLD_SIZE=${VLA_EXPECTED_WORLD_SIZE:-96}

: "${SLURM_JOB_ID:?Run this preflight inside the intended Slurm allocation}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
VLA_WORLD_SIZE=$((SLURM_NNODES * VLA_GPUS_PER_NODE))
test "$VLA_WORLD_SIZE" -eq "$VLA_EXPECTED_WORLD_SIZE"

export VLA_REPO_ROOT VLA_ROOT VLA_SOURCE VLA_DATA VLA_BASE_MODEL VLA_RESUME VLA_GPUS_PER_NODE
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
set -euo pipefail
test -x "$VLA_REPO_ROOT/.venv/bin/python"
test -s "$VLA_DATA/dataset.json"
test -s "$VLA_DATA/train/manifest.jsonl"
test -s "$VLA_DATA/val/manifest.jsonl"
test -s "$VLA_DATA/test/manifest.jsonl"
test -d "$VLA_SOURCE/data"
test -d "$VLA_SOURCE/splits"
test -s "$VLA_BASE_MODEL/config.json"
test -s "$VLA_RESUME"
PYTHONPATH="$VLA_REPO_ROOT" "$VLA_REPO_ROOT/.venv/bin/python" -c "
import platform
import socket
import torch

expected = int(${VLA_GPUS_PER_NODE})
assert torch.cuda.is_available(), \"CUDA is unavailable\"
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
print(socket.gethostname(), platform.machine(), torch.__version__, torch.version.cuda, torch.cuda.device_count())
"
'

echo "FAIR-CW preflight passed on $SLURM_NNODES nodes / $VLA_WORLD_SIZE GPUs"
