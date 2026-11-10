#!/bin/bash
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u examples/minecraft/check_qwen_eval.py "$@"
