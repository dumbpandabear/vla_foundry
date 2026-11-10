#!/bin/bash
set -euo pipefail
if test "$#" -lt 3 || test "$#" -gt 5; then
  echo "usage: $0 RUN_DIR OUTPUT_BUNDLE OUTPUT_JSON [MANIFEST [CHECKPOINT_NUMBER]]" >&2
  exit 2
fi
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_MANIFEST=${4:-$VLA_ROOT/dataset-v2/test/manifest.jsonl}
VLA_CHECKPOINT_ARGS=()
if test "$#" -eq 5; then
  VLA_CHECKPOINT_ARGS=(--checkpoint-number "$5")
fi
if test ! -s "$2/policy.pt"; then
  "$PWD/.venv/bin/python" examples/minecraft/export_discrete.py \
    --run "$1" --audit "$VLA_ROOT/d29-label-audit.json" --output "$2" \
    "${VLA_CHECKPOINT_ARGS[@]}"
fi
"$PWD/.venv/bin/python" -u examples/minecraft/evaluate_discrete.py \
  --bundle "$2" --manifest "$VLA_MANIFEST" --output "$3"
