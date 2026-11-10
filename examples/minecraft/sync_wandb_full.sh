#!/bin/bash
set -euo pipefail

VLA_RUN=${1:-/storage/home/ericli11/minecraft-vla-poc/runs/minecraft-qwen3-d29-full}
VLA_WANDB="$VLA_RUN/wandb"

test -d "$VLA_WANDB"
if test -z "${WANDB_API_KEY:-}" && ! { test -f "$HOME/.netrc" && grep -q 'machine api.wandb.ai' "$HOME/.netrc"; }; then
  echo "Configure WANDB_API_KEY or log in with W&B before syncing $VLA_WANDB" >&2
  exit 2
fi

exec /storage/home/ericli11/Projects/vla_foundry/.venv/bin/python -m wandb sync \
  --include-offline \
  --mark-synced \
  --append \
  --project minecraft-vla-foundry \
  "$VLA_WANDB"
