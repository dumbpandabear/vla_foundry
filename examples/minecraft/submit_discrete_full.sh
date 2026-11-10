#!/bin/bash
set -euo pipefail

VLA_ROOT=/storage/home/ericli11/minecraft-vla-poc
VLA_DATA=/checkpoint/data/ericli11/minecraft-vla-poc/dataset-v2-full/dataset.json
VLA_BALANCED="$VLA_ROOT/policy-qwen3-d29-balanced-poc/policy.json"
VLA_STAGES=${1:-10}
VLA_PREP_JOB=${2:-}

if ! [[ "$VLA_STAGES" =~ ^[1-9][0-9]*$ ]] || { test -n "$VLA_PREP_JOB" && ! [[ "$VLA_PREP_JOB" =~ ^[0-9]+$ ]]; }; then
  echo "usage: $0 [positive-stage-count] [preprocessing-job-id]" >&2
  exit 2
fi
if test -z "$VLA_PREP_JOB"; then
  test -s "$VLA_DATA"
fi
test -s "$VLA_BALANCED"

VLA_PREVIOUS=
for ((VLA_INDEX = 1; VLA_INDEX <= VLA_STAGES; VLA_INDEX++)); do
  VLA_ARGS=(--parsable)
  if test -n "$VLA_PREVIOUS"; then
    VLA_ARGS+=(--dependency="afterany:$VLA_PREVIOUS")
  elif test -n "$VLA_PREP_JOB"; then
    # Never train from an incomplete corpus.  Later stages use afterany so
    # they can resume checkpoints after the 24-hour stage wall-time.
    VLA_ARGS+=(--dependency="afterok:$VLA_PREP_JOB")
  fi
  VLA_SUBMISSION=$(sbatch "${VLA_ARGS[@]}" examples/minecraft/train_discrete_full_stage.slurm)
  VLA_JOB_ID=$(printf '%s\n' "$VLA_SUBMISSION" | tail -n 1 | cut -d';' -f1)
  if ! [[ "$VLA_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch response: $VLA_SUBMISSION" >&2
    exit 1
  fi
  printf 'stage=%d job=%s dependency=%s\n' \
    "$VLA_INDEX" "$VLA_JOB_ID" "${VLA_PREVIOUS:-${VLA_PREP_JOB:+afterok:}${VLA_PREP_JOB:-none}}"
  VLA_PREVIOUS=$VLA_JOB_ID
done
