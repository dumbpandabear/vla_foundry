#!/bin/bash
set -euo pipefail

VLA_STAGES=${1:-7}
VLA_PREVIOUS=${2:-}
if ! [[ "$VLA_STAGES" =~ ^[1-9][0-9]*$ ]] || { test -n "$VLA_PREVIOUS" && ! [[ "$VLA_PREVIOUS" =~ ^[0-9]+$ ]]; }; then
  echo "usage: $0 [positive-stage-count] [existing-first-job-id]" >&2
  exit 2
fi

for ((VLA_INDEX = 1; VLA_INDEX <= VLA_STAGES; VLA_INDEX++)); do
  VLA_ARGS=(--parsable)
  if test -n "$VLA_PREVIOUS"; then
    VLA_ARGS+=(--dependency="afterany:$VLA_PREVIOUS")
  fi
  VLA_SUBMISSION=$(sbatch "${VLA_ARGS[@]}" examples/minecraft/run_training_dashboard.slurm)
  VLA_JOB_ID=$(printf '%s\n' "$VLA_SUBMISSION" | tail -n 1 | cut -d';' -f1)
  if ! [[ "$VLA_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch response: $VLA_SUBMISSION" >&2
    exit 1
  fi
  printf 'dashboard_stage=%d job=%s dependency=%s\n' "$VLA_INDEX" "$VLA_JOB_ID" "${VLA_PREVIOUS:-none}"
  VLA_PREVIOUS=$VLA_JOB_ID
done
