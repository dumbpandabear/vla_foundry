#!/bin/bash
set -euo pipefail

VLA_SMOKE_JOB=${1:?usage: $0 SMOKE_JOB_ID [STAGE_COUNT]}
VLA_STAGES=${2:-6}
if ! [[ "$VLA_SMOKE_JOB" =~ ^[0-9]+$ && "$VLA_STAGES" =~ ^[1-9][0-9]*$ ]]; then
  echo "smoke job and positive stage count must be integers" >&2
  exit 2
fi

VLA_PREVIOUS="$VLA_SMOKE_JOB"
for ((VLA_INDEX = 1; VLA_INDEX <= VLA_STAGES; VLA_INDEX++)); do
  if test "$VLA_INDEX" -eq 1; then
    VLA_DEPENDENCY="afterok:$VLA_PREVIOUS"
  else
    VLA_DEPENDENCY="afterany:$VLA_PREVIOUS"
  fi
  VLA_SUBMISSION=$(sbatch --parsable --dependency="$VLA_DEPENDENCY" \
    examples/minecraft/train_discrete_full_scaled_stage.slurm)
  VLA_JOB_ID=$(printf '%s\n' "$VLA_SUBMISSION" | tail -n 1 | cut -d';' -f1)
  [[ "$VLA_JOB_ID" =~ ^[0-9]+$ ]]
  printf 'scaled_stage=%d job=%s dependency=%s\n' "$VLA_INDEX" "$VLA_JOB_ID" "$VLA_DEPENDENCY"
  VLA_PREVIOUS="$VLA_JOB_ID"
done

VLA_VERIFY=$(sbatch --parsable --dependency="afterok:$VLA_PREVIOUS" \
  examples/minecraft/verify_discrete_full_craftium.slurm)
VLA_VERIFY_ID=$(printf '%s\n' "$VLA_VERIFY" | tail -n 1 | cut -d';' -f1)
[[ "$VLA_VERIFY_ID" =~ ^[0-9]+$ ]]
printf 'craftium_verify_job=%s dependency=afterok:%s\n' "$VLA_VERIFY_ID" "$VLA_PREVIOUS"
