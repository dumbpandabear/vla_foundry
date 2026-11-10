#!/bin/bash
set -euo pipefail

# Run inside a Richmond CPU-capable Slurm allocation. The source profile is
# already resident in Richmond; write generated shards to project checkpoint
# storage, not /home and not the raw-data tree.
VLA_REPO_ROOT=${VLA_REPO_ROOT:-$PWD}
VLA_SOURCE=${VLA_SOURCE:-/s3/profiles/data-foundations-use2-1/minecraft_vpt}
: "${VLA_OUTPUT:?Set VLA_OUTPUT to Richmond project checkpoint storage}"
VLA_AUDIT=${VLA_AUDIT:-$(dirname "$VLA_OUTPUT")/d29-label-audit-full.json}
VLA_WORKERS=${VLA_PREP_WORKERS:-32}
VLA_TASKS_PER_CHILD=${VLA_PREP_TASKS_PER_CHILD:-25}

cd "$VLA_REPO_ROOT"
VLA_PYTHON="$VLA_REPO_ROOT/.venv/bin/python"
test -x "$VLA_PYTHON"
test -d "$VLA_SOURCE/data"
test -d "$VLA_SOURCE/splits"
test -s "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl"
test -s "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl"

export PYTHONPATH="$VLA_REPO_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if test -s "$VLA_OUTPUT/dataset.json"; then
  echo "Reusing completed dataset: $VLA_OUTPUT"
else
  VLA_RESUME_ARGS=()
  if test -e "$VLA_OUTPUT"; then
    VLA_RESUME_ARGS=(--resume)
  fi
  "$VLA_PYTHON" -u -m vla_foundry.data.minecraft.prepare_vla \
    --data-root "$VLA_SOURCE" \
    --sources \
      "$VLA_SOURCE/json_data/train_qwen35_9b_04_29_26.jsonl" \
      "$VLA_SOURCE/json_data/val_qwen35_9b_04_29_26.jsonl" \
    --output "$VLA_OUTPUT" \
    --all-clips \
    "${VLA_RESUME_ARGS[@]}" \
    --workers "$VLA_WORKERS" \
    --max-tasks-per-child "$VLA_TASKS_PER_CHILD"
fi

"$VLA_PYTHON" -u examples/minecraft/audit_discrete.py \
  --dataset "$VLA_OUTPUT" \
  --output "$VLA_AUDIT" \
  --workers "$VLA_WORKERS"

"$VLA_PYTHON" - "$VLA_OUTPUT/dataset.json" "$VLA_AUDIT" <<'PY'
import json
import sys

dataset = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
assert dataset["schema"] == "vpt_buttons_float_camera_v2"
assert dataset["samples"]["train"] > 0
assert dataset["samples"]["val"] > 0
assert dataset["samples"]["test"] > 0
counts = audit["splits"]["train"]["class_counts"]
classes_with_targets = [name for name, count in counts.items() if count > 0]
assert len(counts) == 29
assert len(classes_with_targets) == 21
print(json.dumps({"samples": dataset["samples"], "classes_with_targets": classes_with_targets}, indent=2))
PY
