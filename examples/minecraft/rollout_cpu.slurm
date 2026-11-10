#!/bin/bash
set -euo pipefail
VLA_REPO=/storage/home/ericli11/Projects/vla_foundry
VLA_ENV="$VLA_REPO/.venv"
export PYTHONPATH="$VLA_REPO"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LP_NUM_THREADS=4
export SDL_VIDEODRIVER=offscreen
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
VLA_MESA_JSON="${SLURM_TMPDIR:-/tmp}/craftium-mesa-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"}}' > "$VLA_MESA_JSON"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_MESA_JSON"
export LD_LIBRARY_PATH="$VLA_ENV/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
exec "$VLA_ENV/bin/python" -u \
  /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/run_minecraft_vla.py \
  --bundle "${VLA_BUNDLE:-/storage/home/ericli11/minecraft-vla-poc/policy-v1}" \
  --device cpu --steps "${VLA_STEPS:-64}" \
  --output "/storage/home/ericli11/minecraft-vla-poc/rollout-cpu-${SLURM_JOB_ID}"
