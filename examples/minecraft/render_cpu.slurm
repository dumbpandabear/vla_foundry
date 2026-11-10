#!/bin/bash
set -euo pipefail
VLA_ENV=/storage/home/ericli11/Projects/vla_foundry/.venv
export SDL_VIDEODRIVER=offscreen
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
# The wheel's Mesa depends on a missing libgallium; force the system library.
VLA_MESA_JSON="${SLURM_TMPDIR:-/tmp}/craftium-mesa-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"}}' > "$VLA_MESA_JSON"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_MESA_JSON"
export LP_NUM_THREADS=4
export LD_LIBRARY_PATH="$VLA_ENV/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
exec "$VLA_ENV/bin/python" -u \
  /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/test_cpu_render.py \
  --output "/storage/home/ericli11/minecraft-vla-poc/cpu-render-${SLURM_JOB_ID}" \
  --world-prefix "${SLURM_TMPDIR:-/tmp}/craftium-vla-render-${SLURM_JOB_ID}"
