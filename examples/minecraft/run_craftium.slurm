#!/bin/bash
set -euo pipefail
cd /storage/home/ericli11/Projects/vla_foundry
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SDL_VIDEODRIVER=offscreen
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
VLA_MESA_JSON="${SLURM_TMPDIR:-/tmp}/craftium-mesa-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"}}' > "$VLA_MESA_JSON"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_MESA_JSON"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
exec "$PWD/.venv/bin/python" -u \
 /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/run_minecraft_vla.py "$@"
