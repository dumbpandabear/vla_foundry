#!/bin/bash
#SBATCH --job-name=minecraft-vla-live
#SBATCH --partition=h200
#SBATCH --qos=h200_dev
#SBATCH --account=data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/dashboard-%j.log
set -euo pipefail
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SDL_VIDEODRIVER=offscreen EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
VLA_MESA_JSON="${SLURM_TMPDIR:-/tmp}/craftium-mesa-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"}}' > "$VLA_MESA_JSON"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VLA_MESA_JSON"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
exec "$PWD/.venv/bin/python" -u \
 /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/serve_minecraft_vla.py \
 --bundle /storage/home/ericli11/minecraft-vla-poc/policy-qwen3-d29-balanced-poc \
 --replan-ticks 4 \
 --output "/storage/home/ericli11/minecraft-vla-poc/dashboard/$SLURM_JOB_ID" "$@"
