#!/bin/bash
#SBATCH --job-name=minecraft-d29-dashboard
#SBATCH --account=data
#SBATCH --qos=cpu_lowest
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/storage/home/ericli11/Projects/vla_foundry
#SBATCH --output=/storage/home/ericli11/minecraft-vla-poc/metamate-dashboard-%j.log
set -euo pipefail

exec "$PWD/.venv/bin/python" -u examples/minecraft/build_training_dashboard.py --watch
