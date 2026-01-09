#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=install_umap
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail
IMG=/fast/home/a/abejoy/projects/depmap_bulkrnabert/container/python311.sif
VENV=/fast/home/a/abejoy/projects/depmap_bulkrnabert/env/.venv

/usr/bin/apptainer exec --bind /fast:/fast "$IMG" bash -lc "
  source $VENV/bin/activate
  pip install -U umap-learn
"
