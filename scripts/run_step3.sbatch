#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=depmap_step3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail

PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
IMG=$PROJ/container/python311.sif
VENV=$PROJ/env/.venv

/usr/bin/apptainer exec --bind /fast:/fast "$IMG" bash -lc "
source $VENV/bin/activate
python $PROJ/scripts/step3_pca_umap.py
"
