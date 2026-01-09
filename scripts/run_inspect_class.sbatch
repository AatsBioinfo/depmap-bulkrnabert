#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=bulkrna_class_inspect
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail
PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
IMG=$PROJ/container/python311.sif
VENV=$PROJ/env/.venv
APPTAINER=/usr/bin/apptainer

$APPTAINER exec --bind /fast:/fast "$IMG" bash -lc "
  source $VENV/bin/activate
  python $PROJ/scripts/inspect_model_class.py
"
