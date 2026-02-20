#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=depmap_downstream_min
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail

PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
IMG=$PROJ/container/python311.sif
VENV=$PROJ/env/.venv
SCRIPT=$PROJ/scripts/step3_4_downstream_depmap.py

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# find container runner
if command -v apptainer >/dev/null 2>&1; then
  RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNNER=singularity
else
  echo "ERROR: Neither apptainer nor singularity found on this node."
  exit 1
fi

test -f "$IMG"
test -f "$VENV/bin/activate"
test -f "$SCRIPT"

$RUNNER exec --bind /fast:/fast "$IMG" bash -lc "
  source $VENV/bin/activate
  python -c 'import numpy, pandas, sklearn; import umap; import matplotlib; print(\"OK: deps loaded\")'
  python $SCRIPT
"
