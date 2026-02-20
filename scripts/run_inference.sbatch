#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=depmap_bulkrnabert_inf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail

PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
IMG=$PROJ/container/python311.sif
VENV=$PROJ/env/.venv
APPTAINER=/usr/bin/apptainer

# align threads with allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# JAX memory safety
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# start conservative; increase only after a successful run
export BULKRNABERT_BATCH_SIZE=2
export BULKRNABERT_USE_JIT=1
export BULKRNABERT_RESUME=1

test -f "$IMG"
test -f "$VENV/bin/activate"
test -f "$PROJ/scripts/extract_embeddings_depmap.py"

$APPTAINER exec --bind /fast:/fast "$IMG" bash -lc "
  source $VENV/bin/activate
  python -c 'import jax; print(\"JAX devices:\", jax.devices())'
  python $PROJ/scripts/extract_embeddings_depmap.py
"
