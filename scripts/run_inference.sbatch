#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=depmap_bulkrnabert_inf
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/%x_%j.err

set -euo pipefail

PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
IMG=$PROJ/container/python311.sif
VENV=$PROJ/env/.venv
APPTAINER=/usr/bin/apptainer

# keep threads aligned with allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# sanity checks (fail early with clear error)
test -f "$IMG"
test -f "$VENV/bin/activate"
test -f "$PROJ/scripts/extract_embeddings_depmap.py"

$APPTAINER exec --bind /fast:/fast "$IMG" bash -lc "
  source $VENV/bin/activate
  python -c 'import jax, haiku as hk; print(\"JAX devices:\", jax.devices()); print(\"HK:\", hk.__version__)'
  python $PROJ/scripts/extract_embeddings_depmap.py
"
