#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH --job-name=depmap_figures
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit from the repository folder:  mkdir -p logs && sbatch run_figures.sbatch
set -euo pipefail

PROJ="${SLURM_SUBMIT_DIR:-$PWD}"
BASE="$(dirname "$PROJ")"            # folder searched for embeddings, expression and metadata
IMG="$PROJ/container/python311.sif"
PY="$PROJ/env/.venv/bin/python"
SCRIPT="$PROJ/scripts/make_figures.py"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_CACHE_DIR="$PROJ/logs/numba_cache"

for f in "$IMG" "$PY" "$SCRIPT"; do
  [ -e "$f" ] || [ -L "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done

if command -v apptainer >/dev/null 2>&1; then RUNNER=apptainer; else RUNNER=singularity; fi

BINDS="$(readlink -f "$BASE")"
[ -d /fast ] && BINDS="$BINDS,/fast"

$RUNNER exec --bind "$BINDS" "$IMG" "$PY" "$SCRIPT" --base "$BASE" --outdir "$PROJ"
