#!/bin/bash
#SBATCH -A izsvak_piggy
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=build_bulkrna_gpu_sif
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/build/%x_%j.out
#SBATCH --error=/fast/home/a/abejoy/projects/depmap_bulkrnabert/logs/build/%x_%j.err

set -euo pipefail

PROJ=/fast/home/a/abejoy/projects/depmap_bulkrnabert
DEF=$PROJ/container/gpu/bulkrnabert_gpu.def
OUT=$PROJ/container/gpu/bulkrnabert_gpu.sif

cd $PROJ

echo "Node: $(hostname)"
which apptainer
apptainer --version

echo "Building container..."
apptainer build --fakeroot "$OUT" "$DEF"

echo "DONE: $OUT"
ls -lh "$OUT"

