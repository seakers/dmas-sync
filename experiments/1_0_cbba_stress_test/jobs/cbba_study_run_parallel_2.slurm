#!/bin/bash
#SBATCH --job-name=cbba_stress_parallel
#SBATCH --output=experiments/1_0_cbba_stress_test/logs/cbba_stress_parallel_%j.out
#SBATCH --error=experiments/1_0_cbba_stress_test/logs/cbba_stress_parallel_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# If you know you need a specific partition, uncomment one:
##SBATCH --partition=short
##SBATCH --partition=medium
##SBATCH --partition=long

set -euo pipefail

# --- Go to your project in scratch ---
cd "$SCRATCH/src/dmas-sync"

# Make sure log dir exists
mkdir -p logs

# --- Modules (use ml if that’s what your cluster prefers) ---
module purge
module load GCCcore/11.3.0 Python/3.8.6
module load Anaconda3

# --- Conda activation in non-interactive shells ---
# (This makes `conda activate` work reliably inside sbatch jobs)
eval "$(conda shell.bash hook)"
conda activate ./.venv

# --- Run ---
python ./experiments/1_0_cbba_stress_test/study.py -t lhs_trials-2_samples-1000_seed --trial-range 12:24 -p -m --force-simulate --quiet
