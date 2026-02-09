#!/bin/bash
#SBATCH --job-name=cbba_stress
#SBATCH --output=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.out
#SBATCH --error=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-23

set -euo pipefail

# --- Go to your project in scratch ---
cd "$SCRATCH/src/dmas-sync"
mkdir -p experiments/1_0_cbba_stress_test/logs

# --- Load Modules ---
module purge
module load GCCcore/11.3.0 Python/3.8.6
module load Anaconda3

# --- Conda environment activation ---
eval "$(conda shell.bash hook)"
conda activate ./.venv

# --- Determine trial range based on SLURM_ARRAY_TASK_ID ---
i=$SLURM_ARRAY_TASK_ID
start=$i
end=$((i+1))

# --- Run ---
python ./experiments/1_0_cbba_stress_test/study.py \
  -t lhs_trials-2_samples-1000_seed \
  --trial-range "${start}:${end}" \
  --single-thread --quiet --exceptions #--force-simulate