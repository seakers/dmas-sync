#!/bin/bash
#SBATCH --job-name=cbba_study_exp_1_process
#SBATCH --output=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.out
#SBATCH --error=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-74

# --array=0-35

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

# --- Set ulimit for maximum number of open files ---
ulimit -n 65535

# --- Determine trial range based on SLURM_ARRAY_TASK_ID ---
i=$SLURM_ARRAY_TASK_ID
start=$((i))
end=$((1+i))

# --- Run ---
python ./experiments/1_0_cbba_stress_test/study.py \
  -t full_factorial_trials_2026-02-22 \
  --trial-range "${start}:${end}" \
  --single-thread --quiet --exceptions --force-postprocess --force-summarize \
  # --profile-cpu # --profile-mem