#!/bin/bash
#SBATCH --job-name=cbba_study_exp_2_run
#SBATCH --output=experiments/2_centralized_vs_decentralized/logs/outputs/cbba_stress_%A_%a.out
#SBATCH --error=experiments/2_centralized_vs_decentralized/logs/errors/cbba_stress_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# #SBATCH --array=0-35
# Overal scenario : 288 total trials
# Option 1: 4 trials per job -> 72 jobs  (sbatch --array=0-71)
# Option 2: 8 trials per job -> 36 jobs  (sbatch --array=0-35)  <-- current
# Option 3: 16 trials per job -> 18 jobs (sbatch --array=0-17)

set -euo pipefail

# --- Go to your project in scratch ---
cd "$SCRATCH/src/dmas-sync" || { echo "ERROR: could not cd to $SCRATCH/src/dmas-sync"; exit 1; }
mkdir -p experiments/2_centralized_vs_decentralized/logs
mkdir -p experiments/2_centralized_vs_decentralized/logs/outputs
mkdir -p experiments/2_centralized_vs_decentralized/logs/errors

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
start=$((i * 8))
end=$((start + 8))

# --- Run ---
python ./experiments/2_centralized_vs_decentralized/study.py \
  -t full_factorial_trials_2026-05-04 \
  --trial-range "${start}:${end}" \
  --single-thread --quiet --exceptions \
  # --force-simulate --force-postprocess --force-summarize # \
  # --profile-cpu --profile-mem