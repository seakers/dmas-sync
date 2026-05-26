#!/bin/bash
#SBATCH --job-name=cbba_study_exp_2_full_consensus_run
#SBATCH --output=experiments/2_centralized_vs_decentralized/logs/outputs/study_2_full_consensus_run_%A_%a.out
#SBATCH --error=experiments/2_centralized_vs_decentralized/logs/errors/study_2_full_consensus_run_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
# #SBATCH --array=0-107

# #SBATCH --array=0-0
# #SBATCH --array=1-24
# #SBATCH --array=25-49
# #SBATCH --array=50-74
# #SBATCH --array=75-107


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

# --- Gurobi license ---
export GRB_LICENSE_FILE="$SCRATCH/src/dmas-sync/gurobi.lic"

# --- Conda environment activation ---
eval "$(conda shell.bash hook)"
conda activate ./.venv

# --- Set ulimit for maximum number of open files ---
ulimit -n 65535

# --- Trial ID lists per job (fastest -> slowest) ---
i=$SLURM_ARRAY_TASK_ID
start=$((i * 4))
end=$((start + 4))

# --- Run ---
python ./experiments/2_centralized_vs_decentralized/study.py \
      -t full_factorial_trials_2026-05-25 \
      --trial-range "${start}:${end}" \
      --quiet --single-thread \
      --force-postprocess --force-summarize
      # --force-simulate --only-simulate
      # --exceptions --profile-cpu --profile-mem