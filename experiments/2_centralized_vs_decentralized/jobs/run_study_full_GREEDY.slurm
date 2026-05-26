#!/bin/bash
#SBATCH --job-name=cbba_study_exp_2_full_greedy_run
#SBATCH --output=experiments/2_centralized_vs_decentralized/logs/outputs/study_2_full_greedy_run_%A_%a.out
#SBATCH --error=experiments/2_centralized_vs_decentralized/logs/errors/study_2_full_greedy_run_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
# #SBATCH --array=432-863

# #SBATCH --array=432-499
# #SBATCH --array=500-599
# #SBATCH --array=600-699
# #SBATCH --array=700-799
# #SBATCH --array=800-863

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

# --- Run ---
python ./experiments/2_centralized_vs_decentralized/study.py \
      -t full_factorial_trials_2026-05-25 \
      --single-trial "${i}" \
      --quiet --single-thread \
      --force-postprocess --force-summarize
      # --force-simulate --only-simulate
      # --exceptions --profile-cpu --profile-mem