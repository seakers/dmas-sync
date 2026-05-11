#!/bin/bash
#SBATCH --job-name=cbba_study_exp_2_run
#SBATCH --output=experiments/2_centralized_vs_decentralized/logs/outputs/study_2_run_%A_%a.out
#SBATCH --error=experiments/2_centralized_vs_decentralized/logs/errors/study_2_run_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
# #SBATCH --array=0-5

# Overall scenario: 36 total trials
# 6 jobs x 6 trials each (sbatch --array=0-5)
# Each job covers one (Connectivity, Data Processing) group.
# Within each job, trials run fastest-first to spread wall-clock load:
#   None+Greedy -> DP+None -> DP+Greedy -> Centralized-MILP_priority+None -> None+CBBA -> DP+CBBA
#
#  Job  Connectivity         Data Processing   Trial IDs (in run order)
#  ---  -------------------  ---------------   -------------------------
#   0   GS                   Onboard           30 18 12  0 24  6
#   1   GS                   Oracle            31 19 13  1 25  7
#   2   Interconstellation   Onboard           32 20 14  2 26  8
#   3   Interconstellation   Oracle            33 21 15  3 27  9
#   4   Intraconstellation   Onboard           34 22 16  4 28 10
#   5   Intraconstellation   Oracle            35 23 17  5 29 11

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
TRIAL_IDS=(
    "30 18 12  24  6 0"   # Job 0: GS,                 Onboard
    "31 19 13  25  7 1"   # Job 1: GS,                 Oracle
    "32 20 14  26  8 2"   # Job 2: Interconstellation,  Onboard
    "33 21 15  27  9 3"   # Job 3: Interconstellation,  Oracle
    "34 22 16  28 10 4"   # Job 4: Intraconstellation,  Onboard
    "35 23 17  29 11 5"   # Job 5: Intraconstellation,  Oracle
)

# --- Run ---
for trial_id in ${TRIAL_IDS[$SLURM_ARRAY_TASK_ID]}; do
    python ./experiments/2_centralized_vs_decentralized/study.py \
      -t full_factorial_trials_2026-05-10 \
      --single-trial "$trial_id" \
      --single-thread --quiet \
      --force-simulate --force-postprocess --force-summarize
      # --exceptions --profile-cpu --profile-mem
done