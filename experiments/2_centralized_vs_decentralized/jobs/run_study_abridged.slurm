#!/bin/bash
#SBATCH --job-name=cbba_study_exp_2_run
#SBATCH --output=experiments/2_centralized_vs_decentralized/logs/outputs/study_2_run_%A_%a.out
#SBATCH --error=experiments/2_centralized_vs_decentralized/logs/errors/study_2_run_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
# #SBATCH --array=0-8

# Overall scenario: 36 total trials
# 9 jobs x 4 trials each (sbatch --array=0-8)
# Each job covers one Connectivity x Planner-pair group.
# Planner pairs balance easy+hard within each job:
#   Pair A (jobs 0-2): None+Greedy (easy)  + Centralized-MILP_priority (hardest)
#   Pair B (jobs 3-5): DP+None    (easy)   + DP+CBBA                   (hard)
#   Pair C (jobs 6-8): DP+Greedy  (medium) + None+CBBA                 (medium)
# Within each job, easier planner runs first (both DataProcessing options), then harder.
#
#  Job  Connectivity         Planners                           Trial IDs (run order)
#  ---  -------------------  ---------------------------------  ---------------------
#   0   GS                   None+Greedy, Centralized-MILP      30 31  0  1
#   1   Interconstellation   None+Greedy, Centralized-MILP      32 33  2  3
#   2   Intraconstellation   None+Greedy, Centralized-MILP      34 35  4  5
#   3   GS                   DP+None, DP+CBBA                   18 19  6  7
#   4   Interconstellation   DP+None, DP+CBBA                   20 21  8  9
#   5   Intraconstellation   DP+None, DP+CBBA                   22 23 10 11
#   6   GS                   DP+Greedy, None+CBBA               12 13 24 25
#   7   Interconstellation   DP+Greedy, None+CBBA               14 15 26 27
#   8   Intraconstellation   DP+Greedy, None+CBBA               16 17 28 29

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
    "30 31  0  1"   # Job 0: GS,                None+Greedy + Centralized-MILP
    "32 33  2  3"   # Job 1: Interconstellation, None+Greedy + Centralized-MILP
    "34 35  4  5"   # Job 2: Intraconstellation, None+Greedy + Centralized-MILP
    "18 19  6  7"   # Job 3: GS,                DP+None + DP+CBBA
    "20 21  8  9"   # Job 4: Interconstellation, DP+None + DP+CBBA
    "22 23 10 11"   # Job 5: Intraconstellation, DP+None + DP+CBBA
    "12 13 24 25"   # Job 6: GS,                DP+Greedy + None+CBBA
    "14 15 26 27"   # Job 7: Interconstellation, DP+Greedy + None+CBBA
    "16 17 28 29"   # Job 8: Intraconstellation, DP+Greedy + None+CBBA
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