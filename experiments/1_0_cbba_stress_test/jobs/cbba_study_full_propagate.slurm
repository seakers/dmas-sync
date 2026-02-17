#!/bin/bash
#SBATCH --job-name=cbba_stress_full_propagate_relays
#SBATCH --output=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.out
#SBATCH --error=experiments/1_0_cbba_stress_test/logs/cbba_stress_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-11

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
SCENARIOS=(1 13 25 37 49 61 73 85 97 109 121 133)

SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

echo "Running scenario $SCENARIO"

python ./experiments/1_0_cbba_stress_test/study.py -s $SCENARIO

# --- Run ---
python ./experiments/1_0_cbba_stress_test/study.py \
  -t full_factorial_trials \
  --trial-range "${start}:${end}" \
  --single-thread --quiet --exceptions \
  --force-precompute --only-precompute