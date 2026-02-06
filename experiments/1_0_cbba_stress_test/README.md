# CBBA Internal Validation Stress Test
*GOAL: showcase the reactivity capabilities of the sequence-constrained CBBA to incoming event requests and test edge cases*

## Experiment Formulation
### Research questions
1. How many requests can this CBBA handle?
    1. See runtime issues
    2. How can you guarantee or show convergence?
2. How efficiently does it assign tasks?
    1. How many messages are required to coordinate observations?
    2. How much agreement is actually achieved?

### Simulation Parameters
| Parameter | Values | Units |
|-----------|--------|-------|
| | |
| **Event Parameters** |  |  |
| Event Intensity | $\text{Uniform}(5.0,10.0)$ | - |
| Event Duration  | $\text{Uniform}(5,15)$ | [min] |
| Longitude Target Distributrion | $\text{Uniform}(-180, 180)$  | [deg]
| Target Location Condition | Inland Target | - |
| Decorrelation Time ($t_{corr}$) | $\text{Uniform}(1,10)$ | [min] |
| | |
| **Planner Parameters** | | |
| Replanning Threshold | 1 | [tasks] |
| Optimistic Bidding Threshold ($\nu$) | 1 | - |
| | |
| **Agent Capability Parameters** | | |
| Communications Range | $\text{LOS}$ | - |
| Maximum Slew Rate | $15$ | [deg/s] |
| Instrument | `IMG_A`, `IMG_B`, `IMG_C` | - |

### Test Matrices
#### Trial 1 - Stress Test in Fully Connected Network w/ Varying Latency
| Parameter | Values | Units |
|-----------|--------|-------|
| Number of Satellites | $12, 48, 96, 204$ | - |
| Constellation Connectivity Infrastructure | Alaska Satellite Facility, Full NEN , ISL | - | 
| Task arrival-rate ($\lambda$) | $10, 100, 500, 1000$ | [tasks / day] | 
| Latitude Target Distribution | $±25, ±60, ±90$ | [deg] |
| | |
**Total Cases:** 144

## Running Trials

### Commandline Arguments and Parameters
#### Trial Inputs
 - `t` or `--trials` : (`str`) filename of the `.csv` file outline the simulation trial cases to be ran
 - `i` or `--trial-start` : (`int`) lower bound of simulation trial indeces to be run (inclusive)
 - `j` or `--trial-end` : (`int`) upper bound of simulation trial indeces to be run (non-inclusive).
 - `--trial-range` followed by `START:END` : Trial range in format START:END (overrides start/end).
--- 
#### Profiling
 - `-m` or `--profile-mem` : (`flag`) Enables memory profiling (`tracemalloc`)
 - `-p` or `--profile-cpu` : (`flag`) Enables CPU profiling (`cProfile`)
 > Visualization of runtime profile requires the `snakeviz` library. Use command `snakeviz results/[NAME_OF_SCENARIO]/profile.out` to open results in default browser
--- 
#### Execution modes
 - `-1` or `--single-thread` : (`flag`) Run simulation single-threaded
 - `-r` or `--reduced` : (`flag`) Run reduced-complexity debug scenarios
--- 
#### Logging 
 - `-l` or `--logging-level` : (`str`) Loging level (must be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
 - `-q` or `--quiet` : (`flag`) Disable progress bars and console output (batch-safe mode)
--- 
#### Similation Stage Overwrite Controls
 - `-F` or `--force-all` : (`flag`) Force rerun of all stages of the simulation
 - `--forge-precompute` : (`flag`) Forces the repropagation of the precalculated orbital coverage data
 - `--forge-simulate` : (`flag`) Forces the simulation of the selected trials
 - `--forge-postprocess` : (`flag`) Forces the processing of the existing results of the selected trials
--- 
#### Simulation Stage Isolation
 - `--only-precompute` : (`flag`) Only performs the propagation of the precalculated orbital coverage data.May skip if already performed.
 - `--only-simulate` : (`flag`) Only performs the simulation of the selected trials. May skip if already performed.
 - `--only-postprocess` : (`flag`) Only processes the existing results of the selected trials. May skip if already performed.


### Example Trial Simulation Commands
The following are example commands to be used for running simulations in from a `Unix` terminal. Paths are listed for local directories within the machine used to develop this tool. For use in a separate computer, substitute `cd ~/Documents/GitHub/dmas-sync` with `cd path/to/repository/dmas-sync` when implementing in your own machine. 

#### Full Factorial Trials
*Runs simulations from full list of combinations from test matrix*

Simulate all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_trials
```

Only propagate orbits for all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_trials -p
```

Generate processed results for all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_trials -r 
```

Simulate cases from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_trials --trial-range START:END
```

Generate processed results for selected case from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_trials -r --trial-range START:END
```



#### Latin Hypercube Sampling Trials
*Cases selected from full list of combinations using a LHS $n_{sample} = 2$.*

Simulate all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t lhs_trials-2_samples-1000_seed
```

Simulate cases from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t lhs_trials-2_samples-1000_seed --trial-range START:END
```

Propagate only cases
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t lhs_trials-2_samples-1000_seed --only-precompute
```

In case you want to run the cases from full factorial that are not considered in the LHS cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_no_lhs --trial-range START:END
```
Propagate only cases
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -t full_factorial_no_lhs --only-precompute --force-precompute
```
<!-- ### Mission Definition
#### Mission 1 - Reactive Event Scheduling
##### Default Mission Objectives
*None*
##### Event-Driven Mission Objectives
1. Response Time
    - Pass
2. Revisit Time 
3. Co-Observation Time
4. Observation Number -->


## HPC Commands
*Commands for rnuning batch jobs at the TAMU's HPRC*

### Submit Batch 
#### Run Study
Runs all cases for the LHS trials in a linear manner.
```
cd $SCRATCH/src/dmas-sync
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_run_1.slurm
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_run_2.slurm
```

#### Run Parallellized Study
Runs all cases for the LHS trials in a parallellized process.
```
cd $SCRATCH/src/dmas-sync
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_run_parallel_1.slurm
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_run_parallel_2.slurm
```

#### Propagate Only
Performs orbit data propagation for all cases within the LHS trials.
```
cd $SCRATCH/src/dmas-sync
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_propagate_1.slurm
sbatch ./experiments/1_0_cbba_stress_test/jobs/cbba_study_propagate_2.slurm
```

### Monitor Batch Status
Enlist all current jobs and their status.
```
squeue -u $USER
```

Check status of a job with `<jobid>`
```
squeue --job <jobid>
```

Check the efficiency after `<jobid>` finishes
```
seff <jobid>
```         

Cancel a job with `<jobid>`
```
scancel <jobid>
```