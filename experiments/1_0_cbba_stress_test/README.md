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
- `-n` or `--name` : (`str`) filename of `.csv` file outlining simulation trial cases to be ran,
- `-l` or `--lower-bound` : (`int`) lower bound of simulation trial indeces to be run (inclusive).
- `-u` or `--upper-bound` : (`int`) upper bound of simulation trial indeces to be run (non-inclusive).
- `-p` or `--propagate` : (`flag`) only performs orbit propagation of trial cases, does not simulate satellite mission. 
- `-o` or `--overwrite` : (`flag`) runs simulation trials and overwrites existing results. If not flagged, trials with existing results will not be simulated.
- `-e` or `--evaluate` : (`flag`) runs processing of simulation results and overwrites existing results. If not flagged, trials with existing results will not be reevaluated.
- `-d` or `--debug` : (`flag`) toggles the use of reduced complexity simulation trials for debugging purposes.
- `-r` or `--runtime-profiling` : (`flag`) toggles the profiling of each trial's runtime. Prints results to `results/[NAME_OF_SCENARIO]/profile.out`
    > Visualization of runtime profile requires the `snakeviz` library. Use command `snakeviz results/[NAME_OF_SCENARIO]/profile.out` to open results in default browser


### Trial Simulation Commands
Example commands to be used for running simulations in from a `Unix` terminal. Paths are listed for the local directories of the machine used to develop this tool. Substitute `cd ~/Documents/GitHub/dmas-sync` with `cd path/to/repository/3dchess` when implementing in your own machine. 

#### Full Factorial Trials
*Runs simulations from full list of combinations from test matrix*

Simulate all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials
```

Only propagate orbits for all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials -p
```

Generate processed results for all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials -r 
```

Simulate cases from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials -l LOWER -u UPPER
```

Generate processed results for selected case from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials -r -l LOWER -u UPPER
```



#### Latin Hypercube Sampling Trials
*Cases selected from full list of combinations using a LHS $n_{sample} = 2$.*

Simulate all cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n lhs_trials-2_samples-1000_seed
```

Simulate cases from trial interval `[LOWER, UPPER)`:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n lhs_trials-2_samples-1000_seed -l LOWER -u UPPER
```

Propagate only cases
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n lhs_trials-2_samples-1000_seed -p 
```

In case you want to run the cases from full factorial that are not considered in the LHS cases:
```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_no_lhs -l LOWER -u UPPER
```

```
cd ~/Documents/GitHub/dmas-sync
conda activate ./.venv
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_no_lhs -p 
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


