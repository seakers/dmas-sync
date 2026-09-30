# Decentralized Multi-Agent Satellite Simulation (DMAS)

**DMAS** is a simulation platform for decentralized and distributed satellite systems. It is meant to test and showcase
novel Earth-observing satellite mission concepts that use higher levels of autonomy, from environment detection to
autonomous operations planning. It is the framework used to evaluate the SC-CBBA and to compare centralized and
decentralized task allocation in the SEAK Lab dissertation work.

This implementation simulates a distributed sensor web Earth-observation system described in the NASA AIST 3D-CHESS
project, which aims to demonstrate a new Earth observing strategy based on a context-aware Earth observing sensor web.
The sensor web consists of nodes with a knowledge base, heterogeneous sensors, edge computing, and autonomous
decision-making capabilities. Context awareness is the ability of the nodes to gather, exchange, and leverage contextual
information (e.g., state of the Earth system, state and capabilities of itself and of other nodes, and how those states
relate to dynamic mission objectives) to improve decision making and planning. The current goal is to demonstrate proof
of concept by comparing a 3D-CHESS sensor web against status-quo architectures in a multi-sensor inland hydrologic and
ecologic monitoring system.

## How it works (30-second version)
- Each satellite (or ground operator) is a `SimulationAgent` with an optional **preplanner** and **replanner**.
- `Simulation.execute()` runs a synchronous, event-driven loop: the environment applies every agent's current action and
  returns percepts, each agent decides its next action, and mission time jumps to the earliest action end time.
- **Planning takes no mission time.** Only observe, maneuver, and wait actions advance the clock, so planner
  computation costs wall-clock time but not simulated time.
- Communication is governed by precomputed connectivity intervals; broadcasts reach the agents in the sender's current
  connected component.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for diagrams.

## Repository layout
```
├── dmas/                     main source code
│   ├── core/                 Simulation (execution loop, config -> agents), Environment, messages
│   ├── models/               agent modeling
│   │   ├── agent.py          SimulationAgent.decide_action()
│   │   ├── actions.py        observe / maneuver / broadcast / wait / ...
│   │   ├── planning/         planners (see below)
│   │   └── science/          onboard data processing and request models
│   ├── utils/                orbit data, constellation generation, result processing
│   └── network/              asynchronous distributed wrapper (UNDER DEVELOPMENT, not used by experiments)
├── experiments/              self-contained studies: templates, trials, run scripts, analysis
│   ├── 1_cbba_validation/
│   └── 2_centralized_vs_decentralized/
├── tests/                    unit tests (agents, connectivity, core, planners, tasks)
└── docs/                     architecture docs and diagrams
```

## Available planners
Set in an agent's `planner` block; `@type` selects the class (see `Simulation.__load_preplanner` / `__load_replanner`).

| Role | `@type` | Class | Notes |
|---|---|---|---|
| Preplanner | `earliest` (`naive`, `fifo`) | `EarliestAccessPeriodicPlanner` | Greedy, earliest feasible access |
| Preplanner | `heuristic` | `HeuristicInsertionPeriodicPlanner` | Heuristic insertion |
| Preplanner | `dp` (`dynamic`) | `DynamicProgrammingPlanner` | Dynamic programming |
| Preplanner | `nadir` | `NadirPointingPeriodicPlanner` | Nadir-pointing baseline |
| Preplanner | `announcer` (`@mode`: `oracle` / `groundprocessor`) | `Instant/GroundProcessorEventAnnouncerPlanner` | Event announcement |
| Preplanner | `dealer` (`@mode`: `test` / `milp`) + `worker` | `DealerMILPPlanner`, `WorkerPlanner` | Centralized (MILP needs a Gurobi license) |
| Preplanner | `blank` | `BlankPlanner` | Does nothing; testing |
| Replanner | `consensus` / `cbba` (+ `augmented`) | `HeuristicInsertionConsensusPlanner` | SC-CBBA |
| Replanner | `heuristic`, `earliest`, `nadir` | `*ReactivePlanner` | Greedy plan repair |
| Replanner | `default` | `FixedPointingDefaultPlanner` | Fixed pointing |

## Install
**Requirements:** Python 3.8, [miniconda](https://docs.conda.io/en/latest/miniconda.html),
[`gfortran`](https://fortran-lang.org/learn/os_setup/install_gfortran), and `make`.

1. Install [`instrupy`](https://github.com/Aslan15/instrupy), [`orbitpy`](https://github.com/Aslan15/orbitpy), and
   [`execsatm`](https://github.com/seakers/execsatm).
2. Create and activate a conda environment:
   ```
   conda create -p desired/path/to/virtual/environment python=3.8
   conda activate desired/path/to/virtual/environment
   ```
3. From the repository root, install `dmas` (editable):
   ```
   make
   ```
4. Run the tests (optional):
   ```
   make runtest
   ```

> - Installation is supported on Mac and Linux. On Windows, use WSL.
> - Mac users may have trouble installing the `propcov` dependency inside `orbitpy`; see
>   [orbitpy's installation notes](https://github.com/EarthObservationSimulator/orbitpy/tree/master/propcov).
> - For Windows development, VS Code's remote development in WSL was used
>   ([instructions](https://code.visualstudio.com/docs/remote/wsl-tutorial)).

## Running experiments
Each folder under `experiments/` is a self-contained study with its own README, trial CSVs, templates
(`resources/templates/`), `study.py` entry point, SLURM job scripts, and analysis code. Start with the README in the
study you want to reproduce; it lists the command-line arguments (trial range, single-thread, profiling, reduced mode).

## Developing your own planner
1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
2. Follow [docs/ADDING_A_PLANNER.md](docs/ADDING_A_PLANNER.md): subclass `AbstractPeriodicPlanner` or
   `AbstractReactivePlanner`, register it in `core/simulation.py`, add it to a `planners.json` template, and test it
   with the harness in `tests/planners/`.

## Known limitations
- Ground sensor agents are not implemented (`NotImplementedError`).
- The `network/` asynchronous wrapper is under development.
- Mac installation of `propcov` is fragile (see above).

## License and Copyright
Copyright (c) 2026 Systems Engineering Architecture and Knowledge Lab. Licensed under the MIT License; see
[LICENSE](LICENSE).

## Acknowledgments
This work was supported by the National Aeronautics and Space Administration (NASA) Earth Science Technology Office
(ESTO) through the Advanced Information Systems Technology (AIST) Program, and by the Mexican Ministry of Science,
Humanities, Technology, and Innovation (SECIHTI) through its Graduate Scholarships for Studies in Science and
Humanities Abroad Fellowship.

## Contact
**Principal Investigator:** Daniel Selva Valero - dselva@tamu.edu
**Lead Developers:** Alan Aguilar Jaramillo - aguilaraj15@tamu.edu, Ben Gorr - bgorr@tamu.edu
