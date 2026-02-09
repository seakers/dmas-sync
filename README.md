# Decentralized Multi-Agent Satellite Simulation (DMAS) 

**DMAS** is a simulation platform for decentralized and distributed satellite systems.
Meant to test and showcase novel Earth-observing satellite mission concepts using higher levels of autonomy. 
This autonomy ranges from environment detection to autonomous operations planning.

This implementation simulates a distributed sensor web Earth-Observation system described in the NASA AIST 3D-CHESS project which aims to demonstrate a new Earth
observing strategy based on a context-aware Earth observing
sensor web. This sensor web consists of a set of nodes with a
knowledge base, heterogeneous sensors, edge computing,
and autonomous decision-making capabilities. Context
awareness is defined as the ability for the nodes to gather,
exchange, and leverage contextual information (e.g., state of
the Earth system, state and capabilities of itself and of other
nodes in the network, and how those states relate to the dy-
namic mission objectives) to improve decision making and
planning. The current goal of the project is to demonstrate
proof of concept by comparing the performance of a 3D-
CHESS sensor web with that of status quo architectures in
the context of a multi-sensor inland hydrologic and ecologic
monitoring system.

## Directory structure
```
├───docs (sphinx and other project-related documentation)
|   └───diagrams (figures used for documentation)
|       ├───architecture (simulation architecture diagrams)
|       └───sequence diagrams (simulation sequence)
├───experiments (folder with the input and results files for simulations experiments)
├───tests (unit tests)
└───dmas (folder with main source code)
    ├───core (main simulation wrapper for synchronous execution)
    ├───network (distributed simulation wrapper for asynchronous execution UNDER DEV)
    ├───utils (useful resources for simulation modeling and execution)
    └───models (agent modeling resources)
        ├───planning (planning-based autonomous decision-making tools for agents)
        └───science (onboard data processing modeling tools)
```

<!-- 
---
## Documentation
For documentation please visit: https://dmas.readthedocs.io/ 
-->

## Install
**Requirements:** Python 3.8, [miniconda](https://docs.conda.io/en/latest/miniconda.html),  [`gfortran`](https://fortran-lang.org/learn/os_setup/install_gfortran), and [`make`](https://fortran-lang.org/learn/os_setup/install_gfortran). 

1. Install [`instrupy`](https://github.com/Aslan15/instrupy), [`orbitpy`](https://github.com/Aslan15/orbitpy), and [`execsatm`](https://github.com/seakers/execsatm) libraries.

2. Create and activate a virtual conda environment:

```
conda create -p desired/path/to/virtual/environment python=3.8

conda activate desired/path/to/virtual/environment
```

4. Install `dmas` library by running `make` command in terminal in repository directory:
```
make 
```

5. Run tests (optional)
```
make runtest
```

> ### NOTE: 
> - **Installation instructions above are only supported in Mac or Linux systems.** For windows installation, use a Windows Subsystem for Linux (WSL) and follow the instructions above.
> - Mac users are known to experience issues installing the `propcov` dependency contained within the `orbitpy` library during installation. See [`orbitpy`'s installation notes](https://github.com/EarthObservationSimulator/orbitpy/tree/master/propcov) for fixes.
> - For development in Windows, Visual Studio Code's remote development feature in WSL was used. See [instructions for remote development in VSCode](https://code.visualstudio.com/docs/remote/wsl-tutorial) for more details on WSL installation and remote development environment set-up.

## License and Copyright
Copyright (c) 2026 Systems Engineering Architecture and Knowledge Lab

This project is licensed under the MIT License, a permissive open-source license that allows
use, modification, and distribution with minimal restrictions. See the [LICENSE](LICENSE) file for details.


## Acknowledgments
This work was supported by the National Aeronautics and Space Administration (NASA) Earth Science Technology Office (ESTO) through the Advanced Information Systems Technology (AIST) Program, and by the Mexican Ministry of Science, Humanities, Technology, and Innovation (SECIHTI) through its Graduate Scholarships for Studies in Science and Humanities Abroad Fellowship.

## Contact 
**Principal Investigator:** 
- Daniel Selva Valero - <dselva@tamu.edu>

**Lead Developers:** 
- Alan Aguilar Jaramillo - <aguilaraj15@tamu.edu>
- Ben Gorr - <bgorr@tamu.edu>