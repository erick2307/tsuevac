# Manual to use the RL-TsuEvac model

## Overview of the model

The model consists of training nodes or intersections from a road network (`nodesdb.csv` and `linksdb.csv`) extracted from [OpenStreetMap](https://www.openstreetmap.org) to be able to optimally guide the evacuation of a group of agents or population (`agentsdb.csv`).

A database of actions (`actionsdb.csv`) is constructed out of the possible directions to be taken by an evacuee at an intersection or node in the network. Thus, the matrix is as follows:

$A=[N,M_N,l_i...l_m]$

where, $A$ is the action matrix that contains a set $N$ of $n$ nodes linked to $m$ other nodes through a set of links $l_i ... l_m$. This matrix maps the transition from one node through a particular link.

Similarly, a `transitionsdb.csv` matrix is made to map the possible transitions from one node to another. This is of the form:

$T=[N,M_N,n_i ... n_m]$

where, $T$ is the transition matrix between nodes, and $n_i ... n_m$ are the nodes available from a particular node $n$ in the set of nodes $N$.

## Entry points (`main_*.py`)

Each method has its own entry point in the root directory, importing its class from the matching module:

| Script | Class (module) | Method |
|--------|----------------|--------|
| `main_ql.py`, `main_ql_mod.py` | `QLearning` (`qlearn.py`) | Q-learning |
| `main_sarsa.py` | `SARSA` (`sarsa.py`) | SARSA |
| `main_mc.py` | `MonteCarlo` (`mc.py`) | Monte Carlo |
| `main_ShortPath.py` | `QLearning`, `MonteCarlo` | Shortest-path baseline (no learning) |

Run a script from the repository root; the case to run is chosen in its `__main__` block (e.g. `kochi_ql()`, `arahama_sarsa()`).
The newer workflow in `app/` has its own `main.py`, run from inside `app/`.

The files required as input, for an `<area>`, are in `<area>/data/` (their format is described in the [README](./README.md)):   
* `agentsdb.csv`  
* `nodesdb.csv`  
* `linksdb.csv`  
* `actionsdb.csv`  
* `transitionsdb.csv`

The parameters of the `run_*` functions are:  
* `area` .- The study area, i.e. the folder with the input data (`kochi`, `arahama`, `new_kochi`).  
* `simtime` .- Simulated time in minutes.  
* `meandeparture` .- Mean departure time in minutes (see `meanRayleigh` below).  
* `numSim0`, `numBlocks`, `simPerBlock` .- Index of the first simulation, number of blocks and simulations per block.  
* `name` .- Suffix of the folder (`<area>/state_<name>`) where the states are stored.  

The parameters needed by the classes are:  
* `meanRayleigh` .- This is the mean value of a Rayleigh distribution for the evacuation departure time decision (in seconds).  
* `folderStateNames` .- A name of a folder to store states explored during the learning process.
