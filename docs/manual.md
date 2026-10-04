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

Each method has its own entry point in `scripts/`, importing its class from the matching module:

| Script | Class (module) | Method |
|--------|----------------|--------|
| `scripts/main_ql.py`, `scripts/main_ql_mod.py` | `QLearning` (`src/evacrl/qlearn.py`) | Q-learning (off-policy) |
| `scripts/main_sarsa.py` | `SARSA` (`src/evacrl/sarsa.py`) | SARSA |
| `scripts/main_mc.py` | `MonteCarlo` (`src/evacrl/mc.py`) | Monte Carlo |
| `scripts/main_ShortPath.py` | `QLearning`, `MonteCarlo` | Shortest-path baseline (no learning) |

### The three methods share one engine

`src/evacrl/core.py` defines `EvacuationModel`: the road network and agents, the pedestrian dynamics, the state matrix
(`[node, 10 density codes, 10 action values, 10 visit counts]`), the choice of the next node, the shortest-path baseline and
the plots and videos. The methods differ only in **when and how the action values are updated**, which is one hook,
`tdControl(pedIndx)`, called every time an agent chooses its next node:

| Class | `tdControl` | Learns |
|-------|-------------|--------|
| `MonteCarlo` | not overridden (does nothing) | once per simulation, in `updateValueFunctionDB` |
| `SARSA` | `Q(S0,A0) += alpha*(stepReward*dt + discount*Q(S,A) - Q(S0,A0))`, plus `surviveReward` on arrival | during the simulation (on-policy) |
| `QLearning` | `Q(S0,A0) += alpha*(stepReward*dt + discount*max_a Q(S,a) - Q(S0,A0))`, plus `surviveReward` on arrival | during the simulation (off-policy) |

`SARSA` and `QLearning` share the bookkeeping (`src/evacrl/td.py`, class `TemporalDifference`) and differ in one method,
`bootstrapValue(S, A)`: SARSA returns `Q(S,A)` of the action the agent actually chose, exploration included; Q-learning returns the
maximum of `Q(S,.)` over the actions that exist at that node (`transLinkdb[node, 1]` of the 10 slots: the unused ones hold 0
and would beat every negative value). Reaching an evacuation node ends the episode: `Q(S,A)` is moved towards `surviveReward`
and nothing is bootstrapped from beyond it. When agents always choose the best action (no exploration) the two updates are
identical to the last digit, which `tests/test_td.py` checks; with exploration they differ.

All four entry points explore with a decaying rate (`randomChoiceRate` is 0.99 in the first simulation, then `1 - (s/eoe)^2` for
the first 80% of the simulations of a block, then 0). Before the off-policy update was added, `QLearning` was `SARSA` with no code of its own, and `main_ql.py`
forced `randomChoiceRate = 0` (marked "added to check if this is Q-Learning") so that its greedy choices made SARSA's target
Q-learning's; both are gone now that the update is real. The results of every entry point are protected by the golden tests in
`tests/`; the SARSA and Monte Carlo runs are also pinned with `ModelOptions.legacy()` to the recordings of the original code.

**Discounting.** `discount` (0.9) is applied per *decision* in SARSA and Q-learning and per *second* in Monte Carlo
(`0.9 ** seconds`), which is not the same model. `ModelOptions.discounting` makes it explicit: `"method"` (default) keeps
what each method always did, `"decision"` and `"second"` apply one rule to all. On `kochi2`, Monte Carlo with 0.9 per second
learns nothing useful (a greedy policy worse than random), with 0.9 per decision it does; see
[audits/step2](./audits/step2/README.md).

The parameters of the `run_*` functions are:  
* `area` .- The study area, i.e. the folder with the input data (`kochi`, `arahama`, `new_kochi`).  
* `simtime` .- Simulated time in minutes.  
* `meandeparture` .- Mean departure time in minutes (see `meanRayleigh` below).  
* `numSim0`, `numBlocks`, `simPerBlock` .- Index of the first simulation, number of blocks and simulations per block.  
* `name` .- Suffix of the folder (`cases/<area>/state_<name>`) where the states are stored.  

The parameters needed by the classes are:  
* `meanRayleigh` .- This is the mean value of a Rayleigh distribution for the evacuation departure time decision (in seconds).  
* `folderStateNames` .- A name of a folder to store states explored during the learning process.  
* `options` .- A `ModelOptions` (`src/evacrl/options.py`); default `ModelOptions()`, the recommended settings, see below. Every `run_*` function takes it too.

## Model options

`evacrl.options.ModelOptions` collects the behaviours that differ between the 2021 code and the 2024 Kochi study
(`erick2307/2024_urushibara`). `ModelOptions()` holds the recommended settings; `ModelOptions.legacy()` and
`ModelOptions.kochi2024()` are the two older behaviours, fully explicit so that they do not move when the defaults do.
Pass it as `options=` to any of the classes or `run_*` functions:

```python
from evacrl.options import ModelOptions
from evacrl.qlearn import QLearning

QLearning(..., options=ModelOptions())                     # the defaults
QLearning(..., options=ModelOptions.legacy())              # the 2021 behaviour, including its far-end defect
QLearning(..., options=ModelOptions.kochi2024())           # EVACMODEL3_FocalPoints/SARSA2024.py, bit for bit (for SARSA)
QLearning(..., options=ModelOptions(surviveReward=10**7))  # a default with one change
```

| Option | Default | 2021 (`legacy()`) | 2024 (`kochi2024()`) | What it changes |
|--------|---------|------|------|-----------------|
| `surviveReward` | `100000` | `100000` | `10000000` | reward on reaching an evacuation node |
| `densityLevel` | `"link"` | `"link"` | `"segment"` | density code of a link in the state: whole link, fixed 2 m width / worst segment, real width |
| `entrySpeed` | `"position"` | `"first_segment"` | `"position"` | speed when entering a link: segment where the agent is / first segment |
| `segmentSizing` | `"ceil"` | `"ceil"` | `"round"` | segments of about 2 m per link |
| `segmentIndex` | `"clamped"` | `"raw"` | `"raw"` | `"clamped"` fixes agents freezing at the far end of a link (present in both older codes) |
| `discounting` | `"method"` | `"method"` | `"method"` | `"decision"`, `"second"` or each method's own: see Discounting above |

[engine-reconciliation.md](./engine-reconciliation.md) says what each does to the results and which setting is recommended.
Results produced with `segmentIndex="raw"` (every result produced before the fix existed) can contain agents that stop at the end of a link and never evacuate.

### Shortest-path baseline

`loadShortestPathDB(file)` reads `nextnode.csv` (rows `[node, next node]`; `next node == node` at an evacuation node, `-9999`
where there is no path; a header line is optional). `checkTargetShortestPath()` then moves the agents along it with the same
speed rules as the learning agents, counts them in column 10 of `pedDB` when they arrive (`getNumberEvacuatedPed()`), and
stops an agent that has no path. Since the first step of every agent is random in all methods (`initEvacuationAtTime`), the
baseline is the shortest path *from the second node on*.
