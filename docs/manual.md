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
(`0.9 ** seconds`), which is not the same model. `ModelOptions.discounting` makes it explicit: `"decision"` (default) applies it per decision in all three methods,
`"second"` per second in all three, `"method"` keeps what each always did (so `ModelOptions.legacy()` reproduces the old Monte
Carlo). On `kochi2`, Monte Carlo with 0.9 per second learns nothing useful (a greedy policy worse than random), with 0.9 per
decision it does; see [audits/step2](./audits/step2/README.md).

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

## Building a case

`evacrl.casebuild` turns a road network and the places that are safe into the six tables the model reads, and checks them.

```
raw network ──merge short links──> network ──> actionsdb, transitionsdb, nextnode (shortest path), agentsdb ──> cases/<name>/
 (OSM)          (clusters)                      validated before anything is written
```

From the shell (`pip install -e ".[casebuild]"`; `from-raw` and `validate` need only NumPy and SciPy):

```
python -m evacrl.casebuild from-osm cases/my_area --areas areas.geojson --index 0 \
       --shelters shelters.geojson buildings.geojson --census census.geojson            # downloads the network
python -m evacrl.casebuild from-snapshot GRAPH_DIR cases/my_area ...                    # the same from a stored download
python -m evacrl.casebuild from-raw cases/my_area/raw cases/my_area --agents 5000       # offline, from its raw/ folder
python -m evacrl.casebuild validate cases/my_area                                       # check any case folder
```

Each case gets `data/` (what the model reads), `raw/` (the network before the clean-up, with its OSM ids) and `provenance.json`
(settings, counts, input checksums, versions). From Python the same steps are `evacrl.casebuild.build_case(raw, case_dir,
population=PopulationSpec(...))`, and each stage is a function (`merge_short_links`, `actions_and_transitions`, `next_nodes`,
`start_nodes`, `validate_tables`) that can be used alone.

| Setting | Default | Alternatives |
|---------|---------|--------------|
| `--merge` | `clusters`: nodes joined by links under `--threshold` (5 m) become one node | `legacy`: the 2024 study's pairwise merge (orphan nodes, broken chains) |
| `--strategy` | `uniform`: `--agents` at random start nodes | `per_node`: `--per-node` at every start node; `proportional`: `--agents` placed by the census (`--census`) |
| `--census-method` | `within`: cells entirely inside the area (the 2024 study; undercounts) | `weighted`: every cell touching the area, by the part inside |
| `--include-shelters` | off: agents do not start at a shelter | on: they may (the 2024 study; they count as evacuated at time 0) |
| `--shelters-as` | `attach`: a node at each shelter, joined to the nearest street node by a link as long as the distance between them (the walk counts, no shelter is moved); shelter points within 5 m of each other are one shelter | `snap`: the shelter *is* the nearest node (the 2024 study) |
| `--max-snap-distance` | none: every shelter inside the box of the network is used, however far from a node | metres: leave out shelters farther than this from every node (with `attach`: the longest access link) |
| `--excess` | `error`: a node with more than 10 links stops the build | `prune`: remove the longest links at such nodes (never a link into a shelter) |
| `--legacy` | | the 2024 study's merge, parallel-link rule, snapped shelters and agents at shelters, for reproducing its tables |

**Population layers.** `--census` takes any polygon layer with a numeric column, `--census-column` names it, and `--census-method weighted`
counts every cell that touches the area by the part inside. The Kochi census file is the standard 500 m mesh (about 460 × 580 m
there; 9-digit code in `MESH4_ID`) and has many columns besides `M_TOTPOP_H`; `--census-column M_DPOP_H22`, which by its name is a
daytime population (not checked against a data dictionary), gives 1,844 people for area 2 against 1,704 for `M_TOTPOP_H`. A layer
that is a table of mesh codes (as hourly mobile-phone population statistics usually are) has to be joined to polygons first, for
instance by `MESH4_ID`, and one hour chosen. Each cell's people are shared equally by the nodes inside it, and a cell without a node
goes to the node nearest its centre, so a coarse mesh gives a coarse distribution: a 500 m cell holds many nodes.

**Shelters.** A shelter point is rarely on a street node: in Kochi the median distance to the nearest node is 50–190 m, the
farthest 360 m. `snap` makes the shelter *be* the node, so reaching the node ends the walk of everybody who gets there, however
far the building really is. `attach` keeps the street node an ordinary node and adds a node at the shelter with one link to
it, as long as the straight-line distance (whole metres, like every link), so the walk counts. Details: the clean-up never
merges such a link, however short; a shelter closer than 1 m to a node is that node; shelter points within 5 m of each other are
one shelter; a shelter is not attached to another shelter or to a node without a street; the link has the default width of 3 m.
Limits: the length is the straight line to the nearest *node* (not the nearest point of the nearest street, and not a walking
distance), so it is a lower bound of the walk. `--merge legacy` (the 2024 clean-up, which cannot keep such a link) goes only with
`--shelters-as snap`. Measurements: [audits/step3](./audits/step3/README.md#the-shelters-a3-and-s2).

`validate` checks that the numbers run 0, 1, 2, ... in order; that links join existing nodes and have a length in whole metres;
that there is a shelter; that the actions and transitions are those of the links (each evacuation node has the single choice "stay");
that every next node is a neighbour and following them always ends at a shelter (it is a warning if a step is not on a shortest
walk); and that agents start at real nodes with a way out. Errors stop a build; warnings (isolated nodes, agents at a shelter, next nodes off the shortest walk) are
recorded in `provenance.json`. [audits/step3](./audits/step3/README.md) shows what it finds in the tables of the 2024 study.

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
| `discounting` | `"decision"` | `"method"` | `"method"` | `"decision"`, `"second"` or each method's own: see Discounting above |

[engine-reconciliation.md](./engine-reconciliation.md) says what each does to the results and which setting is recommended.
Results produced with `segmentIndex="raw"` (every result produced before the fix existed) can contain agents that stop at the end of a link and never evacuate.

### Shortest-path baseline

`loadShortestPathDB(file)` reads `nextnode.csv` (rows `[node, next node]`; `next node == node` at an evacuation node, `-9999`
where there is no path; a header line is optional). `checkTargetShortestPath()` then moves the agents along it with the same
speed rules as the learning agents, counts them in column 10 of `pedDB` when they arrive (`getNumberEvacuatedPed()`), and
stops an agent that has no path. Since the first step of every agent is random in all methods (`initEvacuationAtTime`), the
baseline is the shortest path *from the second node on*.
