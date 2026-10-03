# Tsunami Evacuation using Reinforcement Learning

> This document explains the structure of the repository.
> For an explanation on the use of the code see the [Manual](./docs/manual.md).
> Notes on pre-processing and tsunami data are in [docs/preprocessing.md](./docs/preprocessing.md).

## Repository layout

```
cases/               one folder per study area: inputs in data/, run outputs in state_<name>/ (not tracked)
  kochi/             old Kochi network (4,315 nodes)
  new_kochi/         Kochi network with 19,207 nodes, plus its pre-processing pipeline
src/evacrl/          importable code: qlearn.py, sarsa.py, mc.py, evac_plots.py, make_video.py, paths.py
scripts/             entry points: main_ql.py, main_ql_mod.py, main_sarsa.py, main_mc.py, main_ShortPath.py
notebooks/           analysis notebooks
tests/               golden regression tests
docs/                manual, pre-processing notes, diagrams
datasets/            large raw / shared inputs: gis/ (QGIS data and its notebook), legacy/ (older samples)
pre/                 pre-processing and census scripts, with the census data
variants/app_2022/   self-contained 2022 workflow (its own qlearn.py and setup pipeline)
experimental/        new_model/ (object-oriented rewrite), tdcontrol.py, tests_mc.py
results/, weights/   outputs shared by the notebooks
```

| Path | What it is |
|------|------------|
| `scripts/main_*.py` | Entry points. Each one defines `run_*` plus one helper per area (`kochi_*`, `arahama_*`, `new_kochi_*`); pick the case in the `__main__` block. `main_ql_mod.py` is a variant of `main_ql.py` that reloads the best-performing state matrix at the start of each block. `main_ShortPath.py` is the shortest-path baseline (no learning). |
| `src/evacrl/` | The `QLearning`, `SARSA` and `MonteCarlo` classes, `evac_plots.py` (evacuation curves per epoch), `make_video.py` (AVI of a particular epoch) and `paths.py`. |
| `src/evacrl/paths.py` | **The only place that knows the layout.** Case data, `figures/`, `weights/` and `results/` are all resolved through it, so moving a folder means editing this file. |
| `cases/new_kochi/` | Besides its data, holds the pre-processing pipeline of the case (`preProcess.py`, `createLinksAndNodes.py`, `getPopulation.py`, `setActionsAndTransitions.py`, `preprocess.ipynb`); run it from inside this folder (it uses `./data` and `./tmp`). |
| `notebooks/` | `check_policies`, `plot_survivors` (compare policies, survivors per simulation), `operation_*` (batches of runs: survivors vs. simulation and departure time), `CalculateWeights` (link weights from SARSA). Each starts with a bootstrap cell that finds the repository, so they run from any directory. |
| `tests/` | `test_golden_ql.py`: regression tests that run short Q-learning simulations on both Kochi networks and compare them with recorded results. Run them before and after any restructuring. |
| `variants/app_2022/` | Newer, self-contained version of the workflow: `main.py` (Q-learning, uses `bin/qlearn.py`), `setup/` (builds a case from an area-of-study GeoJSON in `input/`), `make_video.py`, `analysis.ipynb`. Run from inside the folder. |
| `experimental/new_model/` | Work-in-progress object-oriented rewrite (`tsuevac` package: `Environment`, `Agent`, `Evacuee`, `Node`, `Shelter`, `Model`). Most methods are still stubs. |
| `datasets/gis/` | GIS data (QGIS projects, rasters) and figures for the tsunami inundation / road network, plus the notebook that reads them (it uses `./data`, so it stays next to the data). |
| `datasets/legacy/` | Older samples: `kochi_old/` (state and results in the 31-column layout) and two evacuee start/end/departure tables. |
| `experimental/` | Also `tdcontrol.py` (toy TD-control skeleton) and `tests_mc.py` (ad-hoc runs of `mc.py`). |
| `results/`, `weights/` | Sample outputs of old Kochi runs; also where the notebooks and `computeWeightsAtLinks` read and write. |

## Quick start

```
pip install -e .                    # optional: makes `evacrl` importable from anywhere
python scripts/main_ql_mod.py       # runs the case chosen in its __main__ block, from any directory
python -m unittest discover tests   # regression tests (about 25 s)
```

The scripts, the tests and the notebooks also work without installing the package.
Dependencies: `numpy`, `matplotlib`, `opencv-python`; `evac_plots.py` and the notebooks also use `scipy` and `pandas`.
Input status of the cases: `cases/new_kochi/data` is complete; `cases/kochi/data` has everything except the real `agentsdb.csv`. `arahama` is not part of this repository.

### Two code stacks (not interchangeable)

`src/evacrl` (the root stack) and `variants/app_2022/` each carry their own `qlearn.py`. They look similar but are **not** duplicates:

| | Root stack (`scripts/main_*.py`, `src/evacrl/qlearn.py`) | `variants/app_2022/` stack (`main.py`, `bin/qlearn.py`, `setup/`) |
|---|---|---|
| Actions/transitions DB | 12 columns, at most 10 links per node (`cases/new_kochi/setActionsAndTransitions.py`) | 20 columns (`variants/app_2022/setup/lib/setActionsAndTransitions.py`) |
| State matrix (`state_*/sim_*.csv`) | fixed 31 columns | `3 x (actions DB width) + 1` columns |
| Input header line | read as a `#` comment | first line always skipped (`skiprows=1`) |

State files written by one stack cannot be loaded by the other (for example `datasets/legacy/kochi_old/state/` is in the 31-column layout).

## Input data (`cases/<area>/data/`)

Header lines start with `#`, so `numpy.loadtxt` treats them as comments.

`agentsdb.csv` => input data of population  
* `age`, `gender`, `hhType`, `hhId`: agent profile  
* `Node`: starting node (int); this is the field the simulation currently uses  

`linksdb.csv` => edges or links of a road network  
* `number`: the ID of the link/edge (int)  
* `node1`: starting node of the edge (int)  
* `node2`: ending node of the edge (int)  
* `length`: length of the edge in meters (int)  
* `width`: width of the road in meters (int)  

`nodesdb.csv` => nodes information (intersections of roads)  
* `number`: node ID  
* `coord_x`, `coord_y`: node coordinates  
* `evacuation`: flag (0=common node; 1=evacuation point)  
* `reward`: abs of the penalty 'reward' given at each node (-1 to account for time pressure in evacuation)  

`actionsdb.csv`, `transitionsdb.csv` => created from the nodes and links with
`setActionsAndTransitions.py` (`cases/new_kochi/setActionsAndTransitions.py` for the root stack, `variants/app_2022/setup/lib/setActionsAndTransitions.py` for the `variants/app_2022/` stack).

## `pre/` directory

The census folders stay here for now: `SetPopDB.py` addresses them by relative name. They move to `datasets/census/` together with that script's path fix.

* `CensusAndBuildingDatabase`, `Household_database`, `Population_database` are folders with the census data (the household database is integrated but not in use at the moment).  
* `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py`, `SetPopDB.py` => population disaggregation (see [docs/preprocessing.md](./docs/preprocessing.md)).  
* `defPathsFromNodes.py` => a function to calculate the next node for a pre-determined shortest path run.  
* `DetectionShelters.py` => to detect evacuation points from the network.  
* `makeUniformPedestrianProfiles.py` => creates an `agentsdb`-style file with a fixed number of agents per (non-evacuation) node.  
* `tests.ipynb` => scratch notebook (uses `variants/app_2022/setup/lib/getPopulation.py`).  

The older `pre/getPopulation.py`, `pre/SetActionsAndTransitions.py` and `pre/lib_ImportOSM.py` were removed because they were superseded
by `variants/app_2022/setup/lib/getPopulation.py`, `cases/new_kochi/setActionsAndTransitions.py` and `cases/new_kochi/createLinksAndNodes.py` (OSM edges to nodes/links DB). They remain available in git history.

## Repository conventions

* Simulation outputs (`cases/*/state_*/`, `figures/`, `weights/w_*.csv`, `*.avi`) and large local GIS data are not tracked; see [`.gitignore`](./.gitignore).
* Input CSVs and images are **not** ignored: commit any `data/*.csv` a case needs to run.
* Locations live in `src/evacrl/paths.py`; do not build `<area>/data/...` paths from the working directory.
* Regression tests: `python -m unittest discover tests`. `GOLDEN_STRICT=1` additionally compares every output file byte for byte (same NumPy/Python only). After an intentional change of behaviour, regenerate the recorded results with `UPDATE_GOLDEN=1 python tests/test_golden_ql.py`.
