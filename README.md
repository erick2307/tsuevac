# Tsunami Evacuation using Reinforcement Learning

> This document explains the structure of the repository.
> For an explanation on the use of the code see the [Manual](./docs/manual.md).
> Notes on pre-processing and tsunami data are in [docs/preprocessing.md](./docs/preprocessing.md).

## Repository layout

| Path | What it is |
|------|------------|
| `main_ql.py`, `main_ql_mod.py`, `main_sarsa.py`, `main_mc.py`, `main_ShortPath.py` | Entry points. Each one defines `run_*` plus one helper per area (`kochi_*`, `arahama_*`, `new_kochi_*`); pick the case in the `__main__` block. `main_ql_mod.py` is a variant of `main_ql.py` that reloads the best-performing state matrix at the start of each block. `main_ShortPath.py` is the shortest-path baseline (no learning). |
| `qlearn.py`, `sarsa.py`, `mc.py` | The `QLearning`, `SARSA` and `MonteCarlo` classes. |
| `evac_plots.py` | Plot evacuation curves from each epoch. |
| `make_video.py` | Create a video (AVI) of a particular epoch (state matrix or policy). |
| `tdcontrol.py`, `tests_mc.py` | Toy TD-control skeleton and ad-hoc tests for `mc.py`. |
| `check_policies.ipynb`, `plot_survivors.ipynb` | Notebooks to compare policies and plot survivors per simulation. |
| `kochi/`, `new_kochi/`, `arahama/` | One folder per study area. Inputs live in `<area>/data/`; runs write to `<area>/state_<name>/` (not tracked, see `.gitignore`). `arahama/` is not part of this repository. Input status: `kochi/data` has everything except the real `agentsdb.csv`; `new_kochi/data` has everything except `linksdb.csv`. |
| `tests/` | `test_golden_ql.py`: regression test that runs short Q-learning simulations on the Kochi network (synthetic population in `tests/fixtures/`) and compares them with recorded results. Run it before and after any restructuring. |
| `new_kochi/` | Also contains the pre-processing pipeline for the Kochi case (`preProcess.py`, `createLinksAndNodes.py`, `getPopulation.py`, `setActionsAndTransitions.py`, `preprocess.ipynb`). |
| `variants/app_2022/` (formerly `app/`) | Newer, self-contained version of the workflow: `main.py` (Q-learning, uses `bin/qlearn.py`), `setup/` (builds a case from an area-of-study GeoJSON in `input/`), `make_video.py`, `analysis.ipynb`. |
| `experimental/new_model/` (formerly `new_model/`) | Work-in-progress object-oriented rewrite (`tsuevac` package: `Environment`, `Agent`, `Evacuee`, `Node`, `Shelter`, `Model`). Most methods are still stubs. |
| `pre/` | Original pre-processing scripts and census/population data (see below). |
| `datasets/gis/` (formerly `system/`) | GIS data (QGIS projects, rasters) and figures for the tsunami inundation / road network, plus the notebook that reads them (it uses `./data`, so it stays next to the data). |
| `docs/` | `manual.md`, `preprocessing.md` (formerly `tegs.md`) and `diagrams/` (draw.io flow charts). |
| `database/` | Notebooks that analyse batches of runs (survivors vs. simulation time and mean departure time). |
| `other/` | Informal notebook for various calculations (e.g. weights from SARSA) and sample outputs. |
| `results/`, `weights/` | Sample outputs. |

Scripts are meant to be run from the repository root, because inputs and outputs
are addressed as `<area>/data/...` and `<area>/state_<name>/...`.

### Two code stacks (not interchangeable)

The root scripts and `variants/app_2022/` each carry their own `qlearn.py`. They look similar but are **not** duplicates:

| | Root stack (`main_*.py`, `qlearn.py`, `make_video.py`) | `variants/app_2022/` stack (`main.py`, `bin/qlearn.py`, `setup/`) |
|---|---|---|
| Actions/transitions DB | 12 columns, at most 10 links per node (`new_kochi/setActionsAndTransitions.py`) | 20 columns (`variants/app_2022/setup/lib/setActionsAndTransitions.py`) |
| State matrix (`state_*/sim_*.csv`) | fixed 31 columns | `3 x (actions DB width) + 1` columns |
| Input header line | read as a `#` comment | first line always skipped (`skiprows=1`) |

State files written by one stack cannot be loaded by the other (for example `other/kochi_old/state/` is in the 31-column layout).

## Input data (`<area>/data/`)

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
`setActionsAndTransitions.py` (`new_kochi/setActionsAndTransitions.py` for the root stack, `variants/app_2022/setup/lib/setActionsAndTransitions.py` for the `variants/app_2022/` stack).

## `pre/` directory

The census folders stay here for now: `SetPopDB.py` addresses them by relative name. They move to `datasets/census/` together with that script's path fix.

* `CensusAndBuildingDatabase`, `Household_database`, `Population_database` are folders with the census data (the household database is integrated but not in use at the moment).  
* `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py`, `SetPopDB.py` => population disaggregation (see [docs/preprocessing.md](./docs/preprocessing.md)).  
* `defPathsFromNodes.py` => a function to calculate the next node for a pre-determined shortest path run.  
* `DetectionShelters.py` => to detect evacuation points from the network.  
* `makeUniformPedestrianProfiles.py` => creates an `agentsdb`-style file with a fixed number of agents per (non-evacuation) node.  
* `tests.ipynb` => scratch notebook (uses `variants/app_2022/setup/lib/getPopulation.py`).  

The older `pre/getPopulation.py`, `pre/SetActionsAndTransitions.py` and `pre/lib_ImportOSM.py` were removed because they were superseded
by `variants/app_2022/setup/lib/getPopulation.py`, `new_kochi/setActionsAndTransitions.py` and `new_kochi/createLinksAndNodes.py` (OSM edges to nodes/links DB). They remain available in git history.

## Repository conventions

* Simulation outputs (`state_*/`, `figures/`, `weights/w_*.csv`, `*.avi`) and large local GIS data are not tracked; see [`.gitignore`](./.gitignore).
* Input CSVs and images are **not** ignored: commit any `data/*.csv` a case needs to run.
* Run the regression test with `python -m unittest discover tests` (needs `numpy`, `matplotlib` and `opencv-python`; about 10 s). After an intentional change of behaviour, regenerate the recorded results with `UPDATE_GOLDEN=1 python tests/test_golden_ql.py`.
