# Tsunami Evacuation using Reinforcement Learning

> This document explains the structure of the repository.
> For an explanation on the use of the code see the [Manual](./Manual.md).
> Notes on pre-processing and tsunami data are in [tegs.md](./tegs.md).

## Repository layout

| Path | What it is |
|------|------------|
| `main_ql.py`, `main_ql_mod.py`, `main_sarsa.py`, `main_mc.py`, `main_ShortPath.py` | Entry points. Each one defines `run_*` plus one helper per area (`kochi_*`, `arahama_*`, `new_kochi_*`); pick the case in the `__main__` block. `main_ql_mod.py` is a variant of `main_ql.py` that reloads the best-performing state matrix at the start of each block. `main_ShortPath.py` is the shortest-path baseline (no learning). |
| `qlearn.py`, `sarsa.py`, `mc.py` | The `QLearning`, `SARSA` and `MonteCarlo` classes. |
| `evac_plots.py` | Plot evacuation curves from each epoch. |
| `make_video.py` | Create a video (AVI) of a particular epoch (state matrix or policy). |
| `tdcontrol.py`, `tests_mc.py` | Toy TD-control skeleton and ad-hoc tests for `mc.py`. |
| `check_policies.ipynb`, `plot_survivors.ipynb` | Notebooks to compare policies and plot survivors per simulation. |
| `kochi/`, `new_kochi/`, `arahama/` | One folder per study area. Inputs live in `<area>/data/`; runs write to `<area>/state_<name>/` (not tracked, see `.gitignore`). `arahama/` is not part of this repository. |
| `new_kochi/` | Also contains the pre-processing pipeline for the Kochi case (`preProcess.py`, `createLinksAndNodes.py`, `getPopulation.py`, `setActionsAndTransitions.py`, `preprocess.ipynb`). |
| `app/` | Newer, self-contained version of the workflow: `main.py` (Q-learning, uses `bin/qlearn.py`), `setup/` (builds a case from an area-of-study GeoJSON in `input/`), `make_video.py`, `analysis.ipynb`. |
| `new_model/` | Work-in-progress object-oriented rewrite (`tsuevac` package: `Environment`, `Agent`, `Evacuee`, `Node`, `Shelter`, `Model`). Most methods are still stubs. |
| `pre/` | Original pre-processing scripts and census/population data (see below). |
| `system/` | GIS data (QGIS projects, rasters) and figures for the tsunami inundation / road network. |
| `database/` | Notebooks that analyse batches of runs (survivors vs. simulation time and mean departure time). |
| `other/` | Informal notebook for various calculations (e.g. weights from SARSA) and sample outputs. |
| `results/`, `weights/` | Sample outputs. |

Scripts are meant to be run from the repository root, because inputs and outputs
are addressed as `<area>/data/...` and `<area>/state_<name>/...`.

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
`setActionsAndTransitions.py` (`pre/SetActionsAndTransitions.py`, or the copy next to each case's `preProcess.py`).

## `pre/` directory

* `CensusAndBuildingDatabase`, `Household_database`, `Population_database` are folders with the census data (the household database is integrated but not in use at the moment).  
* `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py`, `SetPopDB.py` => population disaggregation (see [tegs.md](./tegs.md)).  
* `defPathsFromNodes.py` => a function to calculate the next node for a pre-determined shortest path run.  
* `DetectionShelters.py` => to detect evacuation points from the network.  
* `lib_ImportOSM.py` => to transform OSM data into suitable format.  
* `makeUniformPedestrianProfiles.py` => creates an `agentsdb`-style file with a fixed number of agents per (non-evacuation) node.  
* `getPopulation.py`, `SetActionsAndTransitions.py` => earlier versions of scripts that also exist in `new_kochi/` and `app/setup/lib/` (the copies differ; see `git log`).  

## Repository conventions

* Simulation outputs (`state_*/`, `figures/`, `weights/w_*.csv`, `*.avi`) and large local GIS data are not tracked; see [`.gitignore`](./.gitignore).
* Input CSVs and images are **not** ignored: commit any `data/*.csv` a case needs to run.
