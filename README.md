# Tsunami Evacuation using Reinforcement Learning

> This document explains the structure of the repository.
> For an explanation on the use of the code see the [Manual](./docs/manual.md).
> Notes on pre-processing and tsunami data are in [docs/preprocessing.md](./docs/preprocessing.md).
> Looking for a file you remember at its old location? See [docs/migration.md](./docs/migration.md).

## Repository layout

```
cases/               one folder per study area: inputs in data/, run outputs in state_<name>/ (not tracked)
  kochi/             old Kochi network (4,315 nodes)
  new_kochi/         Kochi network with 19,207 nodes, plus its pre-processing pipeline
src/evacrl/          importable code: core.py (shared engine), td.py (temporal-difference update), qlearn.py, sarsa.py, mc.py, options.py (model settings), tables.py (table reader), evac_plots.py, make_video.py, paths.py
scripts/             entry points: main_ql.py, main_ql_mod.py, main_sarsa.py, main_mc.py, main_ShortPath.py
notebooks/           analysis notebooks
tests/               golden regression tests
docs/                manual, pre-processing notes, migration guide (old -> new paths), diagrams
datasets/            large raw / shared inputs: census/, gis/ (QGIS data and its notebook), legacy/ (older samples)
pre/                 pre-processing scripts (population disaggregation, shelters, shortest paths)
variants/app_2022/   self-contained 2022 workflow (its own qlearn.py and setup pipeline)
experimental/        new_model/ (object-oriented rewrite), tdcontrol.py, tests_mc.py
results/, weights/   outputs shared by the notebooks
```

| Path | What it is |
|------|------------|
| `scripts/main_*.py` | Entry points. Each one defines `run_*` plus one helper per area (`kochi_*`, `arahama_*`, `new_kochi_*`); pick the case on the command line, e.g. `python scripts/main_ql_mod.py kochi` (the default of each script is unchanged; a case whose data is not in `cases/` gets a message listing the ones that are). `main_ql_mod.py` is a variant of `main_ql.py` that reloads the best-performing state matrix at the start of each block. `main_ShortPath.py` is the shortest-path baseline (no learning); it needs a `cases/<area>/data/nextnode.csv`, which is not in the repository (`pre/defPathsFromNodes.py` is the legacy script that produced such a file). |
| `src/evacrl/` | `core.py`: the simulation engine and state matrix, as the class `EvacuationModel`; `td.py` holds the temporal-difference update that `sarsa.py` (`SARSA`, on-policy) and `qlearn.py` (`QLearning`, off-policy: bootstraps from the best action) share and differ in by one method; `mc.py` holds `MonteCarlo`, which learns at the end of the simulation. The three are subclasses of the engine that differ only in how the action values are updated (see the [Manual](./docs/manual.md#the-three-methods-share-one-engine)); `evac_plots.py` (evacuation curves per epoch), `make_video.py` (AVI of a particular epoch), `cli.py` (case selection of the scripts) and `paths.py`. |
| `src/evacrl/options.py`, `src/evacrl/tables.py` | `ModelOptions`: the behaviours that differ between the 2021 code and the 2024 Kochi study (survival reward, how the density code of a link is computed, speed on entering a link, segment lookup, what the discount applies to), with the recommended settings as the default and the 2021 and 2024 behaviours as `ModelOptions.legacy()` and `ModelOptions.kochi2024()`; see [docs/engine-reconciliation.md](./docs/engine-reconciliation.md). `load_table`: reads the case tables whether or not they have a header and whether integers are written as `116` or `116.0`. |
| `src/evacrl/paths.py` | **The only place that knows the layout.** Case data, `figures/`, `weights/` and `results/` are all resolved through it, so moving a folder means editing this file. |
| `cases/new_kochi/` | Besides its data, holds the pre-processing pipeline of the case (`preProcess.py`, `createLinksAndNodes.py`, `getPopulation.py`, `setActionsAndTransitions.py`, `preprocess.ipynb`); run it from inside this folder (it uses `./data` and `./tmp`). |
| `notebooks/` | `check_policies`, `plot_survivors` (compare policies, survivors per simulation), `operation_*` (batches of runs: survivors vs. simulation and departure time), `CalculateWeights` (link weights from SARSA). Each starts with a bootstrap cell that finds the repository, so they run from any directory. |
| `tests/` | `test_golden.py`: regression tests that run short, seeded simulations of Q-learning (`run_ql`, `run_ql_mod`), SARSA and Monte Carlo on the Kochi networks and compare them with recorded results; the recordings are byte-identical to what the original code produced before the repository was reorganised. `test_data_provenance.py`: proves the derived inputs can be regenerated from their sources (actions/transitions from nodes+links, the Kochi population from the census, `new_kochi/data/linksdb.csv` from `tmp/linksdb0.csv`). `test_evac_plots.py` and `test_cli.py` cover `plotSurvivors` and the case selection of the scripts. Run them before and after any restructuring. |
| `variants/app_2022/` | Newer, self-contained version of the workflow: `main.py` (Q-learning, uses `bin/qlearn.py`), `setup/` (builds a case from an area-of-study GeoJSON in `input/`), `make_video.py`, `analysis.ipynb`. Run from inside the folder. |
| `experimental/new_model/` | Work-in-progress object-oriented rewrite (`tsuevac` package: `Environment`, `Agent`, `Evacuee`, `Node`, `Shelter`, `Model`). Most methods are still stubs. |
| `datasets/census/` | Census, household and building databases (`CensusAndBuildingDatabase`, `Household_database`, `Population_database`), the inputs of `pre/SetPopDB.py`. |
| `datasets/gis/` | GIS data (QGIS projects, rasters) and figures for the tsunami inundation / road network, plus the notebook that reads them (it uses `./data`, so it stays next to the data). |
| `datasets/legacy/` | Older samples: `kochi_old/` (state and results in the 31-column layout) and two evacuee start/end/departure tables. |
| `experimental/` | Also `tdcontrol.py` (toy TD-control skeleton) and `tests_mc.py` (ad-hoc runs of `MonteCarlo`: Arahama sequences, shortest-path run, video). |
| `results/`, `weights/` | Sample outputs of old Kochi runs; also where the notebooks and `computeWeightsAtLinks` read and write. |

## Quick start

```
pip install -r requirements.txt         # numpy, matplotlib, opencv-python, scipy, pandas + `evacrl` (editable)
python scripts/main_ql_mod.py kochi      # runs a case (kochi | new_kochi | arahama), from any directory
python -m unittest discover tests        # golden + data-provenance + unit tests (about 1 min)
```

More dependencies are optional groups declared in `pyproject.toml`: `pip install -e ".[notebooks]"` for the notebooks and
`pip install -e ".[preprocessing]"` for building a case from raw data (geopandas, osmnx, rasterio, ...); it pins `osmnx<2`, because the pipeline uses calls that osmnx 2.0 changed or removed, and that in turn selects an older numpy/pandas/geopandas stack. The `osgeo` module used by
`createLinksAndNodes.py` and `SetDatabaseBldMeshCodes.py` comes from GDAL, which is best installed with conda (`conda install gdal`).
The full `opencv-python` is needed (not `-headless`): `makeVideo` calls `cv2.destroyAllWindows()`.
The scripts, the tests and the notebooks also work without installing the package. Tested (all tests, strict mode) with Python 3.10, 3.11, 3.12 and 3.13, i.e. NumPy 1.26 to 2.5. On 3.11 in two environments: numpy 2.4 / pandas 3.0 / matplotlib 3.11 / opencv-python 5.0 (`requirements.txt`), and numpy 1.26 / pandas 2.2 / osmnx 1.9 (`.[preprocessing]`); the golden outputs are byte-identical in both.
Input status of the cases: both `cases/kochi/data` and `cases/new_kochi/data` are complete. `cases/kochi/data/agentsdb.csv` is generated by `python pre/SetPopDB.py` from `datasets/census/` and the case's `nodesdb.csv` (35,930 agents with the same start nodes as the recorded old-Kochi run in `results/`; the agent order of the original file is unknown). `arahama` is not part of this repository.

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

The census data they use is in `datasets/census/` (the household database is integrated but not in use at the moment).

* `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py` => population disaggregation (see [docs/preprocessing.md](./docs/preprocessing.md)).  
* `SetPopDB.py` => builds `cases/<area>/data/agentsdb.csv` (default `kochi`) from the census databases and the case's `nodesdb.csv`: each agent starts on the node closest to its building.  
* `defPathsFromNodes.py` => a function to calculate the next node for a pre-determined shortest path run.  
* `DetectionShelters.py` => to detect evacuation points from the network.  
* `makeUniformPedestrianProfiles.py` => creates an `agentsdb`-style file with a fixed number of agents per (non-evacuation) node.  
* `tests.ipynb` => scratch notebook (uses `variants/app_2022/setup/lib/getPopulation.py`).  

The older `pre/getPopulation.py`, `pre/SetActionsAndTransitions.py` and `pre/lib_ImportOSM.py` were removed because they were superseded
by `variants/app_2022/setup/lib/getPopulation.py`, `cases/new_kochi/setActionsAndTransitions.py` and `cases/new_kochi/createLinksAndNodes.py` (OSM edges to nodes/links DB). They remain available in git history.

## Repository conventions

* Simulation outputs (`cases/*/state_*/`, `figures/`, `weights/w_*.csv`, `*.avi`) and large local GIS data are not tracked; see [`.gitignore`](./.gitignore).
* Input CSVs and images are **not** ignored: commit any `data/*.csv` a case needs to run.
* Locations live in `src/evacrl/paths.py`; do not build `<area>/data/...` paths from the working directory. The repository is found from the checkout, or from the repository you run in; with a plain `pip install` outside it, set `EVACRL_ROOT` to its folder.
* Regression tests: `python -m unittest discover tests`. `GOLDEN_STRICT=1` additionally compares every output file byte for byte (same NumPy/Python only). After an intentional change of behaviour, regenerate the recorded results with `UPDATE_GOLDEN=1 python tests/test_golden.py [entry ...]`.
