# Repository layout

What is where, which parts are maintained, and how the older parts relate to them. The front page is the [README](../README.md); how to
use the code is the [Manual](./manual.md).

## Status of each part

| Part | Status | Tested / linted |
|------|--------|-----------------|
| `src/evacrl/` (engine, `options`, `tables`, `casebuild`, `experiment`, plots) | **maintained**: the package | tests in `tests/`; `ruff` |
| `scripts/main_*.py` | maintained: the 2021 entry points, kept working (golden recordings) | golden tests; `ruff` |
| `cases/kochi_area0, 1, 2, 4/` | maintained: built and rebuilt by `evacrl.casebuild` | rebuilt from `raw/` by the tests |
| `cases/kochi/`, `cases/new_kochi/` | maintained as inputs of the 2021 runs; `new_kochi` also holds its 2021 pre-processing scripts, **superseded by `evacrl.casebuild`** | golden and provenance tests; scripts not linted |
| `tests/`, `docs/` | maintained | |
| `notebooks/` | analysis notebooks of the 2021 work; run by hand, not by the tests | not tested, not linted |
| `pre/` | **legacy**: scripts of the 2021 pre-processing (census disaggregation, shelters), still used by `cases/kochi/` | `SetPopDB` is covered by a provenance test; not linted |
| `variants/app_2022/` | **legacy**: the 2022 self-contained workflow with its own `qlearn.py`; not interchangeable with `src/evacrl` (below) | not tested, not linted |
| `datasets/` | data; **source and terms of several folders are not recorded**, see [data-licences.md](./data-licences.md) | |

## Layout

```
cases/               one folder per study area: inputs in data/, run outputs in state_<name>/ (not tracked)
  kochi/             old Kochi network (4,315 nodes)
  new_kochi/         Kochi network with 19,207 nodes, plus its pre-processing pipeline
  kochi_area0, 1, 2, 4/  four tsunami-evacuation areas of Kochi, built with `evacrl.casebuild` (see cases/README.md)
src/evacrl/          importable code: core.py (shared engine), td.py (temporal-difference update), qlearn.py, sarsa.py, mc.py, options.py (model settings), tables.py (table reader), casebuild/ (building and checking a case), experiment/ (repeated runs, calibration, comparison, manifests), evac_plots.py, make_video.py, paths.py
scripts/             entry points: main_ql.py, main_ql_mod.py, main_sarsa.py, main_mc.py, main_ShortPath.py
notebooks/           analysis notebooks
tests/               regression, provenance, unit and documentation tests
docs/                manual, pre-processing notes, migration guide (old -> new paths), diagrams
datasets/            large raw / shared inputs: census/, gis/ (QGIS data and its notebook), legacy/ (older samples)
pre/                 pre-processing scripts (population disaggregation, shelters, shortest paths)
variants/app_2022/   self-contained 2022 workflow (its own qlearn.py and setup pipeline)
results/, weights/   outputs shared by the notebooks
```

| Path | What it is |
|------|------------|
| `scripts/main_*.py` | Entry points. Each one defines `run_*` plus one helper per area (`kochi_*`, `arahama_*`, `new_kochi_*`); pick the case on the command line, e.g. `python scripts/main_ql_mod.py kochi` (the default of each script is unchanged; a case whose data is not in `cases/` gets a message listing the ones that are). `main_ql_mod.py` is a variant of `main_ql.py` that reloads the best-performing state matrix at the start of each block. `main_ShortPath.py` is the shortest-path baseline (no learning); it needs a `cases/<area>/data/nextnode.csv`, which the `kochi_area*` cases carry (`evacrl.casebuild` writes it; `pre/defPathsFromNodes.py` is the legacy script) and `kochi` and `new_kochi` do not. `python -m evacrl.experiment sp` is the maintained way to run the baseline. |
| `src/evacrl/` | `core.py`: the simulation engine and state matrix, as the class `EvacuationModel`; `td.py` holds the temporal-difference update that `sarsa.py` (`SARSA`, on-policy) and `qlearn.py` (`QLearning`, off-policy: bootstraps from the best action) share and differ in by one method; `mc.py` holds `MonteCarlo`, which learns at the end of the simulation. The three are subclasses of the engine that differ only in how the action values are updated (see the [Manual](./manual.md#the-three-methods-share-one-engine)); `evac_plots.py` (evacuation curves per epoch), `make_video.py` (AVI of a particular epoch), `cli.py` (case selection of the scripts) and `paths.py`. |
| `src/evacrl/options.py`, `src/evacrl/tables.py` | `ModelOptions`: the behaviours that differ between the 2021 code and the 2024 Kochi study (survival reward, how the density code of a link is computed, speed on entering a link, segment lookup, what the discount applies to), with the recommended settings as the default and the 2021 and 2024 behaviours as `ModelOptions.legacy()` and `ModelOptions.kochi2024()`; see [docs/engine-reconciliation.md](./engine-reconciliation.md). `load_table`: reads the case tables whether or not they have a header and whether integers are written as `116` or `116.0`. |
| `src/evacrl/casebuild/` | Building a case from a road network: `python -m evacrl.casebuild` with `from-osm`, `from-snapshot`, `from-raw` or `validate`. Short-link clean-up, actions and transitions, shortest-path table, agents (uniform, per node or by census), shelters attached to the network (or snapped to nodes), validation. Needs only NumPy and SciPy, except downloading and the census (`pip install -e ".[casebuild]"`). See the [Manual](./manual.md#building-a-case) and [cases/README.md](../cases/README.md). |
| `src/evacrl/experiment/` | Experiments on a case: `python -m evacrl.experiment` with `sp` (repeated shortest-path runs, stopped by a convergence rule), `calibrate` (training with checkpoints evaluated greedily, best policy kept), `evaluate` (greedy runs of a stored policy) and `compare` (evacuation curves against the shortest path). Seeds derived from one base seed, the same results for any number of worker processes, a `manifest.json` with every result. Needs only NumPy and Matplotlib. See the [Manual](./manual.md#experiments). |
| `src/evacrl/paths.py` | **The only place that knows the layout.** Case data, `figures/`, `weights/` and `results/` are all resolved through it, so moving a folder means editing this file. |
| `cases/new_kochi/` | Besides its data, holds the pre-processing pipeline of the case (`preProcess.py`, `createLinksAndNodes.py`, `getPopulation.py`, `setActionsAndTransitions.py`, `fixExcessLinks.py`, `preprocess.ipynb`); run it from inside this folder (it uses `./data` and `./tmp`). |
| `notebooks/` | `check_policies`, `plot_survivors` (compare policies, survivors per simulation), `operation_*` (batches of runs: survivors vs. simulation and departure time), `CalculateWeights` (link weights from SARSA). Each starts with a bootstrap cell that finds the repository, so they run from any directory. |
| `tests/` | `test_golden.py`: short, seeded simulations of Q-learning (`run_ql`, `run_ql_mod`), SARSA and Monte Carlo on the Kochi networks compared with recorded results (the `*_legacy` entries of SARSA and Monte Carlo are byte-identical to what the original code produced before the reorganisation; the others were regenerated when a default changed, see the [CHANGELOG](../CHANGELOG.md)). `test_data_provenance.py`: the derived inputs can be regenerated from their sources. `test_td.py`, `test_engine_options.py`, `test_tables.py`: the update rules (hand-worked values), the model options, the table reader. `test_casebuild*.py`: the case builder, the shipped cases rebuilt from `raw/`, the geospatial part (skipped without the `casebuild` extra). `test_experiment.py`: the experiment layer. `test_cli.py`, `test_evac_plots.py`, `test_paths.py`, `test_docs.py`: case selection, plots, where the repository is, links and commands of the documentation. Run them before and after any change. |
| `variants/app_2022/` | Newer, self-contained version of the workflow: `main.py` (Q-learning, uses `bin/qlearn.py`), `setup/` (builds a case from an area-of-study GeoJSON in `input/`), `make_video.py`, `analysis.ipynb`. Run from inside the folder. |
| `datasets/census/` | Census, household and building databases (`CensusAndBuildingDatabase`, `Household_database`, `Population_database`), the inputs of `pre/SetPopDB.py`. |
| `datasets/gis/` | GIS data (QGIS projects, rasters) and figures for the tsunami inundation / road network, plus the notebook that reads them (it uses `./data`, so it stays next to the data). |
| `datasets/legacy/` | Older samples: `kochi_old/` (state and results in the 31-column layout) and two evacuee start/end/departure tables. |
| `results/`, `weights/` | Sample outputs of old Kochi runs; also where the notebooks and `computeWeightsAtLinks` read and write. |

## Installing and the optional groups

See the [README](../README.md#install). More dependencies are optional groups declared in `pyproject.toml`: `pip install -e ".[notebooks]"` for the notebooks and
`pip install -e ".[preprocessing]"` for the 2021 pre-processing scripts (geopandas, osmnx, rasterio, ...); it pins `osmnx<2`, because those scripts use calls that osmnx 2.0 changed or removed. The `osgeo` module used by
`createLinksAndNodes.py` and `SetDatabaseBldMeshCodes.py` comes from GDAL, which is best installed with conda (`conda install gdal`).
The full `opencv-python` is needed (not `-headless`): `makeVideo` calls `cv2.destroyAllWindows()`.
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

* `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py` => population disaggregation (see [docs/preprocessing.md](./preprocessing.md)).  
* `SetPopDB.py` => builds `cases/<area>/data/agentsdb.csv` (default `kochi`) from the census databases and the case's `nodesdb.csv`: each agent starts on the node closest to its building.  
* `defPathsFromNodes.py` => a function to calculate the next node for a pre-determined shortest path run.  
* `DetectionShelters.py` => to detect evacuation points from the network.  
* `makeUniformPedestrianProfiles.py` => creates an `agentsdb`-style file with a fixed number of agents per (non-evacuation) node.  
* `tests.ipynb` => scratch notebook (uses `variants/app_2022/setup/lib/getPopulation.py`).  

The older `pre/getPopulation.py`, `pre/SetActionsAndTransitions.py` and `pre/lib_ImportOSM.py` were removed because they were superseded
by `variants/app_2022/setup/lib/getPopulation.py`, `cases/new_kochi/setActionsAndTransitions.py` and `cases/new_kochi/createLinksAndNodes.py` (OSM edges to nodes/links DB). They remain available in git history.

## Repository conventions

* Simulation outputs (`cases/*/state_*/`, `figures/`, `weights/w_*.csv`, `*.avi`) are not tracked; see [`.gitignore`](../.gitignore). The GIS data under `datasets/gis/data/qgis` and `qgis_1` (277 MB, three files over 65 MB) **are** tracked, which makes a clone large.
* Input CSVs and images are **not** ignored: commit any `data/*.csv` a case needs to run.
* Locations live in `src/evacrl/paths.py`; do not build `<area>/data/...` paths from the working directory. The repository is found from the checkout, or from the repository you run in; with a plain `pip install` outside it, set `EVACRL_ROOT` to its folder.
* Regression tests: `python -m unittest discover tests`. `GOLDEN_STRICT=1` additionally compares every output file byte for byte (same NumPy/Python only). After an intentional change of behaviour, regenerate the recorded results with `UPDATE_GOLDEN=1 python tests/test_golden.py [entry ...]`.
