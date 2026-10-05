# Where did it go? (reorganisation of the repository)

The repository was reorganised without changing what the code does: the golden tests in `tests/` compare
simulations before and after the reorganisation, byte for byte. This page maps the old locations
to the new ones, and says what to do with local files that git does not track.

## Old → new locations

| Old | New |
|-----|-----|
| `kochi/`, `new_kochi/` | `cases/kochi/`, `cases/new_kochi/` |
| `arahama/` (local only, never in the repository) | `cases/arahama/` |
| `qlearn.py`, `sarsa.py`, `mc.py`, `evac_plots.py`, `make_video.py` | `src/evacrl/` |
| `main_ql.py`, `main_ql_mod.py`, `main_sarsa.py`, `main_mc.py`, `main_ShortPath.py` | `scripts/` |
| `tdcontrol.py`, `tests_mc.py` | `experimental/` (removed in 0.2.0; in the git history) |
| `check_policies.ipynb`, `plot_survivors.ipynb` | `notebooks/` |
| `database/operation_arahama.ipynb`, `database/operation_kochi.ipynb`, `other/CalculateWeights.ipynb` | `notebooks/` |
| `other/evacuees_start_end_departure*.csv`, `other/kochi_old/` | `datasets/legacy/` |
| `app/` | `variants/app_2022/` |
| `new_model/` | `experimental/new_model/` (removed in 0.2.0, an unfinished rewrite; in the git history) |
| `new_model/man/*.drawio` | `docs/diagrams/` |
| `system/` | `datasets/gis/` |
| `pre/CensusAndBuildingDatabase/`, `pre/Household_database/`, `pre/Population_database/` | `datasets/census/` |
| `pre/nodesdb.csv` | `cases/kochi/data/nodesdb.csv` |
| `Manual.md`, `tegs.md` | `docs/manual.md`, `docs/preprocessing.md` |

Stayed where it was: the scripts of `pre/`, `results/`, `weights/`, `LICENSE`.

Removed (all remain in git history):

* `pre/getPopulation.py`, `pre/SetActionsAndTransitions.py`, `pre/lib_ImportOSM.py`: older versions of
  `variants/app_2022/setup/lib/getPopulation.py`, `cases/new_kochi/setActionsAndTransitions.py` and
  `cases/new_kochi/createLinksAndNodes.py`.
* `app/qlearn.py`: identical to `app/bin/qlearn.py`.
* `app/ql_arahama_sim_*.avi` (generated videos), `profile.stats`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`.

Code that moved between the modules (the imports `from evacrl.qlearn import QLearning`, `from evacrl.sarsa import SARSA` and
`from evacrl.mc import MonteCarlo` still work, and the simulations give byte-identical results):

* `qlearn.py`, `sarsa.py` and `mc.py` shared about 2,650 lines, 99% and 72% identical. The common code is now
  `src/evacrl/core.py` (`EvacuationModel`); the three classes are subclasses of it. `SARSA` holds the update rule (`tdControl`),
  `QLearning` is a subclass of `SARSA` with no code of its own, and `MonteCarlo` adds nothing. (Since then the update rule has moved to `evacrl/td.py` and `QLearning` has its own, off-policy, `bootstrapValue`; it is no longer a subclass of `SARSA`.)
* `MonteCarlo.updateVelocity(pedIndx, codeLink)` is gone: nothing called it. Every class now has `updateVelocityV1(pedIndx)` (also uncalled),
  which is the same computation but reads the link from the agent's row instead of taking it as an argument.
* `QLearning.loadShortestPathDB` no longer prints the size of the array it loads (a leftover debug line the other two classes never had).
* `simulationShortestPath`, `ArahamaMTRL_20191220_SeqSim`, `testAramaha2020August28`, `ArahamaMTRL_20191220_Video` and
  `SurvivedAgentsPerEvacuationNode` were module-level functions at the end of `mc.py` (with its `__main__` block). They are now only in
  `experimental/tests_mc.py`, which already had the same code for all of them except `testAramaha2020August28` (a debugging
  sketch, moved there as it was); the old `ArahamaMTRL_20191220_Video` also had an unused `t0 = time.time()`.

Find an old file with `git log --follow -- <new path>`, or read it as it was with
`git show 251bdb1:<old path>` (`251bdb1` is the last commit before the reorganisation).

## How to run things now

| Before | Now |
|--------|-----|
| `python main_ql_mod.py` (case chosen by editing the `__main__` block) | `python scripts/main_ql_mod.py kochi` (from any directory) |
| `from qlearn import QLearning` | `from evacrl.qlearn import QLearning` (`pip install -e .`, or add `src/` to `sys.path`) |
| `cd new_kochi && python preProcess.py` | `cd cases/new_kochi && python preProcess.py` |
| `cd pre && python SetPopDB.py` (needed a root-level `data/`) | `python pre/SetPopDB.py` (writes `cases/kochi/data/agentsdb.csv`) |
| Notebooks addressed `./kochi/...`, `../results/...` | Each notebook starts with a bootstrap cell that finds the repository; paths use `{CASES}` and `{RESULTS}` |

## Local files that git does not track

Git moves tracked files when you pull; files it ignores (simulation outputs, GIS data, local case folders)
stay at the old location. Copy them over, check, then delete the old folders:

```
git pull
for d in kochi new_kochi arahama; do [ -d "$d" ] && rsync -a "$d"/ "cases/$d"/; done   # state_*/ outputs, local GIS data, ...
[ -d app ]    && rsync -a app/    variants/app_2022/      # app/arahama/, app/setup/tmp, ...
[ -d system ] && rsync -a system/ datasets/gis/           # the large local rasters and shapefiles
ls kochi new_kochi arahama app system 2>/dev/null          # when you are happy with the copies: rm -r those folders
```

`figures/` and `weights/w_*.csv` are still written at the repository root; `database/`, `other/` and `new_model/` are
empty shells after the pull and can be deleted.
