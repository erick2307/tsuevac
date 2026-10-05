# evacrl: tsunami evacuation guidance with reinforcement learning

`evacrl` simulates the evacuation of a tsunami-threatened area on a road network, agent by agent, and compares two ways of
guiding the evacuees: **walking the shortest path** to a shelter, and a policy **learned with reinforcement learning**
(tabular Q-learning, SARSA or Monte Carlo) that chooses a street at each intersection, seeing how crowded the streets around are.
It also builds a study area from OpenStreetMap, shelter points and a census mesh, and runs the repeated, seeded experiments
that such a comparison needs.

**What the results say so far** (details and the numbers behind them in [docs/audits](./docs/audits)):

* On an area that is not crowded (`kochi2` of the 2024 Kochi study: 622 agents, 30 min; it is not shipped here, the audit reads it from
  the study's repository) Q-learning with the default settings reaches **99 %** of the shortest path's survivors (526 against 529): it
  learns the shortest path and, there, nothing better. Before the discount was corrected it stopped at 81 %; that ceiling was the
  discount (0.9 per decision), not the training.
* On the crowded `kochi_area4` (shipped; 13,502 agents) learned policies do **not yet** beat the shortest path. Trained on 30-minute
  episodes (60 simulations), the link-level code is 2 % behind it after 30 minutes (7,356 against 7,507 agents safe) but has only 74 % of
  the agents safe after 2 h, the shortest path all of them; the segment-level code is 13 % behind at 30 minutes. Trained on 2-hour episodes
  (30 simulations) they reach 89 % and 82 % at 2 h. A tabular policy is arbitrary where training did not go.
  **Whether learned guidance beats the shortest path is not shown by this repository.**
* The evacuation times committed with the 2024 study's shortest-path runs must not be cited; they are regenerated in
  [`results/kochi2024_regenerated`](./results/kochi2024_regenerated/README.md).

## Install

Python 3.10 to 3.13 (tested on all four in CI).

```
git clone https://github.com/erick2307/tsuevac
cd tsuevac
python -m venv .venv && . .venv/bin/activate
pip install -e .                    # numpy, scipy, matplotlib, opencv-python; ".[casebuild]" adds what downloading networks and the census need
python -m unittest discover tests   # about 2 minutes (the tests that need geopandas skip without it)
```

`pip install -r requirements.txt` adds pandas (the evacuation-curve plots). The history is large (a full clone downloads about 150 MB, mostly
files that are no longer in the repository); for a light clone use `git clone --depth 1 https://github.com/erick2307/tsuevac`. The package contains the code only: the
cases (`cases/`), the scripts (`scripts/`) and the audits live in the repository, so work from a clone.

## Quick start

```
python -m evacrl.casebuild validate cases/kochi_area2                                  # check the tables of a shipped case
python -m evacrl.experiment sp        kochi_area2 --runs 10 --time 30 --workers 2 --out runs/sp   # the baseline: 10 shortest-path runs
python -m evacrl.experiment calibrate kochi_area2 --method qlearning --sims 30 --eval-every 10 --eval-runs 3 --out runs/ql --sp runs/sp
python -m evacrl.experiment evaluate  kochi_area2 --state runs/ql/best_state.csv --runs 5 --workers 2 --out runs/ql_eval
python -m evacrl.experiment compare   --sp runs/sp --rl runs/ql_eval --out runs/compare.png   # the two evacuation curves
python -m evacrl.experiment policy    kochi_area2 --state runs/ql/best_state.csv --out runs/policy.png
```

`sp`, `calibrate` and `evaluate` write a `manifest.json` (checksums of the inputs, options, seeds, versions) next to their results; the same
`--seed` gives the same results for any `--workers`. The whole quick start takes about 6 minutes on 4 cores (most of it the training; it is a
demonstration of the commands, 30 simulations do not make a good policy) and writes into `runs/`. The 2021 entry points still work
(`python scripts/main_ql_mod.py kochi` starts a long training run). Building a case
from a road network, the model options and the three methods are in the [Manual](./docs/manual.md).

## Documentation

| | |
|---|---|
| [Manual](./docs/manual.md) | how to build a case, run experiments, choose model options; the three methods and the discount |
| [docs/repository.md](./docs/repository.md) | what is where, which parts are maintained, the two code stacks, the input tables |
| [cases/README.md](./cases/README.md) | the study areas: what was chosen and what it does to the numbers |
| [docs/engine-reconciliation.md](./docs/engine-reconciliation.md) | every difference between the 2021 engine and the 2024 study's, measured |
| [docs/audits](./docs/audits) | one report per completed development step, each regenerable |
| [docs/roadmap.md](./docs/roadmap.md), [CHANGELOG.md](./CHANGELOG.md) | the steps and what changed between versions |
| [docs/data-licences.md](./docs/data-licences.md) | data sources and terms, and what is not recorded |

## Licence, data and citation

The code is GPL-3.0 ([LICENSE](./LICENSE)). **The data keep the terms of their sources**: the networks derive from OpenStreetMap
(© OpenStreetMap contributors, ODbL), the shelters and census mesh of Kochi are from the Kochi Prefectural Office, and the source of
several older data folders is not recorded; see [docs/data-licences.md](./docs/data-licences.md). To cite the software, use
[CITATION.cff](./CITATION.cff) (GitHub: "Cite this repository").

## Development

`python -m unittest discover tests` (golden recordings, data provenance, unit tests; `GOLDEN_STRICT=1` compares every output byte for
byte, same NumPy and Python only; after an intentional change of behaviour regenerate with
`UPDATE_GOLDEN=1 python tests/test_golden.py [entry ...]`). `ruff check src tests scripts` for lint (`pip install -e ".[dev]"`).
Folders hold what they say: simulation outputs (`cases/*/state_*/`, `figures/`, `*.avi`) are not tracked, input CSVs are. Locations
live in `src/evacrl/paths.py`; set `EVACRL_ROOT` to the clone if you run the package from elsewhere.
