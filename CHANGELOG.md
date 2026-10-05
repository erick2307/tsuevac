# Changelog

The audit of each step is in [docs/audits](./docs/audits).

## 0.2.0

Results of 0.1.0 are **not** reproduced by the defaults of 0.2.0 (see *Changed*). `ModelOptions.legacy()` gives the 2021 behaviour of
SARSA and Monte Carlo (pinned bit for bit by golden recordings), `ModelOptions.kochi2024()` the behaviour of the 2024 Kochi study
(SARSA, bit for bit against its code). Q-learning has no 2021 form: in 0.1.0 it was SARSA under another name.

### Added
* `evacrl.options.ModelOptions`: the behaviours in which the 2021 code and the 2024 Kochi study differ (survival reward, density
  code of a link, speed on entering a link, segment lookup, what the discount applies to, the discount) as explicit options
  ([engine reconciliation](./docs/engine-reconciliation.md)).
* `evacrl.tables.load_table`: reads the case tables with or without a header and with integers written as `116` or `116.0`.
* Real off-policy **Q-learning** (the target is the best action at the next state, terminal states handled), distinct from SARSA
  ([step 2](./docs/audits/step2/README.md)).
* `evacrl.casebuild` (`python -m evacrl.casebuild`, `evacrl-casebuild`): a case from a road network: stored or downloaded OSM
  graph, shelters attached by an access link (or snapped), area-weighted census population, short-link clean-up by clusters,
  actions and transitions, shortest-path table, validation ([step 3](./docs/audits/step3/README.md)).
* Cases `kochi_area0`, `kochi_area1`, `kochi_area2`, `kochi_area4`, with `raw/`, `node_population.csv`, `provenance.json` and the
  `nextnode.csv` that the shortest-path baseline needs.
* `evacrl.experiment` (`python -m evacrl.experiment`, `evacrl-experiment`): repeated shortest-path runs with a convergence rule,
  training with checkpoints evaluated greedily and frozen, runs of a stored policy, policy maps, RL-vs-shortest-path plots,
  process-pool parallelism with results independent of the number of workers, seeds derived from one base seed, a `manifest.json`
  with every result ([step 4](./docs/audits/step4/README.md)).
* Tests: from 25 to the full suite of `python -m unittest discover tests`; CI on Python 3.10 to 3.13; `ruff` lint.

### Changed (results differ from 0.1.0)
* `segmentIndex="clamped"` (a defect that froze the evacuees that had walked to the far end of a link is removed) and
  `entrySpeed="position"` are the defaults.
* `QLearning` is real off-policy Q-learning (the target is the best action at the next state); in 0.1.0 it was SARSA. The recorded
  results of `run_ql` and `run_ql_mod` changed accordingly.
* All three methods discount with **0.999 per second** (the default). Before, SARSA and Q-learning discounted by 0.9 once per decision and
  Monte Carlo by 0.9 per second (`legacy()` and `kochi2024()` keep that). With 0.9 per
  decision one node costs about 1,200 s of walking in the target, and the best possible policy reaches 80.7 % of the shortest
  path on `kochi2`; with 0.999 per second Q-learning reaches 99 % ([step 4](./docs/audits/step4/README.md)).
* The shortest-path baseline (`updateTargetShortestPath`) marks arrivals (it never did; the survivors it printed were the agents
  that start on a shelter), moves agents with the same speed model as the learning agents, handles `-9999` (no path) and a
  `nextnode.csv` without a header; no recorded result depended on the old behaviour
  ([defect B](./docs/engine-reconciliation.md)). The evacuation times that the 2024 study committed for its shortest-path runs
  are not reproduced and must not be cited ([regenerated](./results/kochi2024_regenerated/README.md)).
* `new_kochi`: the node with 11 links lost its 11th link at one end only; the longest link there is removed at both ends.

### Removed
* `experimental/` (`new_model/`, an unfinished object-oriented rewrite; `tdcontrol.py`; `tests_mc.py`). They remain in the git history.

### Fixed
* The shortest-path table (`nextnode.csv`, `evacrl.casebuild.routing.next_nodes`) left the choice between equally short walks to
  SciPy, which breaks ties differently in different versions: `kochi_area4` rebuilt on Python 3.10 (SciPy 1.15) differed from the
  shipped table (SciPy 1.18) in one entry. The next node is now the lowest-numbered neighbour on a shortest walk, whatever the order
  of the links and the SciPy version. The shipped tables change in 3 (`kochi_area0`), 3 (`kochi_area1`), 0 (`kochi_area2`) and 13
  (`kochi_area4`) entries, every one between two equally short steps, so no walk gets longer
  ([`docs/audits/step5`](./docs/audits/step5/README.md)); the shortest-path baseline of those cases varies slightly with it. `method="allpairs"` (the
  2024 study's way) keeps SciPy's tie-breaking, to reproduce its tables.
* SciPy is a core dependency: `evacrl.casebuild` (and so `python -m evacrl.casebuild validate`) failed to import after a plain
  `pip install evacrl`.
* Unused variables and imports found by `ruff` (no change of behaviour: the golden recordings are unchanged).

## 0.1.0

The repository as it was reorganised into `src/evacrl`, `scripts`, `cases` ([migration guide](./docs/migration.md)).
