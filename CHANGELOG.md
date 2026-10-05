# Changelog

Dates are those of the commits; the audit of each step is in [docs/audits](./docs/audits).

## 0.2.0

Results of 0.1.0 are **not** reproduced by the defaults of 0.2.0 (see *Changed*). `ModelOptions.legacy()` gives the 2021 behaviour
(bit for bit, the golden recordings), `ModelOptions.kochi2024()` the behaviour of the 2024 Kochi study.

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
* The temporal-difference methods discount with **0.999 per second** (the default), no longer 0.9 per decision. With 0.9 per
  decision one node costs about 1,200 s of walking in the target, and the best possible policy reaches 80.7 % of the shortest
  path on `kochi2`; with 0.999 per second Q-learning reaches 99 % ([step 4](./docs/audits/step4/README.md)).
* The shortest-path baseline (`updateTargetShortestPath`) marks arrivals (it never did; the survivors it printed were the agents
  that start on a shelter), moves agents with the same speed model as the learning agents, handles `-9999` (no path) and a
  `nextnode.csv` without a header; no recorded result depended on the old behaviour
  ([defect B](./docs/engine-reconciliation.md)). The evacuation times that the 2024 study committed for its shortest-path runs
  are not reproduced and must not be cited ([regenerated](./results/kochi2024_regenerated/README.md)).
* `new_kochi`: the node with 11 links lost its 11th link at one end only; the longest link there is removed at both ends.

### Fixed
* Unused variables and imports found by `ruff` (no change of behaviour: the golden recordings are unchanged).

## 0.1.0

The repository as it was reorganised into `src/evacrl`, `scripts`, `cases` ([migration guide](./docs/migration.md)).
