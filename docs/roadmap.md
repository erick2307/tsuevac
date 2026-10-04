# Roadmap: making `evacrl` the reference model

Goal: bring the features of the 2024–25 Kochi study (`erick2307/2024_urushibara`, `EVACMODEL3_FocalPoints`) into
this repository, fix what the comparison exposes, and leave a scientifically defensible, tested, documented
package that can be shared.

Working rules

* Every step ends with an **audit** (tests, a measured comparison, a written finding) and **stops for confirmation**.
  Nothing from the next step is started before that.
* `python -m unittest discover tests` must pass after every step. Where a step *intentionally* changes results,
  the golden recordings are regenerated in that step and the change is reported, never hidden.
* Behaviour that differs between the two code bases is introduced as an explicit option first; defaults change
  only after the audit shows the effect and the decision is confirmed. One exception, reported where it happens:
  behaviour that is plainly defective and that no recorded result depends on (the shortest-path baseline, which had no
  recorded results and never counted arrivals) is fixed directly.
* The Urushibara repository is a read-only reference.

| # | Step | Audit | Status |
|---|------|-------|--------|
| 0 | Baseline: branch `regid/trusting-curie-ogh40k` = `dev`; 25 tests pass | tests | done |
| 1 | **Engine reconciliation + shortest-path baseline fixes**: every divergence between `core.py` and `SARSA2024.py` measured and decided; tolerant table loader; options for the behaviours that differ; the far-end freeze defect found and fixed behind an option | differential run of both engines on the same inputs (bit-exact agreement; paired shortest-path runs; learning ablation); 54 tests pass, golden recordings unchanged. Report: [engine-reconciliation.md](./engine-reconciliation.md) | confirmed (D1–D5), applied in Step 2 |
| 2 | **Real off-policy Q-learning** (target = max over the valid actions at S, terminal handling), distinct from SARSA; the defaults of Step 1 flipped (`segmentIndex="clamped"`, `entrySpeed="position"`) with the golden recordings regenerated once; `discounting` option | hand-computed unit tests (14, each mutation-checked); legacy-pinned golden entries for SARSA and Monte Carlo stay byte-identical; SARSA vs Q-learning and Monte Carlo discounting measured. Report: [audits/step2](./audits/step2/README.md) | **awaiting your confirmation** |
| 3 | **Case-building pipeline** (`evacrl.casebuild`): evacuation-area polygon → OSM network → shelters/evacuation buildings snapped to nodes → census-mesh population (1/5/10 agents per node) → actions/transitions → shortest-path table; Kochi areas 0–4 as shipped cases | rebuild an area offline from stored graph files and compare with the committed Urushibara inputs; data-provenance tests | |
| 4 | **Experiment layer**: calibration (decaying ε, reload best policy), repeated shortest-path runs with convergence (CV) check, RL-vs-SP comparison plots, parallel runs, reproducible seeds and run manifests, CLI | reproduce the committed kochi2 shortest-path distribution; seeded reruns are identical | |
| 5 | **Release engineering**: README/manual, citation, licence and data attribution, CI on Python 3.10–3.13, lint, repository hygiene, retire or label stubs | clean-checkout install and quick start; docs build of every command | |
| 6 | **Final audit**: full suite on all Python versions, end-to-end run of Kochi 0–4, independent code review, list of open items | review findings fixed or listed | |
