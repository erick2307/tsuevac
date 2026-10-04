# Engine reconciliation: `evacrl` (2021) and `SARSA2024.py` (Kochi study, 2024–25)

Two code bases carry the same simulation engine. `src/evacrl/core.py` is the 2021 line; `EVACMODEL3_FocalPoints/SARSA2024.py`
in [erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara) (commit `b66f7dd`) is a fork that was changed for
the Kochi study. This document lists every behavioural difference, says what each does to the results, records two defects that
the comparison exposed, and recommends settings. Everything here can be regenerated with the scripts in
[`docs/audits/step1/`](./audits/step1/README.md).

> **Status after Step 2.** The decisions D1–D5 below were confirmed and applied: `ModelOptions()` now has
> `segmentIndex="clamped"` and `entrySpeed="position"`, and `ModelOptions.legacy()` is the 2021 behaviour. Where this report says
> "2021 settings" it means `legacy()`; the numbers were measured with the options spelled out, so they do not depend on the defaults.

## How it was compared

* **Case:** `kochi2` of the Urushibara repository (309 nodes, 622 agents, 4 shelters). It is the one area that ships with a
  shortest-path table (`nextnode.csv`) and with 1,000 committed shortest-path runs to cross-check.
* **Paired runs:** a seed fixes the departure times, so the same seed gives paired runs across engines and settings. Shortest
  path: 30 seeds, 120 min simulated, mean departure 5 min. Learning: SARSA, 50 training simulations, 3 seeds, then 5 greedy
  evaluation runs, 30 min simulated. The tests on learning compare per-seed means (3 vs 3): the 5 evaluations of a seed share one
  policy. (The first version of this report tested all 15 evaluations and so printed smaller p-values; corrected in Step 2.)
* **Exact agreement:** `ModelOptions.kochi2024()` makes `evacrl` reproduce `SARSA2024.py` **bit for bit**: identical evacuation
  curves on 30 of 30 shortest-path seeds, and identical state matrices after 4 SARSA training simulations on 2 seeds. So the
  differences below are the *complete* list for those code paths.

## Differences between the two engines

Measured with the 30 paired shortest-path seeds (S) and the learning ablation (L). "Last evacuee" is the time at which the last
agent reached a shelter.

| # | Setting (`ModelOptions`) | 2021 | 2024 | What was measured | Recommendation |
|---|---|---|---|---|---|
| 1 | `surviveReward` | 1e5 | 1e7 | L: 429.4 vs 434.8 survivors at 30 min (p = 0.30, 3 seeds); training curves equal | keep 1e5; settle in Step 4 with a long calibration |
| 2 | `densityLevel` | whole link, 2 m width, at this instant | worst 2 m segment, real width, refreshed every 10 s | L: 431.9 vs 434.8 (p = 0.63). **The 2021 code has 309–310 states on this case, one per node: whole-link density almost never exceeds 0.5 persons/m², so the state carries no information on crowding. The 2024 density code gives 419–447 states.** | keep the 2021 value for now; decide after a crowded test in Step 4 |
| 3 | `entrySpeed` | first segment, from either end | segment where the agent is | S (both fixed): last evacuee +43 s (2,645 vs 2,602 s, p = 0.004), survivors at 30 min equal (p = 0.43). L: 436.3 vs 434.8 (p = 0.27) | `"position"` (it is the physically consistent one; small effect) |
| 4 | `segmentSizing` | `ceil(L/2)` | `round(L/2)`, at least 1 | differs on 24–26 % of the links of the four networks checked; S: survivors at 30 min equal in 29 of 30 seeds, last evacuee −61 s (p = 0.50) | keep `"ceil"`; not distinguishable from noise |
| 5 | `checkTarget` skips agents that have not started | no (crashes if one sits on its placeholder target) | yes | no effect on any existing result; ported without an option | ported |
| 6 | table reading | `#` headers, integers | one header line (any), integers written as `116.0` | `evacrl.tables.load_table` reads both; the Urushibara tables can now be used as they are | ported |
| 7 | shortest-path code | see defect B | | | ported (fixed) |

Also in `SARSA2024.py` and **not** ported yet (they are plots, not dynamics): `plotBestChoices`, node labels in `plotNetwork`,
`getSnapshotV2(ifNodeAround=...)` (Step 4); its `calibration.py` and `shortestpath.py` (Step 4); its `preprocess.py` (Step 3).

## Defects found

### A. Agents freeze at the far end of a link (both engines)

`updateVelocityV2` finds the segment of an agent as `floor(distance to the link's first node / segment length)`. At the far
end of a link whose straight-line length is at least its stored length that is the number of segments, one *past* the last
segment. The speed array holds 0 there until the link has once been empty at a refresh (then it holds 1.19, the free speed).
With 0 the agent stops, and since it keeps the link occupied the value never changes: every later agent that enters the link
from that end stops too, until noise (±0.01 m/s) happens to move one of them. This was traced on `kochi2`, seed 1: ten agents
were not evacuated after 2 h; eight sat on link 334 for the whole run, two escaped after about 5,900 s.

| Shortest path, 30 seeds | runs with agents left after 2 h | last evacuee, mean ± sd (s) | median | p95 | survivors at 30 min |
|---|---|---|---|---|---|
| 2021 settings | 13 / 30 | 3,601 ± 1,400 | 2,720 | 6,187 | 526.4 |
| 2024 settings | 19 / 30 | 4,259 ± 1,811 | 3,674 | 6,965 | 515.6 |
| 2021 settings, `segmentIndex="clamped"` | **0 / 30** | **2,602 ± 109** | 2,586 | 2,836 | 528.6 |
| 2024 settings, `segmentIndex="clamped"` | **0 / 30** | **2,645 ± 104** | 2,626 | 2,834 | 529.2 |

The fix (limit the index to the last segment, `segmentIndex="clamped"`) changes the last evacuee by −999 s (2021 settings,
p = 0.0005) and −1,614 s (2024 settings, p = 0.00003), and removes the long tail. The survivors at 30 min move by +2 (2021
settings) and +14 (2024 settings) of 622, so results that only look at a fixed early time are little affected; results about
*time to evacuate everyone*, or the number of agents left, are strongly affected. The learning ablation (L) shows the fix
does not hurt learning (434.8 vs 432.6 survivors at 30 min; 3 seeds, p = 0.06, if anything better with the fix).

**This also affects the committed Urushibara results.** In their 1,000 shortest-path runs per area, 43 % (kochi2), 53 % (kochi1)
and 63 % (kochi0) of the runs ended with agents who had not evacuated after 2 h, and the reported evacuation time has a tail up to
the end of the simulation (kochi2: median 2,692 s, p95 6,976 s). That is the signature measured above. Those numbers should be
regenerated with `segmentIndex="clamped"` before they are cited.

### B. The 2021 shortest-path baseline (`updateTargetShortestPath`)

* It never marked an agent as evacuated, but `scripts/main_ShortPath.py` counts survivors from that mark. On `kochi2` it printed
  **14** in every one of 30 runs (the agents who *start* on a shelter) while 613–622 of 622 actually arrived.
* It gave agents a speed from a different model (`updateSpeed`, whole-link density plus noise) than the learning agents use, so
  baseline and learner were not compared under the same dynamics.
* `-9999` (no path) in `nextnode.csv` raised `IndexError`; `loadShortestPathDB` always skipped the first line, which loses a row
  of a table without a header (the Urushibara format).

All four are fixed (`core.py`, `updateTargetShortestPath`; an agent without a path now stops where it is, whereas in `SARSA2024.py` it
keeps walking in its last direction). No recorded result depended on the old behaviour: the repository has no `nextnode.csv`, and the golden
tests have no shortest-path entry. This is the one place where a default changed without an option; it is the decision D4 below.

## Questions the comparison raises, not changed here

1. **The first step of every agent is random**, in every method and in the shortest-path baseline (`initEvacuationAtTime`). The
   "shortest path" baseline is therefore the shortest path from the second node on. Learning never optimises that first choice.
2. **Discounting differs between methods** (read from the code, not measured): the SARSA update discounts once per *decision*
   (`discount * Q(S, A)`), the Monte Carlo update once per *second* (`G *= discount ** dt`). With `discount = 0.9` the latter
   weights a shelter 100 s away by 0.9^100 ≈ 3·10⁻⁵. The methods are not comparable until this is settled. Step 2 made it an
   option (`ModelOptions.discounting`) and measured it: [audits/step2](./audits/step2/README.md).
3. **The calibration in `EVACMODEL3_FocalPoints/calibration.py` ends at 50 % exploration** (`1 / (s / N + 1)`), and the survivors it
   reports are those of the exploring runs. A greedy evaluation is the right measure of the learned policy (Step 4).
4. **`QLearning` was `SARSA`** in `src/evacrl/qlearn.py`. Resolved in Step 2: it now has its own off-policy update
   (see [manual](./manual.md#the-three-methods-share-one-engine)).
5. After 50 training simulations the greedy policy reaches 435 survivors at 30 min on `kochi2`, against 529 for shortest path, and
   the training curve was still rising. This audit says nothing about the policy quality after the 1,000–15,000 simulations of the
   Kochi study.

## Known and deferred

* `updateTarget` and `updateTargetShortestPath` duplicate the arrival and step logic (the former was already duplicated in 2021).
  The scripts still count survivors with `np.sum(pedDB[:, 10] == 1)` instead of `getNumberEvacuatedPed()`. Both are cleaned up
  with the experiment layer (Step 4).
* `speArrPerLink` is all zeros until the first 10 s refresh, so an agent that departs before it starts with speed ≈ 0 until then
  (same in 2021 and 2024).
* The self-loop of an evacuation node has link number −1. `getStateIndexAtNode` passes it to `computeDensityLevel`, and Python reads
  −1 as the *last* link of the network, so the state of a shelter carries the crowding of an unrelated link (read from the code; the
  consequence is a few extra states at shelters, whose values are terminal anyway). Left as it is: changing it moves state codes.
* `linksdb` is read as whole meters (the documented format; the Urushibara pipeline writes `int64` lengths and removes
  zero-length links). A link of length 0 is now reported by number instead of failing later with `nan`.

## Decisions for you

| | Decision | Recommendation | Outcome |
|---|---|---|---|
| D1 | `segmentIndex="clamped"` as the default | yes: it removes a defect, the evidence is above | confirmed, applied in Step 2 |
| D2 | `entrySpeed="position"` as the default | yes, with D1 (without the clamp it makes defect A more frequent: 19/30 vs 13/30 runs) | confirmed, applied in Step 2 |
| D3 | `surviveReward`, `densityLevel` stay at the 2021 values until Step 4 | yes: the evidence at 622 agents and 50 simulations cannot separate them | confirmed |
| D4 | the shortest-path fixes (B) stay as applied, without an option | yes | confirmed |
| D5 | when to flip the defaults of D1–D2 | at the start of Step 2, when the golden recordings are regenerated anyway, so that they change once | confirmed, done |
| D6 | the Kochi results of the 2024 repository are regenerated with the fix before they are cited | yes (outside this repository) | open, not part of this repository |
