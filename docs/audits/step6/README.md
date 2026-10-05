# Step 6 audit: the final audit

The last step: the decisions D11-D18 applied, the whole package reviewed once more by a reader who had not seen it, the shipped areas run end to end, and the list of what
is still open. Scripts: `end_to_end.sh` (the end-to-end run), `study_metric.py` (D11), `shrink_history.sh` (D15), `clean_checkout.sh` of [Step 5](../step5/README.md).

## The final code review

An independent reviewer read `src/evacrl` and `scripts/` at the release candidate, ran the suite, wrote small reproducing scripts, fuzzed 120 random networks through all
three methods and the shortest-path baseline, and compared the documentation with the code. **14 confirmed defects and 5 suspected**; none changes a recorded result (the golden recordings are
unchanged by every fix below). What it checked and found correct: the time and curve bookkeeping against the engine clock (`curve`, `safe_at`, `last_evacuee`), results
identical for 1 and 3 workers in `sp` and `calibrate` (all three methods, with `restart_from_best`, frozen and learning evaluation), `populationAtLinks` always equal to the agents on each link, the 10-action layout of the state
matrix, the Q-learning maximum ignoring padding, the CSV round trip of a policy, `next_nodes` against brute force on 400 random networks, `merge_short_links` against union-find on
1,500, `apportion` on 200,000 trials, and the README quick start.

| # | Finding | Outcome |
|---|---|---|
| 1 | `scripts/main_ShortPath.py` stopped after its first simulation (the shortest-path table was loaded for the first model only), and could not be run on any shipped case | **fixed**, with a test; every `main_*.py` now takes any folder of `cases/` |
| 2 | `scripts/main_mc.py`'s default resumed simulation 1950 of a run that is not in the repository | **fixed** (a fresh run), with a test |
| 3 | `python -m evacrl.experiment`: a result folder that cannot be made or a missing `--sp` was found out after the whole computation | **fixed**: checked before; tests |
| 4 | `evaluate` and `policy` accepted a state matrix of any larger case (and `policy` raised an `IndexError` on a short one) | **fixed**: states of nodes the case lacks and values in slots of actions a node does not have are refused; checked by hand against every stored state of the Step 4 audit (accepted on its own case, refused on another) |
| 5 | The state of an evacuation node read the *last link of the table* (its only action is the "link" -1) | **fixed** (density code 0); no recorded result changes, and on the shipped cases it had no measured effect |
| 6 | `--discount 1.5` or `0` reached the model (only `ModelOptions` checked) | **fixed** in the engine |
| 7 | The engine imported OpenCV at import, so every command needed its system libraries; the manual says only NumPy and Matplotlib | **fixed**: imported in `makeVideo` only; test in a subprocess with OpenCV unavailable |
| 8 | The manifest recorded the commit of whatever git repository the data folder is in | **fixed**: the commit of the checkout the code is in, else none |
| 9 | The reference curve of `plotSurvivors` used the mean departure time as the Rayleigh scale (the engine uses the mean times sqrt(2/pi)): 0.39 instead of 0.54 of the agents gone at the mean | **fixed**, tested against the engine's own draw |
| 10 | Agents that depart before the first multiple of 10 s move at about 0.01 m/s until it (`speArrPerLink` is first filled at `t % 10 == 0`) | **open** (below) |
| 11 | Documentation: (a) agents that start at a shelter are not counted "when they depart" but from the first second; (b) `main_mc.py` does not explore with the decaying rate; (c) `survivorsPerSim[-1] == case.pedDB.shape[0]` compares a list with an integer, so the early stop never fires and a resumed run overwrites its survivors file | (a), (b) **fixed**; (c) **open** (below) |
| 12 | Engine inputs not validated: an empty agents file, a node with 11 links and an agent on an isolated node failed with cryptic errors; `casebuild --agents 0` or `-5` wrote nothing useful | **fixed**: clear messages, in the engine, the table reader and the case builder; tests |
| 13 | `segmentIndex="raw"` (the 2021 and 2024 presets) raises an `IndexError` instead of freezing when a link's stored length is much shorter than the distance between its nodes | **open** (below) |
| 14 | A run seeds NumPy's global generator and left it seeded | **fixed**: restored after the run; test |
| S1-S5 | S1 the shortest-path agent takes the first of parallel links while the route used the shortest; S2 key overflow on Windows with NumPy < 2; S3 `calibrate --sp` does not check that the reference used the same time and options; S4 `workers > 1` from Python on spawn platforms needs a `__main__` guard; S5 two unused, defective resize methods | S2 **fixed** (64-bit keys; not testable on Linux), S4 **documented**; S1, S3, S5 **open** (below) |

Every fix has a test that fails without it; the new behaviours were also mutation-checked (the engine input checks and the shortest-path script: 8 mutants, 1 equivalent, 1 survivor closed with a test; the
experiment-layer fixes: 3 of 3 killed; the tie-break of Step 5: 5 mutants).

## End to end: Kochi 0, 1, 2, 4 and the two 2021 cases

`end_to_end.sh`, on the final code, four areas in parallel (4 cores): rebuild the case from its `raw/` folder and compare the six tables with the shipped ones, `validate`, `sp`
(10 runs of 30 min), `calibrate` (10 simulations, a checkpoint every 5, 2 evaluation runs), `evaluate` (3 frozen runs of the best state), `compare`, `policy`, and check that every manifest is
valid JSON and every figure is there. Then the 2021 entry point (`run_ql_mod`) on `kochi` and `new_kochi`, 2 simulations of 1 min each. Small settings: this checks that the chain works and
agrees with itself, not how good a policy is.

| Area | Agents | Rebuilt tables = shipped | `sp` | `calibrate` | `evaluate` |
|---|---:|---|---:|---:|---:|
| `kochi_area0` | 4,196 | yes | 95 s | 324 s | 26 s |
| `kochi_area1` | 2,743 | yes | 70 s | 278 s | 19 s |
| `kochi_area2` | 1,704 | yes | 59 s | 171 s | 15 s |
| `kochi_area4` | 13,502 | yes | 242 s | 842 s | 96 s |
| `kochi` (2021 entry point) | 35,930 | | | 22 s | |
| `new_kochi` (2021 entry point) | 148,810 (19,207 nodes) | | | 425 s | |

All steps exit 0, `compare` and `policy` draw their figures, and the manifests are present (`sp`, `calibrate`, `evaluate`). All tables are byte-identical except the node coordinates of `kochi_area4`, where three values differ in the last of their six decimals (1e-6 m: a cluster
centre on a rounding boundary, which NumPy versions round differently; the script allows 1.5e-6, and `tests/test_casebuild_cases.py` compares with a tolerance).
Area 3 of the study has no shelter in its box and is not a case.

## Tests on four Python versions

`clean_checkout.sh` ([Step 5](../step5/README.md)) on commit `8791697`, the last code commit (later commits changed documents only): a fresh clone, a new environment per version, nothing installed beyond
the README's install.

| | 3.10 | 3.11 | 3.12 | 3.13 |
|---|---|---|---|---|
| `pip install -e .`, `python -W error -m unittest discover tests` | 300 OK (2 skipped) | 300 OK (2) | 300 OK (2) | 300 OK (2) |
| `pip install -e ".[casebuild]"`, same | 336 OK | 336 OK | 336 OK | 336 OK |

`ruff` clean, `CITATION.cff` valid, wheel and sdist build and pass `twine check`, the wheel installed elsewhere imports and both console scripts answer `--help`, and the README quick
start runs from it (`sp` 31 s, `calibrate` 346 s, `evaluate` 20 s, `compare` and `policy` 1 s). GitHub Actions on the final commit: lint, build and the 8 test jobs all green.

## The 2024 study's conclusions (D11)

**What there is to examine.** The study's repository ([erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara), "Urushibara's B4 codes") holds
code and data, not conclusions: its README is a title, and the notebooks hold working notes ("the GLEE factor might be the one making it converge always",
"we need to see if reinforcing the best policy is better than just keep iterating with a suboptimal policy"). The text of the thesis is not in it, so its wording
cannot be checked here. What can be checked are the numbers and methods that a conclusion about learned policies would rest on. There are four, and each changes.

**1. The shortest-path baseline.** The committed evacuation times of its 1,000 shortest-path runs per area carry the far-end defect (agents that stop at the end of a
link and never arrive: 43 % of the `kochi2` runs and 53 % of `kochi1` end with agents left; [engine reconciliation](../../engine-reconciliation.md)). They are regenerated
in [results/kochi2024_regenerated](../../../results/kochi2024_regenerated/README.md) (D8): **do not cite the committed times.**

**2. The measure that picks the "best" policy.** The study's calibration trains with a random-choice rate that falls from 1 to 0.5 and keeps the simulation in which
most agents were safe, whatever the agents did in it. In those simulations 50 to 100 % of the choices are random, so the count measures the random-choice rate and
the luck of the draw, not the policy. `study_metric.py` trains on the study's own `kochi2` (622 agents, 30 min, 100 simulations, one seed per configuration) and reads
both the study's measure and the policy that was trained, followed greedily and frozen on 10 other seeds (shortest path: 527.2 ± 6.8 safe at 30 min):

| | most safe in any training run (the study's measure) | last 10 training runs | the final policy, greedy and frozen |
|---|---:|---:|---:|
| A. the study: SARSA, its options, discount 0.9 per decision | 262 (50 % of shortest path) | 246 (47 %) | **456 (87 %)** |
| B. as A, discount 0.999 per second | 277 (53 %) | 262 (50 %) | **481 (91 %)** |
| C. this package's defaults, Q-learning | 284 (54 %) | 257 (49 %) | **527 (100 %)** |

The study's measure reads about half of the shortest path for all three and cannot tell a policy at 87 % from one at 100 %. A statement of the form "learning reaches
X % of the shortest path" that rests on it is a statement about the random-choice rate. **Do not cite such figures.**

**3. The discount.** With the study's options and its discount (A) the trained policy is at 87 % of the shortest path; changing only the discount to 0.999 per second (B)
gives 91 %, and the defaults of this package (C, which also use Q-learning and the whole-link density code) 100 %. Under 0.9 per decision one node costs about 1,200 s
of walking in the learning target, so the policy it converges to walks farther than the shortest path (Step 4: the exact optimum of that target on `kochi2` is
80.7 %). **A conclusion that learned policies stop short of the shortest path because learning is limited is an effect of the discount**; it does not hold for a
corrected one.

**4. Crowded areas.** If the conclusion were that learned guidance improves the evacuation, these results do not support it: on `kochi_area4` (13,502 agents) no
learned policy beat the shortest path within the budgets tried, and over a whole evacuation they did worse (D9, [Step 4](../step4/README.md)). That is "not shown", not "shown not to work".

**What may be said.** (a) Regenerated shortest-path distributions (D8) can be cited; the committed ones cannot. (b) On the study's `kochi2`, Q-learning with the
per-second discount learns a policy as good as the shortest path (100 %, one seed), where the study's configuration reaches 87 % under the same evaluation. (c) On a crowded
area, no advantage of learning over the shortest path has been shown. (d) Policies must be compared in greedy, frozen runs on seeds not used in training, never by the best
training simulation. Limits of this check: one case, 100 simulations and one seed per configuration, 10 evaluation runs.

## D15: smaller clones

Where the size is (measured on the repository as pushed, `shrink_history.sh`):

| | download (`.git`) | files on disk |
|---|---:|---:|
| full clone, before | 154 MB | 343 MB |
| `git clone --depth 1`, before | 61 MB | 343 MB |
| full clone, with the GIS layers out of the tip (this commit) | 154 MB (history unchanged) | 114 MB |
| `git clone --depth 1`, same | 50 MB | 114 MB |
| full clone, after the history is rewritten (below) | **about 54 MB** (measured as the pack of the rewritten repository) | 114 MB |

The GIS layers (277 MB) are mostly zeros in a pack (13 MB compressed), so removing them from the tip shrinks the *checkout* by 277 MB but the *download* little. What
weighs in the download is history: files that were deleted long ago and are still in it (two AVI videos 51 MB, the 2021 Arahama and Kochi state dumps about 50 MB).

**Done now, without touching history:** `datasets/gis/data/qgis` and `qgis_1` are no longer tracked and are ignored (`datasets/gis/README.md` says how to restore them; an 8 MB
backup archive was sent to you); `README.md` tells how to make a light clone (`--depth 1`).

**Prepared, not applied: rewriting the history** (`shrink_history.sh`). It removes from all four branches (`main`, `dev`, `SPvsRL`, `regid/trusting-curie-ogh40k`) the files listed in it,
leaves the tip of every branch identical apart from those paths (checked branch by branch: 1,238 to 1,401 files each, no difference) and shrinks a full clone from 153 MiB
to 54 MiB. It is **not pushed**: it needs a force-push to branches that are not this session's. What a push does:

* every commit of every branch gets a new hash; clones and forks have to be made again, and open pull requests and links to old commits break;
* the old objects stay reachable on GitHub by their old hashes (and in forks) until GitHub collects them; only GitHub Support can purge them early;
* the e-mail addresses in the author fields of the history stay (changing them would rewrite attribution, which is a separate choice);
* the removed files are gone from the repository: keep the backup archive (the GIS layers) and, if the Arahama state dumps matter, a copy of the old history.

To apply it, say so, or run `sh docs/audits/step6/shrink_history.sh --push` yourself with `git-filter-repo` installed.

## What is still open

**Needs your word**
* **Rewriting the history** (D15, above): prepared and verified, not pushed. A full clone drops from 153 MiB to 54 MiB; it needs a force-push of all four branches.
* **The `@author` headers of five files in `pre/`** (D14): they name Moya and luismoya, and `pre/DisaggregationLibrary.py` hard-codes `C:\Users\Moya\ReGID Dropbox\Luis Moya\...`. They were
  not changed to Erick Mas, because that would credit the repository owner with a file whose own paths say another person wrote it. If those files were in fact written by the
  owner, say so and the headers change; if not, their GPL-3.0 licensing needs their author's agreement ([data-licences.md](../../data-licences.md#code-written-by-others)).

**Defects found and not fixed** (each changes a recorded result or is a legacy path; none affects a conclusion of the audits)
* **Agents that depart before second 10 are almost stationary until it** (finding 10): the link speeds are first computed at `t % 10 == 0`. About 1 to 2 agents on the shipped
  cases (mean departure 5 min), many with a mean departure of a few seconds. Fixing it changes every recorded run, so it belongs in a version with the golden recordings regenerated and reported (as in Step 2).
* **The shortest-path agent takes the first of several parallel links** to its next node, although the route was computed with the shortest one (S1): the crowding is booked on the first link. The shipped
  areas have 5 to 26 node pairs with parallel links (`kochi_area4`: 26 pairs among its 1,634 links), `new_kochi` none. Fixing it changes the shortest-path baseline slightly, so the D8 and D9 figures would be measured again.
* **`survivorsPerSim[-1] == case.pedDB.shape[0]` in the five 2021 scripts** (finding 11c) compares a list with an integer: the "stop when everybody has survived" never fires, and a resumed run (`numSim0 > 0`) overwrites its survivors file.
* **`segmentIndex="raw"`** (the 2021 and 2024 presets) can raise an `IndexError` where it is documented to freeze an agent (finding 13); the default is not affected, and none of the shipped cases reaches it.
* **`calibrate --sp`** does not check that the reference run used the same `--time`, `--departure` and options (S3). **`resizePedestrianDB` and `resizePedDB`** are unused and wrong (S5): they can be deleted.

**Not verified or not recorded**
* The Kochi Prefectural Office layers are recorded as public domain on the owner's statement; their terms were not looked at (D13, accepted). The source and terms of the census and building databases,
  the tsunami rasters, the shelter register and the area polygons are not recorded; the date of the OpenStreetMap download is not recorded; basemap images in some notebooks were not checked for attribution
  ([data-licences.md](../../data-licences.md)).
* Only Linux was run (Python 3.10 to 3.13). Windows and macOS are untested.
* The e-mail addresses in the git history stay until the history is rewritten, and then they still stay (changing them rewrites attribution).

**The scientific question**
* Whether tabular reinforcement learning ever beats the shortest path on a crowded area is **not shown** (D9, [Step 4](../step4/README.md)). The long training run that could settle it (D10) was declined. This is what a
  paper built on this repository would still have to do, with a state that can express "this street is full" without the cycling of untrained states.

