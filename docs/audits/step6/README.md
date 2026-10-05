# Step 6 audit (in progress)

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

