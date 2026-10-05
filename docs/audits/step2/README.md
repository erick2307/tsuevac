# Step 2 audit: Q-learning, and what the discount applies to

`learning.py` trains the learning methods on `kochi2` of [erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara)
(309 nodes, 622 agents, 4 shelters; mean departure 5 min; 30 min simulated) and evaluates each learned policy greedily.

```
git clone https://github.com/erick2307/2024_urushibara ~/2024_Urushibara     # or set URUSHIBARA_DIR
pip install -e ".[plots]" scipy
cd docs/audits/step2
python learning.py 100 50 3        # E1 sims, E2 sims, seeds; about 25 min on 4 cores
python analyze_learning.py
```

**Protocol** (as in `EVACMODEL3_FocalPoints/calibration.py`): each simulation starts from the state matrix of the previous one;
the probability of a random choice falls as `1 / (s / N + 1)`, from 1 to 0.5 over the N simulations. Afterwards the policy is
evaluated on 5 fresh simulations in which every agent always takes the best-valued action. The measure is the number of agents
safe after 30 min. Settings: the default `ModelOptions()` (clamped segment index, entry speed by position, reward 1e5, density
code of the whole link), `discounting` as stated. For scale: shortest path reaches **529** of the 622 at 30 min under the same
settings ([Step 1](../../engine-reconciliation.md)).

## E1. SARSA and Q-learning (100 training simulations, 3 seeds)

| | greedy survivors at 30 min | % of shortest path | per seed | training survivors, first 10 → last 10 simulations |
|---|---|---|---|---|
| SARSA | 428.4 | 81.0 % | 421, 433, 432 | 87 → 256 |
| Q-learning | 430.1 | 81.3 % | 431, 430, 429 | 90 → 256 |

Q-learning − SARSA: +1.7 survivors, Welch p = 0.71 on the per-seed means (3 seeds each; the 5 evaluations of a seed share one
policy, so the seeds are the replicates). **No difference can be seen** on this case. That
is not surprising: the two updates differ only through the exploration the agents do, and here exploring costs little (time,
nothing catastrophic) and ends at 50 %. The implementation is verified by hand-computed unit tests (`tests/test_td.py`), not by
this experiment; what this experiment shows is that choosing Q-learning over SARSA is not what limits the results on `kochi2`.

## E2. Monte Carlo and the discount (50 training simulations, 3 seeds)

| Monte Carlo, discount 0.9 applied | greedy survivors at 30 min | % of shortest path | per seed | training survivors, first 5 → last 5 |
|---|---|---|---|---|
| once per **second** (what it always did) | **39.5** | 7.5 % | 41, 38, 39 | 78 → 106 |
| once per **decision** (as SARSA and Q-learning do) | **380.2** | 71.8 % | 379, 389, 372 | 86 → 247 |

Per decision − per second: +340.7 survivors, p = 1.4·10⁻⁴ (per-seed means, 3 seeds each). With `0.9 ** seconds` a shelter 100 s away is weighted by 3·10⁻⁵, so
the shelter's reward is invisible and the step penalty alone steers the agents: the greedy policy is **worse than walking at
random** (about 78 survivors in the first, nearly random, simulations). This is a statement about 0.9 per second, not about discounting by
time as such: a factor close to 1 per second (0.999) was tried in Step 4 and works well for Q-learning. The two rules are not interchangeable, and
the TD methods have always used the per-decision one.

## What this does not show

* Three seeds, one case (`kochi2`, little crowding), 100 and 50 simulations. The training curves were still rising, though the
  greedy result did not improve between 50 simulations (Step 1: 436.3 for SARSA with the same dynamics) and 100 (428.4): the
  difference is inside the noise, and says plateau, not trend.
* **Both TD methods stop at about 81 % of the shortest path** at 30 min. Explained in [Step 4](../step4/README.md): it is not the
  training and not the algorithm but the target. With the discount 0.9 applied once per decision, one node more or less on the way
  to a shelter is worth about as much as 1,200 s of walking, so the best policy of that target minimises the number of nodes, not the
  distance; solved exactly, that target gives 80.7 % of the shortest path, which is what was learned. With 0.999 per *second*
  Q-learning reaches 99 % of the shortest path. (The reward 1e5 and the density code, the other candidates named here, are not what limits it.)
* A policy learned by Monte Carlo is not compared with TD ones beyond this: it had 50 simulations, the TD methods 100.

## Decision

`discounting="decision"` for all three methods as the default (Monte Carlo changes; SARSA and Q-learning do not).
**Confirmed and applied** at the start of Step 3: only the default-option Monte Carlo golden recording changed.
**Superseded in Step 4 (D7):** the default is now 0.999 per *second* for all three methods ([Step 4](../step4/README.md)). The numbers of
this audit were measured with 0.9 per decision (`ModelOptions(discounting="decision", discount=0.9)`; `learning.py` says so).
`ModelOptions.legacy()` keeps `"method"`, so the 2021 Monte Carlo remains reproducible.
