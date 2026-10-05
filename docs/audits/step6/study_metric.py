# -*- coding: utf-8 -*-
"""D11. What the 2024 study measured when it trained, and what the policies it trained are worth.

The study's calibration (`EVACMODEL3_FocalPoints/calibration.py`) trains with a random-choice rate that falls from 1 to 0.5 and keeps the simulation in which
most agents were safe, *whatever the agents were doing in it*: a run in which half the choices were random. This script trains on the study's `kochi2`
(622 agents, 30 min) with the study's protocol (100 simulations, each continuing from the state the last left), and reports for each configuration

  * the study's metric: the most safe in any training simulation, and the mean of the last 10 (exploring runs);
  * what the policy is worth: the final state followed greedily and frozen, 10 runs on seeds not used in training;
  * the shortest path on the same case, 10 runs.

    A  the study's options and SARSA (discount 0.9 per decision)         = what the study did
    B  A with the discount changed to 0.999 per second                    = only the discount differs from A
    C  the defaults of this package, Q-learning

    URUSHIBARA_DIR=~/2024_Urushibara/EVACMODEL3_FocalPoints python study_metric.py [sims]      (about 20 min on 3 cores)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from evacrl.experiment import Case, calibrate, evaluate_policy, repeat_shortest_path  # noqa: E402
from evacrl.options import ModelOptions  # noqa: E402

URU = os.environ.get("URUSHIBARA_DIR", os.path.expanduser("~/2024_Urushibara/EVACMODEL3_FocalPoints"))
CASE = os.path.join(URU, "results_for Usama", "kochi2")
SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CONFIGS = {
    "A": ("the study: SARSA, its options, discount 0.9 per decision", "sarsa", ModelOptions.kochi2024()),
    "B": ("A with 0.999 per second", "sarsa", ModelOptions.kochi2024().replace(discount=0.999, discounting="second")),
    "C": ("this package's defaults: Q-learning", "qlearning", ModelOptions()),
}


def job(key):
    t0 = time.time()
    label, method, options = CONFIGS[key]
    case = Case.load(CASE)
    res = calibrate(case, method=method, sims=SIMS, eval_every=SIMS // 4, eval_runs=5, seed=0, options=options)
    explore = np.array(res.training_safe, dtype=float)
    final = np.array([r.evacuated for r in evaluate_policy(case, method, res.final_state, 10, seed=21, options=options)], dtype=float)
    best = np.array([r.evacuated for r in evaluate_policy(case, method, res.best_state, 10, seed=21, options=options)], dtype=float)
    return dict(key=key, label=label, agents=res.agents, max_explore=explore.max(), argmax=int(explore.argmax()) + 1, last10=explore[-10:].mean(),
                final=final.mean(), best=best.mean(), best_sim=res.best_sim, secs=round(time.time() - t0))


if __name__ == "__main__":
    case = Case.load(CASE)
    sp = repeat_shortest_path(case, 10, seed=3, workers=1, sim_time=1800, horizon=1800)
    spsafe = np.array([r.safe_at(1800) for r in sp.runs], dtype=float)
    agents = sp.runs[0].agents
    print(f"kochi2 of the study: {agents} agents, {SIMS} training simulations of 30 min; shortest path {spsafe.mean():.1f} +- {spsafe.std(ddof=1):.1f} safe at 30 min (10 runs)\n")
    with Pool(3) as pool:
        out = pool.map(job, list(CONFIGS), chunksize=1)
    print(f"{'':58s} {'most safe in a training run':>28s} {'last 10 runs':>13s} {'final policy, greedy':>21s} {'best checkpoint':>16s}")
    for r in out:
        pct = lambda x: f"{x:6.1f} ({100 * x / spsafe.mean():4.0f} %)"
        print(f"{r['key']}  {r['label']:55s} {pct(r['max_explore']):>28s} {pct(r['last10']):>13s} {pct(r['final']):>21s} {pct(r['best']):>16s}   [{r['secs']} s]")
    print("\n(% of the shortest path's mean at 30 min)")
