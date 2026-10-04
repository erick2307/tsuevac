# -*- coding: utf-8 -*-
"""4a. Does the discount explain the plateau? Q-learning with the discount of the 2021 and 2024 code, and with a per-second discount.

    python gamma_learning.py [CASE]        (default kochi2; about 40 min on 4 cores)

Uses the experiment layer (`evacrl.experiment.calibrate`): training as in the study (every episode continues from the state the last one
left, the random-choice rate falls as 1 / (s / N + 1), 30 min simulated, mean departure 5 min), a checkpoint every N / 4 simulations
evaluated greedily on 5 fixed seeds, and the best checkpoint kept. The evaluation runs let the agents go on learning on a copy of the state
(`eval_learn=True`), as the 2024 calibration did and as the numbers of this audit were measured; `frozen_evaluation.sh` evaluates the
saved policies frozen as well, the default of the experiment layer. For each result the policy is also followed from the start node of every agent and compared
with the shortest path (`compare_with_shortest_path`): how much longer its walks are, and how often its first choice is the shortest
path's. Shortest path on the same case is the reference (policy_ceiling.py).

    configuration                      seeds    simulations
    gamma 0.9 per node (2021/2024)     0, 1     100        and one seed of 300 (is it the length of the training?)
    gamma 0.999 per second             0, 1     100        and one seed of 300
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from common import HERE, case_files

from evacrl.experiment import Case, calibrate, compare_with_shortest_path
from evacrl.experiment.output import write_state
from evacrl.options import ModelOptions

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi2"
FILES, SP = case_files(CASE)
CONFIGS = [(0.9, "decision", 100, 0), (0.9, "decision", 100, 1), (0.999, "second", 100, 0), (0.999, "second", 100, 1),
           (0.9, "decision", 300, 0), (0.999, "second", 300, 0)]


def job(config):
    discount, discounting, sims, seed = config
    from common import URU
    case = Case.load(f"{URU}/results_for Usama/kochi2") if CASE == "kochi2" else Case.load(CASE)
    res = calibrate(case, method="qlearning", sims=sims, eval_every=sims // 4, eval_runs=5, seed=seed, discount=discount,
                    options=ModelOptions(discounting=discounting), eval_learn=True)
    c = compare_with_shortest_path(case, res.best_state)
    out = dict(discount=discount, discounting=discounting, sims=sims, seed=seed, agents=res.agents, best_sim=res.best_sim, best_eval=res.best_eval,
               history=[(h.sim, h.epsilon, h.train_safe, h.eval_mean, h.eval_sd) for h in res.history], longer=c.longer, agreement=c.agreement,
               metres=c.metres, metres_sp=c.metres_sp, hops=c.hops, hops_sp=c.hops_sp, never_arrive=c.never_arrive)
    write_state(os.path.join(HERE, f"state_{CASE}_{discounting}_{discount}_{sims}_{seed}.csv"), res.best_state)
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        out = pool.map(job, CONFIGS, chunksize=1)
    json.dump(out, open(os.path.join(HERE, f"gamma_learning_{CASE}.json"), "w"))
    n = out[0]["agents"]
    print(f"Q-learning on {CASE} ({n} agents): best checkpoint, greedy evaluation on 5 fixed seeds, safe at the end of 30 min\n")
    print(f"{'discount':>8s} {'per':>9s} {'sims':>5s} {'seed':>4s} {'safe at 30 min':>15s} {'= %':>6s} {'best after':>10s} {'walk vs shortest':>17s} {'first choice = SP':>18s} {'nodes passed':>13s}")
    for r in out:
        print(f"{r['discount']:8g} {r['discounting']:>9s} {r['sims']:5d} {r['seed']:4d} {r['best_eval']:15.1f} {100 * r['best_eval'] / n:5.1f}% {r['best_sim']:10d} "
              f"{r['longer']:+16.1f}% {100 * r['agreement']:17.1f}% {r['hops']:6.1f} vs {r['hops_sp']:.1f}")
    print("\nlearning curves (greedy evaluation of the checkpoints: sims, mean safe)")
    for r in out:
        print(f"{r['discount']:g} per {r['discounting']:8s} seed {r['seed']}, {r['sims']} sims: " + ", ".join(f"{h[0]}: {h[3]:.0f}" for h in r["history"]))
