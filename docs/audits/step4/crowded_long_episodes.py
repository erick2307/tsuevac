# -*- coding: utf-8 -*-
"""D9 (continued). Train on the whole evacuation: episodes of 120 min instead of 30.

    python crowded_long_episodes.py [CASE] [SIMS]        (default kochi_area4 30; about 3 h on 2 cores)

crowded_area.py trained on 30 min of the evacuation and the policies then left a quarter of the agents on their way after 2 h (the tabular
policy is flat, so arbitrary, at the nodes and crowded states that 30 minutes of training never showed it; initialising new states from the
node's values did not help, it made it worse). Here the episodes are 120 min, so that every phase of the evacuation is seen in training. The
checkpoints (every SIMS / 3 simulations) are evaluated greedily and frozen on 2 seeds over the 120 min, and the best is kept; then 5 fresh runs of it.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from common import HERE

from evacrl.experiment import Case, calibrate, compare_with_shortest_path, evaluate_policy
from evacrl.experiment.output import write_state
from evacrl.options import ModelOptions

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi_area4"
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
T = 7200


def job(level):
    t0 = time.time()
    case = Case.load(CASE)
    options = ModelOptions(densityLevel=level)
    log = open(os.path.join(HERE, f"long_{CASE}_{level}.progress"), "a")
    note = lambda h: (log.write(f"sim {h.sim} epsilon {h.epsilon:.3f} exploring {h.train_safe:.0f} greedy {h.eval_mean:.1f} +- {h.eval_sd:.1f} after {round(time.time() - t0)} s\n"), log.flush())
    res = calibrate(case, method="qlearning", sims=SIMS, eval_every=SIMS // 3, eval_runs=2, seed=9, options=options, sim_time=T, progress=note)
    write_state(os.path.join(HERE, f"state_{CASE}_{level}_long.csv"), res.best_state)
    runs = evaluate_policy(case, "qlearning", res.best_state, 5, seed=13, options=options, sim_time=T)
    c = compare_with_shortest_path(case, res.best_state)
    return dict(level=level, agents=res.agents, history=[(h.sim, h.epsilon, h.train_safe, h.eval_mean, h.eval_sd) for h in res.history], best_sim=res.best_sim,
                safe120=[r.evacuated for r in runs], safe30=[r.safe_at(1800) for r in runs], safe60=[r.safe_at(3600) for r in runs],
                last=[r.last_evacuee for r in runs], agreement=c.agreement, longer=c.longer, secs=round(time.time() - t0))


if __name__ == "__main__":
    with Pool(2) as pool:
        out = pool.map(job, ["link", "segment"], chunksize=1)
    json.dump(out, open(os.path.join(HERE, f"long_{CASE}.json"), "w"))
    n = out[0]["agents"]
    print(f"{CASE}: {n} agents, Q-learning trained on episodes of 120 min, {SIMS} simulations; 5 frozen greedy runs of the best checkpoint\n")
    print(f"{'density code':14s} {'safe at 30 min':>16s} {'at 60 min':>10s} {'at 120 min':>11s} {'evacuated':>10s} {'best after':>10s}")
    for r in out:
        print(f"{r['level']:14s} {np.mean(r['safe30']):9.1f} ± {np.std(r['safe30'], ddof=1):4.1f} {np.mean(r['safe60']):10.1f} {np.mean(r['safe120']):11.1f} {100 * np.mean(r['safe120']) / n:9.1f}% {r['best_sim']:10d}")
    for r in out:
        print(f"{r['level']:8s} greedy evaluation of the checkpoints (sims: safe at 120 min): " + ", ".join(f"{h[0]}: {h[3]:.0f}" for h in r["history"]))
