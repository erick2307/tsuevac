# -*- coding: utf-8 -*-
"""D9. Does learning beat the shortest path where crowding binds? `kochi_area4` (1,110 nodes, 15 shelters, 13,502 agents).

    python crowded_area.py [CASE] [SIMS]        (default kochi_area4 60; about 1.5 h on 3 cores)

On `kochi2` (622 agents) the shortest path is nearly the optimum, so a learner can at best equal it (Step 4a). Here the shortest path puts
13,502 people on the same streets and about 55 % are safe after 30 min, so a policy that sees crowding could do better. Q-learning with
the default options (0.999 per second), trained as in the study (every episode continues from the state the last one left, the
random-choice rate falls as 1 / (s / N + 1), 30 min, mean departure 5 min), a checkpoint every SIMS / 3 simulations evaluated greedily and
frozen on 3 fixed seeds, the best checkpoint kept; then 10 fresh greedy runs of it. Two density codes of the state: `link` (the default: the whole
link, 2 m wide) and `segment` (the worst 2 m segment, real width). The shortest-path baseline is 10 runs of 30 min.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from common import HERE

from evacrl.experiment import (Case, calibrate, compare_with_shortest_path, evaluate_policy, repeat_shortest_path)
from evacrl.experiment.output import write_state
from evacrl.options import ModelOptions

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi_area4"
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
T = 1800


def job(what):
    t0 = time.time()
    case = Case.load(CASE)
    if what == "sp":
        res = repeat_shortest_path(case, 10, seed=3, workers=1, sim_time=T, horizon=T, batch=5)
        return dict(what="sp", agents=res.runs[0].agents, safe=[r.safe_at(T) for r in res.runs], secs=round(time.time() - t0))
    options = ModelOptions(densityLevel=what)
    log = open(os.path.join(HERE, f"crowded_{CASE}_{what}.progress"), "a")        # what has been done survives an interrupted run
    note = lambda h: (log.write(f"sim {h.sim} epsilon {h.epsilon:.3f} exploring {h.train_safe:.0f} greedy {h.eval_mean:.1f} +- {h.eval_sd:.1f} after {round(time.time() - t0)} s\n"), log.flush())
    res = calibrate(case, method="qlearning", sims=SIMS, eval_every=SIMS // 3, eval_runs=3, seed=7, options=options, sim_time=T, progress=note)
    write_state(os.path.join(HERE, f"state_{CASE}_{what}.csv"), res.best_state)
    runs = evaluate_policy(case, "qlearning", res.best_state, 10, seed=11, options=options, sim_time=T)
    c = compare_with_shortest_path(case, res.best_state)
    crowded = (res.best_state[:, 1:11] > 0).any(axis=1)
    visits = res.best_state[:, 21:31].sum(axis=1)
    return dict(what=what, agents=res.agents, history=[(h.sim, h.epsilon, h.train_safe, h.eval_mean, h.eval_sd) for h in res.history],
                best_sim=res.best_sim, safe=[r.safe_at(T) for r in runs], longer=c.longer, agreement=c.agreement, hops=c.hops, hops_sp=c.hops_sp,
                states=int(res.best_state.shape[0]), crowded_share=float(100 * visits[crowded].sum() / visits.sum()), secs=round(time.time() - t0))


if __name__ == "__main__":
    with Pool(3) as pool:
        out = pool.map(job, ["link", "segment", "sp"], chunksize=1)
    json.dump(out, open(os.path.join(HERE, f"crowded_{CASE}.json"), "w"))
    n = out[0]["agents"] if out[0]["what"] != "sp" else out[2]["agents"]
    sp = next(r for r in out if r["what"] == "sp")
    print(f"{CASE}: {n} agents; safe at 30 min (mean ± sd of 10 runs)\n")
    print(f"{'':28s} {'safe at 30 min':>20s} {'% of agents':>12s} {'vs shortest path':>17s} {'walk vs SP':>11s} {'first choice = SP':>18s} {'states':>7s} {'decisions in crowded':>21s}")
    s = np.array(sp["safe"], dtype=float)
    print(f"{'shortest path':28s} {s.mean():12.1f} ± {s.std(ddof=1):5.1f} {100 * s.mean() / n:11.1f}% {'':17s}")
    for r in out:
        if r["what"] == "sp":
            continue
        x = np.array(r["safe"], dtype=float)
        print(f"{'Q-learning, ' + r['what']:28s} {x.mean():12.1f} ± {x.std(ddof=1):5.1f} {100 * x.mean() / n:11.1f}% {100 * (x.mean() / s.mean() - 1):+16.1f}% "
              f"{r['longer']:+10.1f}% {100 * r['agreement']:17.1f}% {r['states']:7d} {r['crowded_share']:20.2f}%")
    print("\nlearning curves (greedy evaluation of the checkpoints on 3 seeds: sims, mean safe)")
    for r in out:
        if r["what"] != "sp":
            print(f"{r['what']:8s} " + ", ".join(f"{h[0]}: {h[3]:.0f}" for h in r["history"]) + f"   (best after {r['best_sim']}; exploring runs: " + ", ".join(f"{h[2]:.0f}" for h in r["history"]) + ")")
