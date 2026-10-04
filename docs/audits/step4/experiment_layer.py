# -*- coding: utf-8 -*-
"""4b. The experiment layer against the study's own results.

    python experiment_layer.py [RUNS] [WORKERS]        (default 200 4; about 25 min on 4 cores; needs scipy and the 2024 repository)

A  the shortest-path distribution of the study's kochi2: its 1,000 committed runs (results_time.csv, results_df.csv) against
   `repeat_shortest_path` with `ModelOptions.kochi2024()` (the study's engine, bit for bit), and with the default options (the
   far-end freeze fixed). Same case, 120 min simulated, mean departure 5 min.
B  seeded reruns are identical: the same seed gives the same curves in 1 or 4 worker processes and on a second run.
C  `shortest_path_run` is the loop of the earlier audits: the same seed gives the same curve to the last agent.
D  how many runs the convergence rule asks for.
"""
import os
import sys

import numpy as np
from common import URU

from evacrl.experiment import Case, derive_seeds, repeat_shortest_path, shortest_path_run
from evacrl.options import ModelOptions
from evacrl.sarsa import SARSA

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FOLDER = f"{URU}/results_for Usama/kochi2"
SIM = 7200


def committed():
    """The study's 1,000 runs: evacuation time and the safe agents at 30 min (the value recorded after second 1799)."""
    times = np.loadtxt(f"{FOLDER}/results_time.csv", delimiter=",", skiprows=1)
    data = np.loadtxt(f"{FOLDER}/results_df.csv", delimiter=",", skiprows=1)
    starts = np.r_[0, np.where(np.diff(data[:, 0]) < 0)[0] + 1, len(data)]
    safe30, final = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        run = data[a:b]
        safe30.append(run[run[:, 0] == 1799, 1][0])
        final.append(run[-1, 1])
    return times, np.array(safe30), np.array(final), len(starts) - 1


def line(label, x):
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"{label:28s} n={len(x):5d} mean {x.mean():8.1f} sd {x.std(ddof=1):7.1f} cv {x.std(ddof=1) / x.mean():.4f}   "
            f"5/25/50/75/95 %: " + " ".join(f"{v:7.0f}" for v in q))


if __name__ == "__main__":
    from scipy.stats import ks_2samp, mannwhitneyu
    case = Case.load(FOLDER)
    times, safe30, final, nruns = committed()
    print(f"kochi2: {case.name}, {RUNS} new runs per variant against the committed {nruns}\n")

    print("A  distribution of the evacuation time (s) and of the agents safe at 30 min\n")
    print(line("committed, evacuation time", times))
    print(line("committed, safe at 30 min", safe30))
    new = {}
    for label, options in (("kochi2024 options", ModelOptions.kochi2024()), ("default options", ModelOptions())):
        res = repeat_shortest_path(case, RUNS, seed=1, workers=WORKERS, options=options, sim_time=SIM, mean_departure=5.0, batch=20)
        new[label] = res
        last, safe = res.metric("last_evacuee"), res.metric("safe", 1800)
        print(line(f"{label}, evacuation time", last))
        print(line(f"{label}, safe at 30 min", safe))
        print(f"{'':28s} two-sample tests against the committed runs: evacuation time KS p = {ks_2samp(last, times).pvalue:.3f}, "
              f"Mann-Whitney p = {mannwhitneyu(last, times).pvalue:.3f}; safe at 30 min KS p = {ks_2samp(safe, safe30).pvalue:.3f}, "
              f"Mann-Whitney p = {mannwhitneyu(safe, safe30).pvalue:.3f}")
        print(f"{'':28s} runs with agents left after {SIM // 60} min: {int(np.sum([r.evacuated < r.agents for r in res.runs]))} of {len(res.runs)} "
              f"({100 * np.mean([r.evacuated < r.agents for r in res.runs]):.0f} %); committed: {int(np.sum(final < 622))} of {nruns} ({100 * np.mean(final < 622):.0f} %)")

    print("\nB  the same seed, 1 and 4 workers, and a second time (default options, 12 runs)\n")
    a = repeat_shortest_path(case, 12, seed=7, workers=1, sim_time=SIM, batch=6)
    b = repeat_shortest_path(case, 12, seed=7, workers=4, sim_time=SIM, batch=6)
    c = repeat_shortest_path(case, 12, seed=7, workers=4, sim_time=SIM, batch=6)
    same = lambda x, y: all(np.array_equal(p.curve, q.curve) for p, q in zip(x.runs, y.runs)) and x.seeds == y.seeds
    print(f"1 worker = 4 workers: {same(a, b)}    4 workers twice: {same(b, c)}    another seed differs: "
          f"{not same(a, repeat_shortest_path(case, 12, seed=8, workers=4, sim_time=SIM, batch=6))}")

    print("\nC  the loop of the earlier audits (Step 3, effect_on_results.py), same seeds\n")
    ok = []
    for seed in derive_seeds(7, 3, "shortest_path"):
        np.random.seed(seed)
        m = SARSA(meanRayleigh=5 * 60, options=ModelOptions(), **case.kwargs)
        m.loadShortestPathDB(case.nextnode)
        t0, curve = int(min(m.pedDB[:, 9])), []
        for t in range(t0, SIM):
            m.initEvacuationAtTime()
            m.stepForward()
            m.checkTargetShortestPath()
            if not t % 10:
                m.computePedHistDenVelAtLinks()
                m.updateVelocityAllPedestrians()
            curve.append(m.getNumberEvacuatedPed())
        ok.append(np.array_equal(np.array(curve), shortest_path_run(case, seed, sim_time=SIM).curve))
    print("identical curves for 3 seeds:", all(ok))

    print("\nD  the convergence rule: running mean, CV and relative standard error of the new runs\n")
    for label, res in new.items():
        print(label)
        for metric in ("last_evacuee", "safe"):
            rows = res.trace[metric]
            picks = [c for c in rows if c.n in (20, 40, 100, 200) or c is rows[-1]]
            print(f"  {metric:13s} " + "   ".join(f"n={c.n}: mean {c.mean:.0f}, cv {c.cv:.3f}, sem {100 * c.sem_rel:.2f}%" for c in picks))
