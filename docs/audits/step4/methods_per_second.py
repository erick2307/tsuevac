# -*- coding: utf-8 -*-
"""4a (continued). Do SARSA and Monte Carlo also reach the shortest path with a per-second discount of 0.999?

    python methods_per_second.py [CASE]        (default kochi2; about 10 min on 2 cores)

One seed, 60 training simulations as in gamma_learning.py, checkpoints every 20 simulations evaluated greedily on 5 seeds.
"""
import sys
from multiprocessing import Pool

from common import URU

from evacrl.experiment import Case, calibrate, compare_with_shortest_path
from evacrl.options import ModelOptions

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi2"


def job(method):
    case = Case.load(f"{URU}/results_for Usama/kochi2") if CASE == "kochi2" else Case.load(CASE)
    res = calibrate(case, method=method, sims=60, eval_every=20, eval_runs=5, seed=0, discount=0.999, options=ModelOptions(discounting="second"), eval_learn=True)
    c = compare_with_shortest_path(case, res.best_state)
    return method, res, c


if __name__ == "__main__":
    with Pool(2) as pool:
        out = pool.map(job, ["sarsa", "mc"], chunksize=1)
    print(f"{CASE}: discount 0.999 per second, 60 simulations, one seed; greedy evaluation of the checkpoints on 5 seeds (shortest path reaches 529 of 622)\n")
    for method, res, c in out:
        print(f"{method:9s} safe at 30 min: " + ", ".join(f"{h.sim}: {h.eval_mean:.0f}" for h in res.history) +
              f"   best {res.best_eval:.1f} of {res.agents}; walk {c.longer:+.1f} % against the shortest path, first choice = SP {100 * c.agreement:.1f} %")
