# -*- coding: utf-8 -*-
"""4c. Does the state carry information about crowding? (the open question of decision D3 of Step 1)

    python density_states.py [CASE] [SIMS]        (default kochi_area2 10; a few minutes)

The state of an agent is its node and the density code (0, 1, 2) of each link around it. `densityLevel="link"` (the 2021 code) codes the
whole link by its population over 2 m x length, which on a few hundred agents almost never leaves 0; `"segment"` (2024) codes the worst
2 m segment with the real width. A tabular learner can only tell crowded from empty if the states differ, so this counts, after SIMS
training simulations of Q-learning (the study's schedule), the states that exist and the share of the decisions made in a state with a
crowded link around.
"""
import sys

import numpy as np
from common import REPO

from evacrl.experiment import Case, calibrate
from evacrl.options import ModelOptions

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi_area2"
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

if __name__ == "__main__":
    case = Case.load(CASE)
    print(f"{CASE}: Q-learning, {SIMS} training simulations, 30 min, mean departure 5 min\n")
    print(f"{'densityLevel':14s} {'nodes':>6s} {'states':>7s} {'visits':>9s} {'in a state with a crowded link':>32s} {'greedy safe':>12s}")
    for level in ("link", "segment"):
        res = calibrate(case, method="qlearning", sims=SIMS, eval_every=SIMS, eval_runs=3, seed=0, options=ModelOptions(densityLevel=level), eval_learn=True)
        state = res.final_state
        nodes = int(state[:, 0].max()) + 1
        visits = state[:, 21:31].sum(axis=1)
        crowded = (state[:, 1:11] > 0).any(axis=1)
        print(f"{level:14s} {nodes:6d} {state.shape[0]:7d} {int(visits.sum()):9d} {100 * visits[crowded].sum() / visits.sum():31.2f}% {res.best_eval:12.1f}")
