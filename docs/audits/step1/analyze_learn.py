# -*- coding: utf-8 -*-
"""Summarise learn_50.json (battery.py learn 50 3). The unit of replication is the training run (seed): the 5 greedy
evaluations of one seed share one learned policy, so tests are done on the per-seed means, not on all evaluations."""
import json
import sys

import numpy as np
from scipy import stats

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "learn_50.json"))
cfgs = list(dict.fromkeys(r["cfg"] for r in rows))
seed_means = {c: np.array([np.mean(r["greedy"]) for r in rows if r["cfg"] == c]) for c in cfgs}
base = seed_means["base"]
print("kochi2, SARSA, 50 training simulations; greedy survivors at 30 min; the tests compare per-seed means (3 vs 3)\n")
print(f"{'config':16s} {'mean':>6s} {'per seed':>16s} {'states':>7s}   vs base: difference, Welch p")
for c in cfgs:
    m = seed_means[c]
    t = stats.ttest_ind(m, base, equal_var=False)
    states = np.mean([r["nstates"] for r in rows if r["cfg"] == c])
    print(f"{c:16s} {m.mean():6.1f} {str(np.round(m).astype(int).tolist()):>16s} {states:7.0f}   {m.mean() - base.mean():+6.1f}, p = {t.pvalue:.2f}")
