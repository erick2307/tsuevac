# -*- coding: utf-8 -*-
"""Summarise learning.json (see learning.py)."""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "learning.json" if len(sys.argv) < 2 else sys.argv[1])))
SHORTEST_PATH_AT_30_MIN = 529.2     # Step 1: shortest path on kochi2, 30 seeds, same settings (docs/audits/step1)
N = 622


def group(method, discounting):
    return [r for r in rows if r["method"] == method and r["discounting"] == discounting]


def summary(label, g):
    greedy = np.array([r["greedy"] for r in g])        # seeds x 5 evaluations
    seed_means = greedy.mean(1)
    curve = np.array([r["curve"] for r in g]).mean(0)  # mean over seeds, by training simulation
    n = len(curve)
    w = max(n // 10, 1)
    print(f"{label:28s} greedy survivors at 30 min {greedy.mean():6.1f} ({100 * greedy.mean() / N:4.1f}% of {N}; "
          f"{100 * greedy.mean() / SHORTEST_PATH_AT_30_MIN:4.1f}% of shortest path)  seeds {np.round(seed_means).astype(int).tolist()}"
          f"  | training survivors, first {w} sims {curve[:w].mean():5.0f} .. last {w} {curve[-w:].mean():5.0f}"
          f"  | states {np.mean([r['states'] for r in g]):.0f}")
    return greedy


def compare(a_label, a, b_label, b):
    """Welch test on the per-seed means: the 5 evaluations of a seed share one learned policy, so the seeds, not the
    evaluations, are the replicates."""
    am, bm = a.mean(1), b.mean(1)
    t = stats.ttest_ind(am, bm, equal_var=False)
    print(f"   {a_label} - {b_label}: {a.mean() - b.mean():+6.1f} survivors (Welch p = {t.pvalue:.3g} on the per-seed means, "
          f"{len(am)} seeds each)")


e1 = [r for r in rows if r["method"] in ("sarsa", "qlearning")]
e2 = [r for r in rows if r["method"] == "mc"]
if e1:
    print(f"E1: {e1[0]['nsims']} training simulations\n")
    s = summary("SARSA", group("sarsa", "method"))
    q = summary("Q-learning", group("qlearning", "method"))
    compare("Q-learning", q, "SARSA", s)
if e2:
    print(f"\nE2: Monte Carlo, {e2[0]['nsims']} training simulations\n")
    a = summary("Monte Carlo, per second", group("mc", "second"))
    b = summary("Monte Carlo, per decision", group("mc", "decision"))
    compare("per decision", b, "per second", a)
    if e1:
        print("\nReference after the same number of simulations is not available for E1 (it ran longer); see the curves in learning.json.")
