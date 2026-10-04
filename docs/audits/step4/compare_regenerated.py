# -*- coding: utf-8 -*-
"""D8. The 2024 study's committed shortest-path results against the regenerated ones (results/kochi2024_regenerated, see regenerate_2024.sh).

    python compare_regenerated.py        (needs scipy and the 2024 repository)

For each area: the study's 1,000 runs (evacuation time `results_time.csv`; the safe agents at 30 min and at the end from `results_df.csv`) and the
regenerated runs with the default options, with how many ended with agents who never got safe within 120 min.
"""
import csv
import os

import numpy as np
from common import REPO, URU

STUDY = f"{URU}/results_for Usama"
NEW = os.path.join(REPO, "results", "kochi2024_regenerated")
AREAS = ("kochi0", "kochi1", "kochi2", "kochi4", "kochi42")


def committed(area):
    times = np.loadtxt(f"{STUDY}/{area}/results_time.csv", delimiter=",", skiprows=1)
    data = np.loadtxt(f"{STUDY}/{area}/results_df.csv", delimiter=",", skiprows=1)
    starts = np.r_[0, np.where(np.diff(data[:, 0]) < 0)[0] + 1, len(data)]
    safe30, final = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        run = data[a:b]
        safe30.append(run[run[:, 0] == 1799, 1][0])
        final.append(run[-1, 1])
    agents = int(np.loadtxt(f"{STUDY}/{area}/population_1.csv", delimiter=",", skiprows=1).shape[0])
    return times, np.array(safe30), np.array(final), agents


def regenerated(area):
    with open(os.path.join(NEW, area, "runs.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = np.array([float(r["last_evacuee_s"]) if r["last_evacuee_s"] else np.nan for r in rows])
    safe30 = np.array([float(r["safe_at_1800s"]) for r in rows])
    left = np.array([int(r["evacuated"]) < int(r["agents"]) for r in rows])
    with open(os.path.join(NEW, area, "convergence.csv"), encoding="utf-8") as f:
        conv = [r for r in csv.DictReader(f)]
    converged = all(int([r for r in conv if r["metric"] == m][-1]["converged"]) for m in ("last_evacuee", "safe"))
    sem = {m: float([r for r in conv if r["metric"] == m][-1]["sem_rel"]) for m in ("last_evacuee", "safe")}
    return last, safe30, left, converged, sem, int(rows[0]["agents"])


if __name__ == "__main__":
    from scipy.stats import ks_2samp
    print(f"{'area':7s} {'':12s} {'runs':>5s} {'agents':>6s} {'evacuation time, s: mean ± sd':>31s} {'median':>7s} {'p95':>7s} {'safe at 30 min':>16s} {'agents left at 120 min':>23s}")
    for area in AREAS:
        if not os.path.exists(os.path.join(NEW, area, "runs.csv")):
            continue
        t, s30, final, n = committed(area)
        last, ns30, left, converged, sem, n2 = regenerated(area)
        print(f"{area:7s} {'committed':12s} {len(t):5d} {n:6d} {t.mean():20.0f} ± {t.std(ddof=1):6.0f} {np.median(t):7.0f} {np.percentile(t, 95):7.0f} "
              f"{s30.mean():9.1f} ± {s30.std(ddof=1):4.1f} {int(np.sum(final < n)):10d} of {len(t)} ({100 * np.mean(final < n):.0f} %)")
        ok = ~np.isnan(last)
        print(f"{'':7s} {'regenerated':12s} {len(last):5d} {n2:6d} {last[ok].mean():20.0f} ± {last[ok].std(ddof=1):6.0f} {np.median(last[ok]):7.0f} {np.percentile(last[ok], 95):7.0f} "
              f"{ns30.mean():9.1f} ± {ns30.std(ddof=1):4.1f} {int(left.sum()):10d} of {len(last)} ({100 * np.mean(left):.0f} %)   "
              f"converged: {converged} (standard errors {100 * sem['last_evacuee']:.2f} % and {100 * sem['safe']:.2f} %)")
        print(f"{'':7s} {'KS p':12s} evacuation time {ks_2samp(last[ok], t).pvalue:.3g}, safe at 30 min {ks_2samp(ns30, s30).pvalue:.3g}")
