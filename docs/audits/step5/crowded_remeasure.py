# -*- coding: utf-8 -*-
"""The shortest-path figures of the D9 audit (docs/audits/step4) on `kochi_area4`, measured again with the tie-break of 0.2.0
(Step 5, finding 1: 13 of the 1,110 entries of the table changed, each between two equally short steps).

  1. 10 shortest-path runs of 30 min, seed 3 (`crowded_area.py`)
  2. 5 runs of 120 min, seed 5 (`crowded_full_evacuation.sh`)
  3. the stored learned policies against the shortest path: walk length, first choice (`compare_with_shortest_path`)

    python crowded_remeasure.py [workers]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from evacrl.experiment import Case, compare_with_shortest_path, repeat_shortest_path  # noqa: E402

workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
case = Case.load("kochi_area4")

res = repeat_shortest_path(case, 10, seed=3, workers=workers, sim_time=1800, horizon=1800, batch=5)
safe = np.array([r.safe_at(1800) for r in res.runs], dtype=float)
print(f"1. 30 min, 10 runs, seed 3: safe at 30 min {safe.mean():.1f} +- {safe.std(ddof=1):.1f} of {res.runs[0].agents} ({100 * safe.mean() / res.runs[0].agents:.1f} %)")

res = repeat_shortest_path(case, 5, seed=5, workers=workers, sim_time=7200, horizon=3600, batch=5)
at30 = np.array([r.safe_at(1800) for r in res.runs], dtype=float)
at60 = np.array([r.safe_at(3600) for r in res.runs], dtype=float)
last = np.array([r.last_evacuee for r in res.runs], dtype=float)
final = np.array([r.curve[-1] for r in res.runs], dtype=float)
print(f"2. 120 min, 5 runs, seed 5: safe at 30 min {at30.mean():.1f}, at 60 min {at60.mean():.1f}, at 120 min {final.mean():.1f} "
      f"(all {res.runs[0].agents}: {bool((final == res.runs[0].agents).all())}), last evacuee {last.mean():.0f} +- {last.std(ddof=1):.0f} s")

for level in ("link", "segment"):
    state = np.loadtxt(os.path.join(HERE, "..", "step4", f"state_kochi_area4_{level}.csv"), delimiter=",")
    c = compare_with_shortest_path(case, state)
    print(f"3. {level}: walk against shortest path {c.longer:+.1f} %, first choice = shortest path's {100 * c.agreement:.0f} %, "
          f"never arrive {c.never_arrive}")
