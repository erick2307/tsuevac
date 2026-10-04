# -*- coding: utf-8 -*-
"""Repeated shortest-path runs: the baseline distribution that a learned policy is compared with."""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from evacrl.experiment.convergence import Convergence, convergence_trace
from evacrl.experiment.runs import RunResult, map_jobs, shortest_path_run
from evacrl.experiment.seeds import derive_seeds


@dataclass
class ShortestPathResult:
    runs: List[RunResult]
    converged: bool
    trace: dict = field(default_factory=dict)   # metric -> [Convergence per batch]
    base_seed: int = 0

    def metric(self, name, horizon=1800):
        if name == "last_evacuee":
            return np.array([r.last_evacuee for r in self.runs])
        if name == "safe":
            return np.array([r.safe_at(horizon) for r in self.runs], dtype=float)
        raise ValueError("metric must be 'last_evacuee' or 'safe'")

    @property
    def seeds(self):
        return [r.seed for r in self.runs]

    @property
    def incomplete(self):
        """Runs that ended with agents still on their way: their `last_evacuee` is the last arrival within the simulated time, not the
        time everybody was safe, so it understates the evacuation time (the study's evacuation time has the same limit)."""
        return int(sum(r.evacuated < r.agents for r in self.runs))


def repeat_shortest_path(case, runs=None, *, until_converged=False, max_runs=1000, min_runs=30, batch=10, tol=0.01, horizon=1800,
                         seed=0, workers=1, options=None, sim_time=7200, mean_departure=5.0):
    """Shortest-path runs with seeds derived from `seed`, in batches of `batch` (at least `min_runs`).

    runs            a fixed number of runs, or
    until_converged keep adding batches until the evacuation time (`last_evacuee`) and the number safe at `horizon` seconds both
                    have a relative standard error and a relative change of their running mean under `tol` (see `Convergence`),
                    or `max_runs` is reached. The decision is taken after whole batches, in seed order, so it does not depend
                    on `workers`.
    Returns a `ShortestPathResult`; `converged` is False for a fixed number of runs unless the rule happens to be met."""
    if runs is None and not until_converged:
        raise ValueError("give a number of runs, or until_converged=True")
    if (max_runs if until_converged else runs) < 1 or batch < 1:
        raise ValueError("the number of runs and the batch size must be at least 1")
    target = max_runs if until_converged else int(runs)
    seeds = derive_seeds(seed, target, "shortest_path")
    done, results = 0, []
    converged, trace = False, {}
    while done < target:
        step = min(batch, target - done)
        results += map_jobs(shortest_path_run, [(case, s, options, sim_time, mean_departure) for s in seeds[done:done + step]], workers)
        done += step
        out = ShortestPathResult(results, False, {}, seed)
        trace = {m: convergence_trace(out.metric(m, horizon), batch, min_runs, tol) for m in ("last_evacuee", "safe")}
        converged = all(t[-1].converged for t in trace.values())
        if until_converged and converged:
            break
    return ShortestPathResult(results, converged, trace, seed)
