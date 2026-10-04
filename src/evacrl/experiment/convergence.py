# -*- coding: utf-8 -*-
"""When are enough repeated runs enough? The running mean of a metric settles, and its coefficient of variation (sd / mean)
stops moving. The 2024 study ran 1,000 shortest-path simulations and looked at the running CV by eye; this makes it a rule."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Convergence:
    """What was found after `n` runs of one metric: `mean`, `sd`, `cv` (sd / mean), `sem_rel` (standard error of the mean over the
    mean) and `change` (relative change of the mean over the last batch). `converged`: at least `min_runs` runs with a value, sem_rel < tol and change < tol."""
    n: int
    mean: float
    sd: float
    cv: float
    sem_rel: float
    change: float
    converged: bool


def convergence_trace(values, batch=10, min_runs=30, tol=0.01):
    """The `Convergence` of the first `batch`, `2 * batch`, ... values (and of all of them): a list, one entry per batch.

    A metric with mean 0 never converges (its relative statistics are undefined)."""
    values = np.asarray(values, dtype=float)
    trace, previous = [], None
    sizes = list(range(batch, len(values) + 1, batch))
    if not sizes or sizes[-1] != len(values):
        sizes.append(len(values))
    for n in sizes:
        v = values[:n]
        v = v[~np.isnan(v)]
        if len(v) == 0:                                    # nothing to measure yet (for instance, nobody was evacuated)
            trace.append(Convergence(n, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False))
            previous = None
            continue
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        ok = mean != 0 and np.isfinite(sd)
        cv = sd / mean if ok else float("nan")
        sem = cv / np.sqrt(len(v)) if ok else float("nan")
        change = abs(mean - previous) / abs(mean) if (previous is not None and mean != 0) else float("nan")
        converged = bool(ok and len(v) >= min_runs and abs(sem) < tol and np.isfinite(change) and change < tol)
        trace.append(Convergence(n, mean, sd, cv, float(sem), float(change), converged))
        previous = mean
    return trace
