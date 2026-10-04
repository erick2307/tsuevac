# -*- coding: utf-8 -*-
"""Writing and reading the results of the experiments: CSV tables and the evacuation curves."""
import csv
import os

import numpy as np

from evacrl.experiment.runs import RunResult


def curves_matrix(results, sim_time):
    """`(time, safe)`: the seconds `0 .. sim_time - 1` and the agents safe that each run records at each of them (`RunResult.curve`; 0
    before its first departure, the last value after its end)."""
    time = np.arange(int(sim_time))
    safe = np.zeros((len(results), len(time)), dtype=np.int64)
    for i, r in enumerate(results):
        n = min(len(r.curve), len(time) - r.t0)
        safe[i, r.t0:r.t0 + n] = r.curve[:n]
        safe[i, r.t0 + n:] = r.curve[n - 1] if n > 0 else 0
    return time, safe


def write_runs(folder, results, horizon=1800):
    """`runs.csv` (one row per run) and `curves.npz` (the whole curves)."""
    if not results:
        raise ValueError("no runs to write")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "agents", f"safe_at_{int(horizon)}s", "evacuated", "last_evacuee_s"])
        for i, r in enumerate(results):
            w.writerow([i, r.seed, r.agents, r.safe_at(horizon), r.evacuated, "" if np.isnan(r.last_evacuee) else int(r.last_evacuee)])
    sim_time = max(r.t0 + len(r.curve) for r in results)
    time, safe = curves_matrix(results, sim_time)
    np.savez_compressed(os.path.join(folder, "curves.npz"), time=time, safe=safe, agents=np.array([r.agents for r in results]),
                        seeds=np.array([r.seed for r in results]))


def read_curves(folder):
    """`(time, safe, agents)` written by `write_runs`."""
    with np.load(os.path.join(folder, "curves.npz")) as z:
        return z["time"], z["safe"], z["agents"]


def write_convergence(folder, trace):
    with open(os.path.join(folder, "convergence.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "sd", "cv", "sem_rel", "change", "converged"])
        for metric, rows in trace.items():
            for c in rows:
                w.writerow([metric, c.n, f"{c.mean:.6g}", f"{c.sd:.6g}", f"{c.cv:.6g}", f"{c.sem_rel:.6g}", f"{c.change:.6g}", int(c.converged)])


def write_history(folder, history):
    with open(os.path.join(folder, "calibration.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sim", "epsilon", "train_safe", "eval_mean", "eval_sd", "best"])
        for c in history:
            w.writerow([c.sim, f"{c.epsilon:.6f}", f"{c.train_safe:.3f}", f"{c.eval_mean:.3f}", f"{c.eval_sd:.3f}", int(c.best)])


def read_history(folder):
    """The rows of `calibration.csv` as a dict of arrays."""
    with open(os.path.join(folder, "calibration.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def write_state(path, state):
    """A state matrix in the format of `EvacuationModel.exportStateMatrix`."""
    fmt = ["%d"] * 11 + ["%.6f"] * 10 + ["%d"] * 10
    np.savetxt(path, state, delimiter=",", fmt=fmt)
