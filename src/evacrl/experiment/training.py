# -*- coding: utf-8 -*-
"""Training with a measure of the policy that is not the training noise.

The 2024 study trained with a decaying random-choice rate and kept the simulation with the most survivors as the policy to build
on. Those survivors come from runs that were still choosing at random 50-100 % of the time, so the "best" simulation is the luckiest
one, not the best policy. Here every `eval_every` simulations the policy at that point is run *greedily* (no random choices) on the
same `eval_runs` departure-time seeds each time, and the best of those checkpoints is the result. The training episodes continue
from the latest state (or, with `restart_from_best`, go back to the best checkpoint after a checkpoint that did not improve on it).
The evaluation runs the policy frozen: nothing is learned during an evaluation run (`eval_learn=True` lets the agents go on learning on a
copy, as the 2024 calibration did).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from evacrl.experiment.runs import evaluate_state, make_model, map_jobs, run_episode
from evacrl.experiment.seeds import derive_seeds, seeded

SCHEDULES = ("calibration", "quadratic", "constant")


def epsilon_at(schedule, s, n, constant=0.5):
    """Probability of a random choice in training simulation `s` of `n`.

    "calibration"  1 / (s / n + 1): from 1 to 0.5, the schedule of EVACMODEL3_FocalPoints/calibration.py
    "quadratic"    1 - (s / (0.8 n))**2 until 80 % of the simulations, then 0 (scripts/main_ql_mod.py)
    "constant"     `constant`"""
    if schedule == "calibration":
        return 1.0 / (s / n + 1.0)
    if schedule == "quadratic":
        end = 0.8 * n
        return 1.0 - (s / end) ** 2 if s < end else 0.0
    if schedule == "constant":
        return float(constant)
    raise ValueError(f"schedule must be one of {SCHEDULES}")


@dataclass
class Checkpoint:
    """After `sim` training simulations: the random-choice rate of the last one, the mean safe of the exploring runs since the
    previous checkpoint, and the greedy evaluation of the policy (mean and sd over the evaluation seeds)."""
    sim: int
    epsilon: float
    train_safe: float
    eval_mean: float
    eval_sd: float
    best: bool


@dataclass
class CalibrationResult:
    history: List[Checkpoint]
    best_sim: int
    best_eval: float
    best_state: np.ndarray
    final_state: np.ndarray
    agents: int
    seeds: dict = field(default_factory=dict)
    training_safe: List[int] = field(default_factory=list)   # safe at the end of every training simulation (the exploring runs)


def calibrate(case, *, method="qlearning", sims=300, schedule="calibration", eval_every=25, eval_runs=5, sim_time=1800,
              mean_departure=5.0, options=None, discount=None, seed=0, workers=1, restart_from_best=False, eval_learn=False, progress=None):
    """Train `method` on `case` for `sims` simulations; see the module docstring. Training is sequential, the evaluations of
    a checkpoint run in `workers` processes. `progress(checkpoint)` is called after each checkpoint. Reproducible from `seed`.

    restart_from_best  after a checkpoint that is not better than the best so far, the next training episode starts from a copy of the
                       best checkpoint instead of the latest state (`final_state` is still the state at the end of training)"""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if eval_every < 1 or eval_runs < 1 or sims < 1:
        raise ValueError("sims, eval_every and eval_runs must be at least 1")
    train_seeds = derive_seeds(seed, sims, "training")
    eval_seeds = derive_seeds(seed, eval_runs, "evaluation")
    state, best_state, best_eval, best_sim = None, None, -np.inf, 0
    history, exploring, agents, training_safe = [], [], 0, []
    go_back = False
    for s in range(sims):
        with seeded(train_seeds[s]):
            model = make_model(case, method, options, mean_departure, discount)
            agents = model.numPedestrian
            if go_back:
                model.stateMat = np.array(best_state, copy=True)   # the episode changes its state matrix in place: not the stored best
                go_back = False
            elif state is not None:
                model.stateMat = state
            eps = epsilon_at(schedule, s, sims)
            curve = run_episode(model, sim_time, epsilon=eps)
        exploring.append(int(curve[-1]) if len(curve) else 0)
        training_safe.append(exploring[-1])
        state = model.stateMat
        if (s + 1) % eval_every == 0 or s + 1 == sims:
            runs = map_jobs(evaluate_state, [(case, method, state, es, options, sim_time, mean_departure, discount, eval_learn) for es in eval_seeds], workers)
            safe = np.array([r.evacuated for r in runs], dtype=float)
            better = safe.mean() > best_eval
            if better:
                best_eval, best_sim, best_state = float(safe.mean()), s + 1, np.array(state, copy=True)
            check = Checkpoint(s + 1, eps, float(np.mean(exploring)), float(safe.mean()), float(safe.std(ddof=1)) if len(safe) > 1 else float("nan"), better)
            history.append(check)
            exploring = []
            go_back = restart_from_best and not better
            if progress is not None:
                progress(check)
    return CalibrationResult(history, best_sim, best_eval, best_state, np.array(state, copy=True), agents,
                             dict(base=seed, training=train_seeds, evaluation=eval_seeds), training_safe)


def evaluate_policy(case, method, state, runs=20, *, seed=0, workers=1, options=None, sim_time=1800, mean_departure=5.0, discount=None, learn=False):
    """`runs` greedy runs of the policy in `state`, departure-time seeds derived from `seed`: a list of `RunResult`. The policy is frozen
    unless `learn` (see `evaluate_state`)."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    seeds = derive_seeds(seed, runs, "policy")
    return map_jobs(evaluate_state, [(case, method, state, s, options, sim_time, mean_departure, discount, learn) for s in seeds], workers)
