# -*- coding: utf-8 -*-
"""One simulation, and many of them: the building blocks of the experiments.

`run_episode` is the loop every entry script of the repository repeats (departures, steps, target checks, the 10 s refresh of
densities and speeds). `shortest_path_run` and `evaluate_state` are fixed-seed runs that return a `RunResult`; the functions that
take `workers` run their runs in a process pool and give the same results for any number of workers (the seed of a run is
derived from its index, not from the order in which a worker gets to it).
"""
import multiprocessing
from dataclasses import dataclass

import numpy as np

from evacrl.mc import MonteCarlo
from evacrl.options import ModelOptions
from evacrl.qlearn import QLearning
from evacrl.sarsa import SARSA

METHODS = {"sarsa": SARSA, "qlearning": QLearning, "mc": MonteCarlo}


@dataclass
class RunResult:
    """The outcome of one simulation. `curve[i]` is the number of agents safe that the loop records at its second `t = t0 + i`, after
    that second has been simulated (the `time, safe` rows of the 2024 study's results; the engine's clock reads `t + 1` then). So
    `safe_at(s)`, the agents safe after `s` seconds, is `curve[s - t0 - 1]`, and `last_evacuee`, the study's evacuation time, is the
    recorded second at which the final count appears: `safe_at(last_evacuee + 1)` is the whole count. Agents that start at a shelter
    count when they depart, as in the engine."""
    seed: int
    agents: int
    t0: int
    curve: np.ndarray

    @property
    def evacuated(self):
        return int(self.curve[-1]) if len(self.curve) else 0

    def safe_at(self, seconds):
        """Agents safe after `seconds` of simulated time (the whole simulation if it is shorter)."""
        i = int(seconds) - self.t0 - 1
        if i < 0 or len(self.curve) == 0:
            return 0
        return int(self.curve[min(i, len(self.curve) - 1)])

    @property
    def last_evacuee(self):
        """The second at which the last agent that got safe did so (the study's evacuation time); NaN if nobody did."""
        if not len(self.curve) or self.curve[-1] == 0:
            return float("nan")
        return float(self.t0 + int(np.argmax(self.curve >= self.curve[-1])))


def make_model(case, method="sarsa", options=None, mean_departure=5.0, discount=None):
    """A model of `case` (`Case`), `method` one of `METHODS`, departures Rayleigh-distributed with a mean of `mean_departure` minutes. `discount`: None takes the discount of the options."""
    return METHODS[method](meanRayleigh=mean_departure * 60, discount=discount,
                           options=ModelOptions() if options is None else options, **case.kwargs)


def run_episode(model, sim_time, epsilon=0.0, shortest_path=False, learn=True):
    """Run `model` for `sim_time` seconds and return the number of safe agents recorded at each second from the first departure.

    shortest_path  agents follow the table loaded by `loadShortestPathDB`
    epsilon        else each second all arriving agents choose at random with this probability, and by their action values otherwise
    learn          False: the action values are not updated (the temporal-difference update is switched off, and a Monte Carlo model
                   does not learn at the end): the policy that is in the state matrix is what is run. A state the policy has not met
                   is added with the initial values (all equal), as always. True: the model learns as it goes (temporal-difference)
                   or at the end (Monte Carlo): evaluate on a copy of the state matrix to leave it as it was.
    """
    if not learn:
        model.tdControl = lambda *args, **kwargs: None
    curve = []
    for t in range(int(min(model.pedDB[:, 9])), int(sim_time)):
        model.initEvacuationAtTime()
        model.stepForward()
        if shortest_path:
            model.checkTargetShortestPath()
        else:
            explore = epsilon > 0 and bool(np.random.choice(2, p=[1.0 - epsilon, epsilon]))
            model.checkTarget(ifOptChoice=not explore)
        if not t % 10:
            model.computePedHistDenVelAtLinks()
            model.updateVelocityAllPedestrians()
        curve.append(model.getNumberEvacuatedPed())
    if learn and isinstance(model, MonteCarlo):
        model.updateValueFunctionDB()
    return np.array(curve, dtype=np.int64)


def _first_second(model):
    return int(min(model.pedDB[:, 9]))


def shortest_path_run(case, seed, options=None, sim_time=7200, mean_departure=5.0):
    """One shortest-path run (`case.nextnode` is needed). `seed` fixes the departure times."""
    if case.nextnode is None:
        raise ValueError(f"the case {case.name!r} has no nextnode.csv: build one with evacrl.casebuild")
    np.random.seed(seed)
    model = make_model(case, "sarsa", options, mean_departure)
    model.loadShortestPathDB(case.nextnode)
    curve = run_episode(model, sim_time, shortest_path=True)
    return RunResult(seed, model.numPedestrian, _first_second(model), curve)


def check_state(model, state):
    """`state` must be the state matrix of this case: 31 columns, and its first rows the states of the nodes 0, 1, 2, ..."""
    state = np.asarray(state)
    n = model.nodesdb.shape[0]
    if state.ndim != 2 or state.shape[1] != 31 or state.shape[0] < n or not np.array_equal(state[:n, 0], np.arange(n)):
        raise ValueError(f"not a state matrix of this case ({n} nodes): shape {state.shape}, first rows {state[:3, 0].tolist() if state.ndim == 2 else '?'}")


def evaluate_state(case, method, state, seed, options=None, sim_time=1800, mean_departure=5.0, discount=None, learn=False):
    """One greedy run (no random choices) of the policy in the state matrix `state`, which is left unchanged.

    learn=False (default): the policy is frozen, nothing is learned during the run. learn=True: the agents keep learning on a copy of
    the state (what the 2024 calibration and the audits of Steps 2-4 did), so the result is that of a policy that adapts as it goes."""
    np.random.seed(seed)
    model = make_model(case, method, options, mean_departure, discount)
    check_state(model, state)
    model.stateMat = np.array(state, copy=True)
    curve = run_episode(model, sim_time, epsilon=0.0, learn=learn)
    return RunResult(seed, model.numPedestrian, _first_second(model), curve)


def _call(job):
    function, args = job
    return function(*args)


def map_jobs(function, argument_lists, workers=1):
    """`[function(*args) for args in argument_lists]`, in a pool of `workers` processes if more than 1; the order is kept."""
    jobs = [(function, args) for args in argument_lists]
    if workers is None or workers <= 1 or len(jobs) <= 1:
        return [_call(job) for job in jobs]
    with multiprocessing.get_context().Pool(min(workers, len(jobs))) as pool:
        return pool.map(_call, jobs, chunksize=1)
