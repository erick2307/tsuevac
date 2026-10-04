# -*- coding: utf-8 -*-
"""Step 2 audit: SARSA vs Q-learning, and Monte Carlo with its two discounting rules, on kochi2.

    python learning.py E1_SIMS E2_SIMS NSEEDS     (e.g. 100 50 3; about 25 min on 4 cores)

Case: `kochi2` of erick2307/2024_urushibara (309 nodes, 622 agents, 4 shelters), mean departure 5 min, 30 min simulated.
Training follows EVACMODEL3_FocalPoints/calibration.py: every simulation starts from the state matrix of the previous
one, and the probability of a random choice falls as 1 / (s / N + 1), i.e. from 1 to 0.5. After training the policy is
evaluated greedily (no random choices) on 5 fresh simulations: the survivors at 30 min of those are the measure.

E1: SARSA and Q-learning, the default ModelOptions (discounting once per decision, as both always did)
E2: Monte Carlo, discounting "second" (what it always did) and "decision"
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
URU = os.environ.get("URUSHIBARA_DIR", os.path.expanduser("~/2024_Urushibara/EVACMODEL3_FocalPoints"))
CASE = f"{URU}/results_for Usama/kochi2"

from evacrl.mc import MonteCarlo          # noqa: E402
from evacrl.options import ModelOptions   # noqa: E402
from evacrl.qlearn import QLearning       # noqa: E402
from evacrl.sarsa import SARSA            # noqa: E402

T = 30 * 60
CLASSES = {"sarsa": SARSA, "qlearning": QLearning, "mc": MonteCarlo}


def make(method, discounting="method"):
    return CLASSES[method](
        agentsProfileName=f"{CASE}/population_1.csv", nodesdbFile=f"{CASE}/nodes.csv", linksdbFile=f"{CASE}/edges.csv",
        transLinkdbFile=f"{CASE}/actionsdb.csv", transNodedbFile=f"{CASE}/transitionsdb.csv", meanRayleigh=5 * 60,
        options=ModelOptions(discounting=discounting, discount=0.9))   # the settings of the study (not the defaults any more)


def simulate(m, eps):
    for t in range(int(min(m.pedDB[:, 9])), T):
        m.initEvacuationAtTime()
        m.stepForward()
        m.checkTarget(ifOptChoice=bool(np.random.choice(2, p=[eps, 1.0 - eps])))
        if not t % 10:
            m.computePedHistDenVelAtLinks()
            m.updateVelocityAllPedestrians()
    if isinstance(m, MonteCarlo):
        m.updateValueFunctionDB()          # Monte Carlo learns at the end of the simulation
    return m.getNumberEvacuatedPed()


def job(args):
    method, discounting, seed, nsims = args
    np.random.seed(seed)
    t0, state, curve = time.time(), None, []
    for s in range(nsims):
        m = make(method, discounting)
        if state is not None:
            m.stateMat = state
        curve.append(simulate(m, 1.0 / (s / nsims + 1.0)))
        state = m.stateMat
    greedy = []
    for _ in range(5):
        m = make(method, discounting)
        m.stateMat = state.copy()
        greedy.append(simulate(m, 0.0))
        m.stateMat = None
    return dict(method=method, discounting=discounting, seed=seed, nsims=nsims, curve=curve, greedy=greedy,
                states=int(state.shape[0]), secs=round(time.time() - t0))


if __name__ == "__main__":
    e1, e2, nseeds = (int(v) for v in sys.argv[1:4])
    jobs = [(m, "method", s, e1) for s in range(nseeds) for m in ("sarsa", "qlearning")]
    jobs += [("mc", d, s, e2) for s in range(nseeds) for d in ("second", "decision")]
    with Pool(4) as pool:
        out = pool.map(job, jobs, chunksize=1)
    json.dump(out, open(os.path.join(HERE, "learning.json"), "w"))
    print("done", len(out), "jobs")
