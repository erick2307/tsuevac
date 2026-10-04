# -*- coding: utf-8 -*-
"""Differential audit of evacrl.core against EVACMODEL3_FocalPoints/SARSA2024.py (Urushibara, read-only).

Run from this folder (python harness.py) with numpy, scipy and opencv-python installed. Everything runs on the
committed Urushibara case kochi2 (309 nodes, 622 agents) with a nextnode.csv,
which tsuevac can now read directly (evacrl.tables). Nothing is written into the Urushibara repository.
"""
import importlib.util
import os
import sys
import time

sys.dont_write_bytecode = True          # do not drop __pycache__ into the read-only reference repo
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

# URUSHIBARA_DIR: the EVACMODEL3_FocalPoints folder of erick2307/2024_urushibara (read-only reference)
URU = os.environ.get("URUSHIBARA_DIR", os.path.expanduser("~/2024_Urushibara/EVACMODEL3_FocalPoints"))
TSU = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, f"{TSU}/src")
from evacrl.options import ModelOptions                      # noqa: E402
from evacrl.sarsa import SARSA as TsuSARSA                   # noqa: E402

CASES = f"{URU}/results_for Usama"


def uru_class():
    spec = importlib.util.spec_from_file_location("SARSA2024", f"{URU}/SARSA2024.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SARSA


def files(case="kochi2", popfile=1):
    d = f"{CASES}/{case}"
    return dict(agentsProfileName=f"{d}/population_{popfile}.csv", nodesdbFile=f"{d}/nodes.csv",
                linksdbFile=f"{d}/edges.csv", transLinkdbFile=f"{d}/actionsdb.csv",
                transNodedbFile=f"{d}/transitionsdb.csv"), f"{d}/nextnode.csv"


def make(kind, case="kochi2", meanray=5, options=None, cls=None):
    """kind: 'uru' (SARSA2024) or 'tsu' (evacrl, with `options`: a ModelOptions)."""
    f, _ = files(case)
    if kind == "uru":
        return uru_class()(**f, meanRayleigh=meanray * 60)
    return (cls or TsuSARSA)(**f, meanRayleigh=meanray * 60, options=options)


def arrived(m):
    """Agents standing on an evacuation node, independent of the flag in column 10."""
    return int(np.sum(np.isin(m.pedDB[:, 8], m.evacuationNodes)))


def run_sp(m, nextnode, T=120 * 60):
    m.loadShortestPathDB(nextnode)
    flag, node = [], []
    for t in range(int(min(m.pedDB[:, 9])), T):
        m.initEvacuationAtTime()
        m.stepForward()
        m.checkTargetShortestPath()
        if not t % 10:
            m.computePedHistDenVelAtLinks()
            m.updateVelocityAllPedestrians()
        flag.append(int(np.sum(m.pedDB[:, 10] == 1)))
        node.append(arrived(m))
    return np.array(flag), np.array(node)


def evac_time(curve, t0):
    """Seconds at which the last new evacuee arrived (as shortestpath.run of Urushibara computes it)."""
    final = curve[-1]
    return t0 + int(np.argmax(curve >= final))


def run_rl(m, T, eps):
    """One learning/evaluation simulation as in calibration.run: eps = probability of a random choice."""
    for t in range(int(min(m.pedDB[:, 9])), T):
        m.initEvacuationAtTime()
        m.stepForward()
        opt = bool(np.random.choice(2, p=[eps, 1.0 - eps]))
        m.checkTarget(ifOptChoice=opt)
        if not t % 10:
            m.computePedHistDenVelAtLinks()
            m.updateVelocityAllPedestrians()
    return int(np.sum(m.pedDB[:, 10] == 1))


if __name__ == "__main__":
    f, nn = files()
    np.random.seed(1)
    t0 = time.time()
    m = make("uru")
    print("uru built", round(time.time() - t0, 2), "s; agents", m.numPedestrian, "nodes", m.nodesdb.shape[0])
    t0 = time.time()
    flag, node = run_sp(m, nn)
    print("uru SP 7200 s sim:", round(time.time() - t0, 1), "s wall; evacuated", flag[-1], "by node", node[-1])
    np.random.seed(1)
    t0 = time.time()
    m2 = make("tsu", options=ModelOptions.kochi2024())
    flag2, node2 = run_sp(m2, nn)
    print("tsu(kochi2024) SP:", round(time.time() - t0, 1), "s wall; evacuated", flag2[-1], "by node", node2[-1])
    print("identical curves:", np.array_equal(flag, flag2))
