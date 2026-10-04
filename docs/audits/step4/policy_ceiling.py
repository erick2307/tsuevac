# -*- coding: utf-8 -*-
"""4a. What the learning target can reach: the best any learner could do under the reward the temporal-difference methods use.

    python policy_ceiling.py [CASE] [SEEDS]        (default kochi2, 20; a few minutes on 4 cores)

The methods learn Q(S, A) = E[ sum_k gamma^k * (-1 s) * dt_k  +  gamma^n * surviveReward ]: a reward of -1 per second, a lump sum
surviveReward (1e5) on arriving, and a discount `gamma` (0.9) applied ONCE PER DECISION, i.e. per node passed. With 20 nodes to a
shelter the lump sum is worth 1e5 * 0.9^20 = 12,158, and one node more or less changes it by about 1,200: as much as 1,200 s of
walking. So the target prefers fewer nodes over a shorter walk. This script solves that target exactly (value iteration on the
network, no learning) for several discounts, runs the resulting fixed policies in the simulator like the shortest-path baseline, and
compares them: it is the ceiling of what a perfect learner of that target would reach.
"""
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from common import case_files

from evacrl.options import ModelOptions
from evacrl.sarsa import SARSA
from evacrl.tables import load_table

CASE = sys.argv[1] if len(sys.argv) > 1 else "kochi2"
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
T = 30 * 60
SPEED = 1.19          # m/s, the free walking speed of the model
REWARD = 1e5
FILES, SP_FILE = case_files(CASE)
WORK = tempfile.mkdtemp(prefix="step4_ceiling_")


def graph():
    nodes = load_table(FILES["nodesdbFile"])
    links = load_table(FILES["linksdbFile"])
    n = len(nodes)
    best = {}
    for _, a, b, length, _ in links:
        a, b = int(a), int(b)
        if a == b:
            continue
        for u, v in ((a, b), (b, a)):
            best[(u, v)] = min(best.get((u, v), np.inf), float(length))      # the shortest of parallel links, as the baseline
    neighbours = [[] for _ in range(n)]
    for (u, v), length in best.items():
        neighbours[u].append((v, length))
    return nodes, neighbours


def solve(nodes, neighbours, gamma, mode):
    """Value iteration for the deterministic network. mode "decision": gamma once per node; "second": gamma ** seconds.
    Returns the next node of every node (nodes are shelters: themselves; no way out: -9999)."""
    n = len(nodes)
    shelter = nodes[:, 3] == 1
    V = np.full(n, -np.inf)
    V[shelter] = REWARD
    nxt = np.full(n, -9999)
    nxt[shelter] = np.where(shelter)[0]
    for _ in range(10 * n):
        changed = False
        for u in range(n):
            if shelter[u]:
                continue
            for v, length in neighbours[u]:
                if V[v] == -np.inf:
                    continue
                dt = length / SPEED
                g = gamma if mode == "decision" else gamma ** dt
                value = -dt + g * V[v]
                if value > V[u] + 1e-9:
                    V[u], nxt[u], changed = value, v, True
        if not changed:
            break
    return nxt


def write_policy(name, nxt):
    path = os.path.join(WORK, f"{name}.csv")
    np.savetxt(path, np.column_stack([np.arange(len(nxt)), nxt]), delimiter=",", fmt="%d", header="node,next node")
    return path


def walk(nodes, neighbours, nxt, start):
    """Metres and nodes passed from `start` to a shelter following `nxt`."""
    lengths = {(u, v): l for u in range(len(nodes)) for v, l in neighbours[u]}
    metres, hops, u = 0.0, 0, start
    while nodes[u, 3] != 1:
        v = int(nxt[u])
        metres += lengths[(u, v)]
        hops += 1
        u = v
        if hops > len(nodes):
            return np.nan, np.nan
    return metres, hops


def run(job):
    name, path, seed = job
    np.random.seed(seed)
    m = SARSA(meanRayleigh=5 * 60, options=ModelOptions(), **FILES)
    m.loadShortestPathDB(path)
    t0 = int(min(m.pedDB[:, 9]))
    curve = []
    for t in range(t0, 120 * 60):
        m.initEvacuationAtTime()
        m.stepForward()
        m.checkTargetShortestPath()
        if not t % 10:
            m.computePedHistDenVelAtLinks()
            m.updateVelocityAllPedestrians()
        curve.append(m.getNumberEvacuatedPed())
    curve = np.array(curve)
    return name, dict(agents=m.numPedestrian, end=int(curve[-1]), last=t0 + int(np.argmax(curve >= curve[-1])), at30=int(curve[T - t0 - 1]))


if __name__ == "__main__":
    nodes, neighbours = graph()
    agents = load_table(FILES["agentsProfileName"], dtype=int)[:, 4]
    policies = {"shortest path (metres)": SP_FILE}
    for gamma, mode in ((0.9, "decision"), (0.99, "decision"), (0.999, "decision"), (0.999, "second"), (0.9999, "second")):
        label = f"gamma {gamma} per {'node' if mode == 'decision' else 'second'}"
        policies[label] = write_policy(f"g_{mode}_{gamma}", solve(nodes, neighbours, gamma, mode))
    print(f"{CASE}: {len(nodes)} nodes, {len(agents)} agents, {int(nodes[:, 3].sum())} shelters\n")
    print("1  the walk each policy gives (metres and nodes passed, over the agents' start nodes)\n")
    print(f"{'policy':26s} {'metres mean':>12s} {'vs shortest':>12s} {'nodes mean':>11s} {'start nodes with a longer walk':>31s}")
    base = None
    for label, path in policies.items():
        nxt = load_table(path, dtype=int)[:, 1]
        res = np.array([walk(nodes, neighbours, nxt, int(s)) for s in agents])
        if base is None:
            base = res[:, 0]
        print(f"{label:26s} {np.nanmean(res[:, 0]):12.0f} {100 * (np.nanmean(res[:, 0]) / np.nanmean(base) - 1):+11.1f}% {np.nanmean(res[:, 1]):11.1f} "
              f"{100 * np.mean(res[:, 0] > base + 0.5):30.1f}%")
    jobs = [(label, path, s) for s in range(SEEDS) for label, path in policies.items()]
    with Pool(4) as pool:
        out = pool.map(run, jobs, chunksize=1)
    print(f"\n2  the fixed policies in the simulator ({SEEDS} departure-time seeds, mean departure 5 min, 120 min simulated)\n")
    print(f"{'policy':26s} {'agents':>6s} {'safe at 30 min':>15s} {'= % of shortest':>16s} {'last evacuee, s':>20s}")
    ref = np.mean([r["at30"] for n, r in out if n == "shortest path (metres)"])
    for label in policies:
        rows = [r for n, r in out if n == label]
        at30 = np.array([r["at30"] for r in rows])
        last = np.array([r["last"] for r in rows])
        print(f"{label:26s} {rows[0]['agents']:6d} {at30.mean():8.1f} ({100 * at30.mean() / rows[0]['agents']:4.1f}%) {100 * at30.mean() / ref:15.1f}% "
              f"{last.mean():12.0f} ± {last.std(ddof=1):4.0f}")
