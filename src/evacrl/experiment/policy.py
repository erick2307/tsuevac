# -*- coding: utf-8 -*-
"""What a learned policy does: the next node it chooses at every node, and how that compares with the shortest path.

The state matrix has one row per state; the first `n` rows are the states of the nodes with no crowded link (density codes 0), so
the greedy choice at node `i` is the best-valued of the first `k` action values of row `i` (`k` = the links of the node). That is the
choice an agent makes on an empty network; the crowding states of the same node can choose differently."""
from dataclasses import dataclass

import numpy as np

from evacrl.experiment.runs import validate_state
from evacrl.tables import load_table

NO_PATH = -9999


def _tables(case):
    nodes = load_table(case.kwargs["nodesdbFile"])
    links = load_table(case.kwargs["linksdbFile"], dtype=int)
    transitions = load_table(case.kwargs["transNodedbFile"], dtype=int)
    return nodes, links, transitions


def greedy_next_nodes(case, state):
    """The node each node's greedy choice leads to (a shelter: itself; a node without links: -9999)."""
    nodes, _, transitions = _tables(case)
    state = np.asarray(state)
    validate_state(state, len(nodes), transitions[:, 1])
    out = np.full(len(nodes), NO_PATH)
    for i in range(len(nodes)):
        if nodes[i, 3] == 1:
            out[i] = i
            continue
        k = int(transitions[i, 1])
        if k == 0:
            continue
        if i >= len(state) or int(state[i, 0]) != i:
            raise ValueError(f"row {i} of the state matrix is not the state of node {i}: not a state matrix of this case")
        out[i] = transitions[i, 2 + int(np.argmax(state[i, 11:11 + k]))]
    return out


def walks(case, next_nodes, starts=None):
    """`(metres, nodes passed)` of the walk from each start node following `next_nodes` to a shelter; NaN for a walk that never
    arrives (a loop, or a node without a way on). `starts`: node numbers (default: every node). The shortest of parallel links counts."""
    nodes, links, _ = _tables(case)
    length = {}
    for _, a, b, l, _ in links:
        for u, v in ((int(a), int(b)), (int(b), int(a))):
            length[(u, v)] = min(length.get((u, v), np.inf), float(l))
    starts = np.arange(len(nodes)) if starts is None else np.asarray(starts, dtype=int)
    metres, hops = np.full(len(starts), np.nan), np.full(len(starts), np.nan)
    for j, start in enumerate(starts):
        u, m, h = int(start), 0.0, 0
        while nodes[u, 3] != 1 and h <= len(nodes):
            v = int(next_nodes[u])
            if v == NO_PATH or (u, v) not in length:
                h = len(nodes) + 1
                break
            m += length[(u, v)]
            h += 1
            u = v
        if h <= len(nodes):
            metres[j], hops[j] = m, h
    return metres, hops


@dataclass
class PolicyComparison:
    """A policy against the shortest path over the agents' start nodes."""
    agents: int
    agreement: float            # share of the agents whose first node choice is the shortest path's
    metres: float               # mean walk of the policy (agents that arrive)
    metres_sp: float
    longer: float               # relative excess over the shortest path, %
    hops: float
    hops_sp: float
    never_arrive: int           # agents whose walk loops or ends


def compare_with_shortest_path(case, state, next_sp=None):
    """Follow the greedy policy of `state` and the shortest path from the start node of every agent of the case."""
    if next_sp is None:
        if case.nextnode is None:
            raise ValueError(f"the case {case.name!r} has no nextnode.csv")
        next_sp = load_table(case.nextnode, dtype=int)[:, 1]
    starts = load_table(case.kwargs["agentsProfileName"], dtype=int)[:, 4]
    policy = greedy_next_nodes(case, state)
    m, h = walks(case, policy, starts)
    ms, hs = walks(case, next_sp, starts)
    ok = ~np.isnan(m) & ~np.isnan(ms)
    nodes, _, _ = _tables(case)
    moving = nodes[starts, 3] != 1
    return PolicyComparison(agents=len(starts), agreement=float(np.mean(policy[starts][moving] == next_sp[starts][moving])),
                            metres=float(m[ok].mean()), metres_sp=float(ms[ok].mean()), longer=float(100 * (m[ok].mean() / ms[ok].mean() - 1)),
                            hops=float(h[ok].mean()), hops_sp=float(hs[ok].mean()), never_arrive=int(np.isnan(m).sum()))
