# -*- coding: utf-8 -*-
"""Where the agents start (`agentsdb.csv`: one row per agent, the last column is its start node).

The model reads only the start node. An agent can start at any node that has a way to a shelter; by default not at a
shelter itself (it would be counted as evacuated at time 0).
"""
import numpy as np

from evacrl.casebuild.routing import NO_PATH


def _check_weights(w):
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("the weights must be finite and not negative")


def candidate_nodes(network, nextnode, exclude_shelters=True):
    """Numbers of the nodes an agent can start at: those with a way to a shelter, and not a shelter."""
    ok = nextnode[:, 1] != NO_PATH
    if exclude_shelters:
        ok &= network.nodes[:, 3] != 1
    nodes = np.where(ok)[0]
    if len(nodes) == 0:
        raise ValueError("no node has a way to a shelter that is not a shelter itself")
    return nodes


def start_nodes(network, nextnode, count, rng, weights=None, exclude_shelters=True):
    """`count` start nodes drawn at random, each candidate node equally likely, or in proportion to `weights` (one per
    node of the network). `rng` is a `numpy.random.Generator`; the result is sorted."""
    nodes = candidate_nodes(network, nextnode, exclude_shelters)
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[nodes]
        _check_weights(w)
        if w.sum() <= 0:
            raise ValueError("the weights of the candidate nodes are all zero")
        p = w / w.sum()
    return np.sort(rng.choice(nodes, size=count, p=p))


def start_nodes_per_node(network, nextnode, per_node, exclude_shelters=True):
    """`per_node` agents at every candidate node."""
    return np.repeat(candidate_nodes(network, nextnode, exclude_shelters), per_node)


def apportion(weights, total):
    """Whole numbers in proportion to `weights` that add up to `total` (largest remainders): no randomness."""
    w = np.asarray(weights, dtype=float)
    _check_weights(w)
    if w.sum() <= 0:
        raise ValueError("the weights are all zero")
    exact = w / w.sum() * total
    counts = np.floor(exact).astype(int)
    left = int(total - counts.sum())
    if left:
        # the largest fractions first; ties go to the lower node, so the result is always the same
        order = np.lexsort((np.arange(len(w)), -(exact - counts)))
        counts[order[:left]] += 1
    return counts


def start_nodes_proportional(network, nextnode, weights, total, exclude_shelters=True):
    """`total` start nodes spread over the candidate nodes in proportion to `weights` (one per node of the network)."""
    nodes = candidate_nodes(network, nextnode, exclude_shelters)
    w = np.asarray(weights, dtype=float)[nodes]
    return np.repeat(nodes, apportion(w, total))


def agents_table(starts):
    """`agentsdb` rows [age, gender, hhType, hhId, node]; the model only reads the node, the rest is 0."""
    starts = np.asarray(starts, dtype=int)
    table = np.zeros((len(starts), 5), dtype=int)
    table[:, 4] = starts
    return table
