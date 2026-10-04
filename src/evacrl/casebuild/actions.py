# -*- coding: utf-8 -*-
"""Actions and transitions: what an agent can do at each node, as the model reads them (`actionsdb.csv`, `transitionsdb.csv`).

Row `i` of both tables describes node `i`: `[i, k, ...]` with `k` the number of choices and then `k` entries (padded
with 0 to 2 + 10 columns, the state matrix of the model has room for 10 choices). In `actionsdb` the entries are the
links, in `transitionsdb` the node each leads to. A node with `k = 0` has no links (it is isolated). An evacuation
node has exactly one choice, to stay: link -1, to itself. Links are listed in the order of the links table, first
those that start at the node, then those that end at it.
"""
import numpy as np

MAX_ACTIONS = 10


def actions_and_transitions(network, max_actions=MAX_ACTIONS):
    """`(actions, transitions)`, int arrays of shape (n, 2 + max_actions). Raises `ValueError` if a node has more
    than `max_actions` links (the state matrix of the model cannot hold more)."""
    n = network.num_nodes
    links = network.links
    m = len(links)
    nodes = np.arange(n)
    is_shelter = network.nodes[:, 3] == 1

    # one entry per (node, link) incidence: node, side (0: link starts here, 1: ends here), link index, other end
    node_of = np.concatenate([links[:, 1], links[:, 2]]).astype(int)
    side = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    link_index = np.concatenate([np.arange(m), np.arange(m)])
    other_end = np.concatenate([links[:, 2], links[:, 1]]).astype(int)
    keep = ~is_shelter[node_of]                      # an evacuation node ignores its links
    node_of, side, link_index, other_end = node_of[keep], side[keep], link_index[keep], other_end[keep]
    order = np.lexsort((link_index, side, node_of))  # by node, then side, then link order
    node_of, link_index, other_end = node_of[order], link_index[order], other_end[order]

    degree = np.bincount(node_of, minlength=n)
    too_many = np.where(degree > max_actions)[0]
    if len(too_many):
        raise ValueError(f"{len(too_many)} nodes have more than {max_actions} links, the most the model can hold "
                         f"(first: node {int(too_many[0])} with {int(degree[too_many[0]])})")
    first = np.concatenate([[0], np.cumsum(degree)[:-1]])
    position = np.arange(len(node_of)) - first[node_of]

    actions = np.zeros((n, 2 + max_actions), dtype=int)
    transitions = np.zeros((n, 2 + max_actions), dtype=int)
    actions[:, 0] = transitions[:, 0] = nodes
    actions[:, 1] = transitions[:, 1] = degree
    actions[node_of, 2 + position] = links[link_index, 0].astype(int)
    transitions[node_of, 2 + position] = other_end
    shelters = np.where(is_shelter)[0]
    actions[shelters, 1] = transitions[shelters, 1] = 1
    actions[shelters, 2] = -1
    transitions[shelters, 2] = shelters
    return actions, transitions


def prune_excess_links(network, max_actions=MAX_ACTIONS):
    """Remove links until no node has more than `max_actions`: at the node with the most links, the longest one goes first.

    Returns `(network, removed)`, `removed` being the numbers (in the given network) of the links taken out. A removed
    link is gone from both of its ends: the model cannot hold more choices than `max_actions` per node, and a table that
    silently dropped the extra link at one end only (as `cases/new_kochi` does) would let agents walk it one way. This can
    cut a part of the network off; `validate_tables` says so. Evacuation nodes ignore their links and are not counted."""
    from evacrl.casebuild.network import Network
    links = network.links
    shelter = network.nodes[:, 3] == 1
    alive = np.ones(len(links), dtype=bool)
    ends = links[:, 1:3].astype(int)
    removed = []
    while True:
        degree = np.zeros(network.num_nodes, dtype=int)
        for column in (0, 1):
            np.add.at(degree, ends[alive, column], 1)
        degree[shelter] = 0
        worst = int(np.argmax(degree))
        if degree[worst] <= max_actions:
            break
        mine = np.where(alive & ((ends[:, 0] == worst) | (ends[:, 1] == worst)))[0]
        longest = mine[np.argmax(links[mine, 3])]          # the first of the longest
        alive[longest] = False
        removed.append(int(longest))
    kept = links[alive].copy()
    kept[:, 0] = np.arange(len(kept))
    return Network(network.nodes.copy(), kept, osmid=network.osmid, crs=network.crs), removed

