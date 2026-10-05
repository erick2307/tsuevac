# -*- coding: utf-8 -*-
"""The shortest-path table (`nextnode.csv`): for every node, the next node on its shortest walk to the nearest shelter.

Rows are `[node, next node]`: a shelter's next node is itself, and a node that has no way to a shelter has -9999. Links
can be walked in both directions; their length is the cost. This is the baseline the learned policy is compared with.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

NO_PATH = -9999


def _adjacency(network, parallel):
    """Symmetric weight matrix, sparse. Several links between the same two nodes: `parallel="min"` keeps the shortest,
    `"last"` the last one in the table (what the 2024 study did)."""
    if parallel not in ("min", "last"):
        raise ValueError("parallel must be 'min' or 'last'")
    n = network.num_nodes
    a = network.links[:, 1].astype(int)
    b = network.links[:, 2].astype(int)
    w = network.links[:, 3]
    usable = (a != b) & (w > 0)                      # a loop leads nowhere; a link without length is not a link
    a, b, w = a[usable], b[usable], w[usable]
    if len(a) == 0:
        return csr_matrix((n, n))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * n + hi
    if parallel == "min":
        order = np.lexsort((w, key))                 # within a pair, the shortest first
        sorted_key = key[order]
        pick = order[np.concatenate([[True], sorted_key[1:] != sorted_key[:-1]])]
    else:
        order = np.argsort(key, kind="stable")       # within a pair, in the order of the table
        sorted_key = key[order]
        pick = order[np.concatenate([sorted_key[1:] != sorted_key[:-1], [True]])]
    lo, hi, w = lo[pick], hi[pick], w[pick]
    return csr_matrix((np.concatenate([w, w]), (np.concatenate([lo, hi]), np.concatenate([hi, lo]))), shape=(n, n))


def _lowest_step_on_a_shortest_walk(adjacency, dist):
    """For every node, the neighbour with the lowest number among those that lie on a shortest walk to a shelter, that is,
    those with `dist[neighbour] + length == dist[node]` (`NO_PATH` where `dist` is infinite). Every length is positive, so
    the neighbour is strictly nearer and the steps lead to a shelter."""
    n = len(dist)
    a = adjacency.tocoo()
    row, col, length = a.row, a.col, a.data
    ok = np.isfinite(dist[row]) & np.isclose(dist[col] + length, dist[row], rtol=1e-12, atol=1e-9)
    row, col = row[ok], col[ok]
    order = np.lexsort((col, row))
    row, col = row[order], col[order]
    first = np.ones(len(row), dtype=bool)
    first[1:] = row[1:] != row[:-1]
    step = np.full(n, NO_PATH, dtype=int)
    step[row[first]] = col[first]
    return step


def next_nodes(network, method="nearest", parallel="min"):
    """`(n, 2)` int array `[node, next node]`.

    method="nearest" (default): one Dijkstra search started from all the shelters at once, so the memory is that of
        the network, not of its square. Where two shelters are equally near, or two walks are equally short, the next
        node is the one with the lowest number among those on a shortest walk: the table depends neither on the order of
        the links nor on the version of SciPy (its own tie-breaking changes between versions).
    method="allpairs": the 2024 study's way, a full distance matrix (n x n: too large beyond a few thousand nodes) and
        the first shelter among the nearest, ties as SciPy's `dijkstra` breaks them (they can differ between SciPy
        versions). Kept to reproduce its tables.
    """
    n = network.num_nodes
    shelters = network.shelters
    table = np.full((n, 2), NO_PATH, dtype=int)
    table[:, 0] = np.arange(n)
    if len(shelters) == 0:
        return table
    adjacency = _adjacency(network, parallel)
    if method == "nearest":
        dist = dijkstra(adjacency, directed=False, indices=shelters, min_only=True)
        table[:, 1] = _lowest_step_on_a_shortest_walk(adjacency, dist)
    elif method == "allpairs":
        dist, predecessors = dijkstra(adjacency, directed=False, return_predecessors=True)
        predecessors[np.arange(n), np.arange(n)] = np.arange(n)
        for node in range(n):
            target = shelters[np.argmin(dist[node, shelters])]
            table[node, 1] = predecessors[target, node]
    else:
        raise ValueError("method must be 'nearest' or 'allpairs'")
    table[shelters, 1] = shelters
    return table


def distance_to_shelter(network, parallel="min"):
    """Walking distance in metres from every node to its nearest shelter (inf where there is none)."""
    n = network.num_nodes
    if len(network.shelters) == 0:
        return np.full(n, np.inf)
    dist = dijkstra(_adjacency(network, parallel), directed=False, indices=network.shelters, min_only=True)
    return dist
