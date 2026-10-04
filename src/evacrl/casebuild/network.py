# -*- coding: utf-8 -*-
"""The road network of a case, and the clean-up that turns a raw OpenStreetMap network into one the model can run on.

A `Network` holds the two tables the model reads (`nodesdb.csv`, `linksdb.csv`) as arrays:

    nodes  [number, x, y, evacuation, reward]     x, y in metres (a projected CRS); evacuation 1 = shelter
    links  [number, node1, node2, length, width]  length in whole metres; a link can be walked in both directions

Raw networks come from `network_from_edges` (what `evacrl.casebuild.geo` makes of an OSM graph) or from the
tables of an earlier run (`evacrl.casebuild.case.read_raw`). `merge_short_links` then removes the links that are
too short to matter and merges the nodes they join.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SHELTER_REWARD = 1000  # nodesdb "reward" column: informational, the model does not read it
STREET_REWARD = 1
DEFAULT_WIDTH = 3      # metres; OSM has no usable width for most streets


@dataclass
class Network:
    nodes: np.ndarray                 # float, (n, 5): number, x, y, evacuation, reward
    links: np.ndarray                 # float, (m, 5): number, node1, node2, length, width
    osmid: Optional[np.ndarray] = None  # (n,) id of each node in the source (OSM), same order as `nodes`
    crs: Optional[str] = None         # CRS of x, y, e.g. "EPSG:32653"

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 5)
        self.links = np.asarray(self.links, dtype=float).reshape(-1, 5)
        if self.osmid is not None:
            self.osmid = np.asarray(self.osmid)
            if len(self.osmid) != len(self.nodes):
                raise ValueError("osmid must have one entry per node")

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_links(self):
        return len(self.links)

    @property
    def shelters(self):
        """Numbers of the evacuation nodes."""
        return np.where(self.nodes[:, 3] == 1)[0]

    def copy(self):
        return replace(self, nodes=self.nodes.copy(), links=self.links.copy(),
                       osmid=None if self.osmid is None else self.osmid.copy())


def network_from_edges(osmid, xy, edges, evacuation_osmids=(), width=DEFAULT_WIDTH, length_rounding="floor", crs=None):
    """The raw network of an OSM graph that has been projected and made undirected.

    osmid   (n,) OSM id of every node, in the order the nodes are numbered
    xy      (n, 2) projected coordinates, metres
    edges   iterable of (osmid of node 1, osmid of node 2, length in metres)
    evacuation_osmids   OSM ids of the nodes that are shelters

    Loops and links shorter than 1 m are dropped. Lengths become whole metres: `length_rounding="floor"` truncates
    (what the 2024 study did, and so what its tables contain), `"round"` rounds to the nearest metre.
    """
    if length_rounding not in ("floor", "round"):
        raise ValueError("length_rounding must be 'floor' or 'round'")
    osmid = np.asarray(osmid)
    xy = np.asarray(xy, dtype=float)
    number = {int(o): i for i, o in enumerate(osmid)}
    nodes = np.zeros((len(osmid), 5))
    nodes[:, 0] = np.arange(len(osmid))
    nodes[:, 1:3] = xy
    nodes[:, 4] = STREET_REWARD
    for o in evacuation_osmids:
        if int(o) in number:
            nodes[number[int(o)], 3] = 1
            nodes[number[int(o)], 4] = SHELTER_REWARD
    rows = []
    for u, v, length in edges:
        if u == v:
            continue
        whole = int(length) if length_rounding == "floor" else int(round(length))
        if whole == 0:
            continue
        rows.append((number[int(u)], number[int(v)], whole))
    links = np.zeros((len(rows), 5))
    if rows:
        links[:, 0] = np.arange(len(rows))
        links[:, 1:4] = rows
    links[:, 4] = width
    return Network(nodes, links, osmid=osmid, crs=crs)


def attach_shelters(network, xy, merge_radius=5.0, length_rounding="floor", width=DEFAULT_WIDTH):
    """Add a shelter at each location of `xy`, joined to the nearest street node by an access link: `(network, info)`.

    network   a `Network`; the coordinates of `xy` are in its CRS (metres)
    xy        (k, 2) locations of the shelters
    merge_radius   locations that lie within this many metres of each other (directly, or through a chain of such
              locations) are one shelter, at their mean position: the same building listed twice is one shelter

    The new node sits where the shelter is, and its only link goes to the nearest node that is not itself a shelter
    (nothing can pass through a shelter, so a shelter attached to another one would be out of reach). The link's length
    is the distance between the two, in whole metres (`length_rounding` as in `network_from_edges`), its width
    `width`. So the walk from the street to the shelter counts, however far the shelter is from the network; and the
    street node stays an ordinary node, which a shelter snapped *onto* it would not.

    A shelter closer than 1 m to a node is that node: it is marked, and nothing is added. The links to the new
    nodes are numbered after the existing ones, the nodes likewise; the new nodes have a negative `osmid` (-1, -2, ...
    below any that exist), as OpenStreetMap ids are positive. `info` counts what was done.
    """
    from scipy.spatial import cKDTree
    if length_rounding not in ("floor", "round"):
        raise ValueError("length_rounding must be 'floor' or 'round'")
    if merge_radius < 0:
        raise ValueError("merge_radius must not be negative")
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    k = len(xy)
    out = network.copy()
    info = dict(points=k, shelters=0, merged_points=0, on_a_node=0, new_nodes=0)
    if k == 0:
        return out, info
    street = np.where(network.nodes[:, 3] != 1)[0]
    if street.size == 0:
        raise ValueError("every node is already a shelter: there is nothing to attach the shelters to")

    # the locations that are one shelter
    pairs = cKDTree(xy).query_pairs(merge_radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(k, k))
    ngroups, label = connected_components(graph, directed=False)
    smallest = np.full(ngroups, k)
    np.minimum.at(smallest, label, np.arange(k))
    rank = np.empty(ngroups, dtype=int)
    rank[np.argsort(smallest)] = np.arange(ngroups)   # groups in the order of their first location
    group = rank[label]
    size = np.bincount(group, minlength=ngroups)
    where = np.column_stack([np.bincount(group, weights=xy[:, 0], minlength=ngroups) / size,
                             np.bincount(group, weights=xy[:, 1], minlength=ngroups) / size])

    distance, nearest = cKDTree(network.nodes[street, 1:3]).query(where)
    nearest = street[nearest]
    on_a_node = distance < 1.0
    for node in np.unique(nearest[on_a_node]):
        out.nodes[int(node), 3] = 1
        out.nodes[int(node), 4] = SHELTER_REWARD
    added = np.where(~on_a_node)[0]
    n, m = network.num_nodes, network.num_links
    nodes = np.zeros((len(added), 5))
    nodes[:, 0] = n + np.arange(len(added))
    nodes[:, 1:3] = where[added]
    nodes[:, 3] = 1
    nodes[:, 4] = SHELTER_REWARD
    lengths = np.floor(distance[added]) if length_rounding == "floor" else np.round(distance[added])
    links = np.zeros((len(added), 5))
    links[:, 0] = m + np.arange(len(added))
    links[:, 1] = nearest[added]
    links[:, 2] = nodes[:, 0]
    links[:, 3] = lengths
    links[:, 4] = width
    out.nodes = np.vstack([out.nodes, nodes])
    out.links = np.vstack([out.links, links])
    if network.osmid is not None:
        first = min(int(network.osmid.min()), 0) - 1
        out.osmid = np.concatenate([network.osmid, first - np.arange(len(added))])
    info.update(shelters=int(ngroups), merged_points=int(k - ngroups), on_a_node=int(on_a_node.sum()), new_nodes=len(added))
    if len(added):
        info.update(access_length_median=round(float(np.median(lengths)), 1), access_length_max=int(lengths.max()),
                    access_over_100_m=int((lengths > 100).sum()))
    return out, info


def merge_short_links(network, threshold=5.0, method="clusters"):
    """Remove the links shorter than `threshold` metres and merge the nodes they join.

    Returns `(network, report)`. `report` counts what happened.

    method="clusters" (default): the nodes joined by a chain of short links form a cluster; the cluster becomes one
        node at the mean position of its members, a shelter if any member is one; the other links are re-attached to
        it, a link that would join the cluster to itself is dropped, and the nodes are renumbered 0, 1, 2, ...
    method="legacy": what `EVACMODEL3_FocalPoints/preprocess.py` of the 2024 study did, reproduced exactly. It
        handles only a pair of nodes at a time: it leaves one orphan row (no links, the merged position, which no
        link uses) per short link, merges chains of short links wrongly (the nodes of a chain end up in different
        places), and keeps the position of the first node. Kept to reproduce that study's tables.
    """
    if method == "legacy":
        return _merge_legacy(network, threshold)
    if method != "clusters":
        raise ValueError("method must be 'clusters' or 'legacy'")
    n = network.num_nodes
    links = network.links
    short = links[:, 3] < threshold
    a = links[short, 1].astype(int)
    b = links[short, 2].astype(int)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    ncomp, label = connected_components(graph, directed=False)
    smallest = np.full(ncomp, n)
    np.minimum.at(smallest, label, np.arange(n))
    order = np.argsort(smallest)             # clusters in the order of their smallest member
    rank = np.empty(ncomp, dtype=int)
    rank[order] = np.arange(ncomp)
    new_id = rank[label]                     # new number of every old node
    size = np.bincount(new_id, minlength=ncomp)

    nodes = np.zeros((ncomp, 5))
    nodes[:, 0] = np.arange(ncomp)
    nodes[:, 1] = np.bincount(new_id, weights=network.nodes[:, 1], minlength=ncomp) / size
    nodes[:, 2] = np.bincount(new_id, weights=network.nodes[:, 2], minlength=ncomp) / size
    np.maximum.at(nodes[:, 3], new_id, network.nodes[:, 3])
    np.maximum.at(nodes[:, 4], new_id, network.nodes[:, 4])

    keep = ~short
    n1 = new_id[links[keep, 1].astype(int)]
    n2 = new_id[links[keep, 2].astype(int)]
    loop = n1 == n2
    out = np.zeros((int((~loop).sum()), 5))
    out[:, 0] = np.arange(len(out))
    out[:, 1] = n1[~loop]
    out[:, 2] = n2[~loop]
    out[:, 3] = links[keep][~loop, 3]
    out[:, 4] = links[keep][~loop, 4]
    osmid = None if network.osmid is None else network.osmid[smallest[order]]
    shelters_before = int(network.nodes[:, 3].sum())
    report = dict(method="clusters", threshold=threshold, short_links=int(short.sum()),
                  nodes_before=n, nodes_after=ncomp, links_before=network.num_links, links_after=len(out),
                  clusters_of_3_or_more=int((size >= 3).sum()), links_dropped_as_loops=int(loop.sum()),
                  shelters_before=shelters_before, shelters_after=int(nodes[:, 3].sum()),
                  clusters_with_several_shelters=int((np.bincount(new_id, weights=network.nodes[:, 3], minlength=ncomp) > 1).sum()))
    return Network(nodes, out, osmid=osmid, crs=network.crs), report


def _merge_legacy(network, threshold):
    """`clean_short_links` of EVACMODEL3_FocalPoints/preprocess.py, on arrays instead of files (same operations)."""
    links = network.links.copy()
    nodes = network.nodes
    updated_nodes, updated_links, node_mapping = [], [], {}
    for link in links:
        if link[3] < threshold:
            node1, node2 = int(link[1]), int(link[2])
            node1_data = nodes[nodes[:, 0] == node1][0]
            node2_data = nodes[nodes[:, 0] == node2][0]
            merged_x = (node1_data[1] + node2_data[1]) / 2
            merged_y = (node1_data[2] + node2_data[2]) / 2
            updated_nodes.append([node1, merged_x, merged_y, node1_data[3], node1_data[4]])
            node_mapping[node2] = node1
        else:
            updated_links.append([link[0], link[1], link[2], link[3], link[4]])
    for link in updated_links:
        link[1] = node_mapping.get(link[1], link[1])
        link[2] = node_mapping.get(link[2], link[2])
    updated_nodes.extend([node for node in nodes if node[0] not in node_mapping])
    updated_links = np.array(updated_links).reshape(-1, 5)
    updated_nodes = np.array(updated_nodes)
    orphans = len(updated_nodes) - len(np.unique(updated_nodes[:, 0]))  # rows of a node that also has another row
    new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(updated_nodes[:, 0])}
    updated_nodes[:, 0] = np.arange(len(updated_nodes))
    for link in updated_links:
        link[1] = new_id_mapping.get(link[1], link[1])
        link[2] = new_id_mapping.get(link[2], link[2])
    updated_links[:, 0] = np.arange(len(updated_links))
    short = int((network.links[:, 3] < threshold).sum())
    report = dict(method="legacy", threshold=threshold, short_links=short, nodes_before=network.num_nodes,
                  nodes_after=len(updated_nodes), links_before=network.num_links, links_after=len(updated_links),
                  orphan_rows=int(orphans))
    return Network(updated_nodes, updated_links, osmid=None, crs=network.crs), report
