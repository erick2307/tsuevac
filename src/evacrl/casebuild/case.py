# -*- coding: utf-8 -*-
"""Writing, reading and checking a case folder.

    cases/<name>/data/     nodesdb.csv linksdb.csv actionsdb.csv transitionsdb.csv agentsdb.csv nextnode.csv
    cases/<name>/raw/      nodes.csv links.csv: the network before the clean-up, so that the case can be rebuilt offline
    cases/<name>/provenance.json   how the case was made: inputs, settings, versions, counts
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from evacrl.casebuild.actions import actions_and_transitions, MAX_ACTIONS
from evacrl.casebuild.network import Network
from evacrl.casebuild.routing import NO_PATH, distance_to_shelter, _adjacency
from evacrl.tables import load_table

FILES = dict(nodes="nodesdb.csv", links="linksdb.csv", actions="actionsdb.csv", transitions="transitionsdb.csv",
             nextnode="nextnode.csv", agents="agentsdb.csv")


# ----------------------------------------------------------------------------------------------- writing / reading
def write_tables(data_dir, network, actions, transitions, nextnode=None, agents=None):
    """Write the tables of a case into `data_dir` (created if needed)."""
    os.makedirs(data_dir, exist_ok=True)
    path = lambda key: os.path.join(data_dir, FILES[key])
    np.savetxt(path("nodes"), network.nodes, delimiter=",", fmt="%d,%.6f,%.6f,%d,%d",
               header="number,coord_x,coord_y,evacuation,reward")
    np.savetxt(path("links"), network.links, delimiter=",", fmt="%d",
               header="number,node1,node2,length,width")
    np.savetxt(path("actions"), actions, delimiter=",", fmt="%d")
    np.savetxt(path("transitions"), transitions, delimiter=",", fmt="%d")
    if nextnode is not None:
        np.savetxt(path("nextnode"), nextnode, delimiter=",", fmt="%d", header="node,next node")
    if agents is not None:
        np.savetxt(path("agents"), agents, delimiter=",", fmt="%d", header="age,gender,hhType,hhId,Node")


def read_tables(data_dir):
    """The tables of a case as a dict of arrays (`nextnode` and `agents` only if the files exist)."""
    out = {}
    for key, name in FILES.items():
        file = os.path.join(data_dir, name)
        if os.path.exists(file):
            out[key] = load_table(file, dtype=float if key in ("nodes", "links") else int)
    return out


def write_raw(raw_dir, network):
    """The network before the clean-up: nodes with their source ids, links."""
    os.makedirs(raw_dir, exist_ok=True)
    osmid = network.osmid if network.osmid is not None else np.full(network.num_nodes, -1)
    nodes = np.column_stack([network.nodes[:, 0], osmid, network.nodes[:, 1:3], network.nodes[:, 3:5]])
    header = f"crs={network.crs}\nnumber,osmid,x,y,evacuation,reward" if network.crs else "number,osmid,x,y,evacuation,reward"
    np.savetxt(os.path.join(raw_dir, "nodes.csv"), nodes, delimiter=",", fmt="%d,%d,%.9f,%.9f,%d,%d", header=header)
    np.savetxt(os.path.join(raw_dir, "links.csv"), network.links, delimiter=",", fmt="%d",
               header="number,node1,node2,length,width")


def read_raw(raw_dir):
    """The `Network` written by `write_raw`."""
    nodes_file = os.path.join(raw_dir, "nodes.csv")
    crs = None
    with open(nodes_file, encoding="utf-8") as f:
        first = f.readline()
        if first.startswith("#") and "crs=" in first:
            crs = first.split("crs=", 1)[1].strip()
    nodes = load_table(nodes_file)
    links = load_table(os.path.join(raw_dir, "links.csv"))
    osmid = nodes[:, 1].astype(np.int64)
    arr = np.column_stack([nodes[:, 0], nodes[:, 2:4], nodes[:, 4:6]])
    return Network(arr, links, osmid=None if np.all(osmid < 0) else osmid, crs=crs)


def write_provenance(case_dir, info):
    """`provenance.json` of a case: whatever is needed to know how it was made."""
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        f.write("\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} to provenance.json")


# ---------------------------------------------------------------------------------------------------- validation
@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.errors

    def __str__(self):
        lines = [f"stats: {self.stats}"]
        lines += [f"ERROR   {e}" for e in self.errors]
        lines += [f"warning {w}" for w in self.warnings]
        lines.append("OK" if self.ok else f"{len(self.errors)} error(s)")
        return "\n".join(lines)


def validate_tables(nodes, links, actions, transitions, nextnode=None, agents=None):
    """Check that the tables of a case agree with each other and with what the model assumes.

    Errors would make the model fail or give wrong results; warnings are things worth knowing (isolated nodes, agents
    that start at a shelter, a next-node table that is not on shortest walks)."""
    r = Report()
    nodes = np.asarray(nodes, dtype=float)
    links = np.asarray(links, dtype=float)
    n, m = len(nodes), len(links)
    r.stats.update(nodes=n, links=m, shelters=int((nodes[:, 3] == 1).sum()))

    if not np.array_equal(nodes[:, 0], np.arange(n)):
        r.errors.append("node numbers are not 0, 1, 2, ... in order")
    if not np.array_equal(links[:, 0], np.arange(m)):
        r.errors.append("link numbers are not 0, 1, 2, ... in order")
    ends = links[:, 1:3]
    if m and (not np.array_equal(ends, np.rint(ends)) or ends.min() < 0 or ends.max() >= n):
        r.errors.append("some links point at nodes that do not exist")
        return r  # the rest assumes valid links
    if m and not np.array_equal(links[:, 3:5], np.rint(links[:, 3:5])):
        r.errors.append("link lengths and widths must be whole numbers (metres)")
    if m and (links[:, 3] <= 0).any():
        r.errors.append(f"{int((links[:, 3] <= 0).sum())} links have no length")
    if m and (links[:, 4] <= 0).any():
        r.errors.append(f"{int((links[:, 4] <= 0).sum())} links have no width")
    if m and (links[:, 1] == links[:, 2]).any():
        r.errors.append(f"{int((links[:, 1] == links[:, 2]).sum())} links join a node to itself")
    if r.stats["shelters"] == 0:
        r.errors.append("no node is an evacuation node")

    a, b = links[:, 1].astype(int), links[:, 2].astype(int)
    degree = np.bincount(np.concatenate([a, b]), minlength=n)
    isolated = int((degree == 0).sum())
    r.stats["isolated_nodes"] = isolated
    if isolated:
        r.warnings.append(f"{isolated} nodes have no links")
    ncomp, label = connected_components(coo_matrix((np.ones(m), (a, b)), shape=(n, n)), directed=False)
    r.stats["components"] = int(ncomp) - isolated
    if r.stats["components"] > 1:
        r.warnings.append(f"the network has {r.stats['components']} separate parts")

    net = Network(nodes, links)
    # actions and transitions
    try:
        exp_actions, exp_trans = actions_and_transitions(net)
    except ValueError as exc:
        r.errors.append(str(exc))
        exp_actions = exp_trans = None
    actions = np.asarray(actions)
    transitions = np.asarray(transitions)
    for name, got, exp in (("actionsdb", actions, exp_actions), ("transitionsdb", transitions, exp_trans)):
        if got.shape != (n, 2 + MAX_ACTIONS):
            r.errors.append(f"{name} has shape {got.shape}, expected {(n, 2 + MAX_ACTIONS)}")
        elif exp is not None and not np.array_equal(got, exp):
            same_sets = np.array_equal(np.sort(got, axis=1), np.sort(exp, axis=1))
            if same_sets:
                r.warnings.append(f"{name} lists the choices of some nodes in another order than the links table")
            else:
                bad = int((np.sort(got, axis=1) != np.sort(exp, axis=1)).any(axis=1).sum())
                r.errors.append(f"{name} does not match the links table at {bad} nodes")

    # next node
    if nextnode is not None:
        nextnode = np.asarray(nextnode)
        if nextnode.shape != (n, 2) or not np.array_equal(nextnode[:, 0], np.arange(n)):
            r.errors.append("nextnode must have one row [node, next node] per node, in order")
        else:
            _check_next_nodes(net, nextnode[:, 1], degree, r)

    # agents
    if agents is not None:
        agents = np.asarray(agents)
        r.stats["agents"] = len(agents)
        start = agents[:, 4].astype(int) if agents.ndim == 2 and agents.shape[1] >= 5 else None
        if start is None:
            r.errors.append("agentsdb needs 5 columns, the last is the start node")
        elif len(start) and (start.min() < 0 or start.max() >= n):
            r.errors.append("some agents start at nodes that do not exist")
        elif len(start):
            at_shelter = float(np.mean(nodes[start, 3] == 1))
            r.stats["agents_at_shelters_%"] = round(100 * at_shelter, 2)
            if at_shelter:
                r.warnings.append(f"{100 * at_shelter:.1f}% of the agents start at a shelter (counted as evacuated at time 0)")
            if (degree[start] == 0).any():
                r.errors.append(f"{int((degree[start] == 0).sum())} agents start at nodes with no links")
            if nextnode is not None and nextnode.shape == (n, 2):
                stuck = int((nextnode[start, 1] == NO_PATH).sum())
                if stuck:
                    r.errors.append(f"{stuck} agents start at nodes with no way to a shelter")
    return r


def _check_next_nodes(net, nxt, degree, r):
    n = net.num_nodes
    shelter = net.nodes[:, 3] == 1
    dist = distance_to_shelter(net)
    r.stats["nodes_without_a_way_out"] = int(((nxt == NO_PATH) & (degree > 0) & ~shelter).sum())
    has = nxt != NO_PATH
    if (nxt[shelter] != np.where(shelter)[0]).any():
        r.errors.append("the next node of a shelter must be the shelter itself")
    if (((nxt < 0) & (nxt != NO_PATH)) | (nxt >= n)).any():
        r.errors.append("nextnode refers to nodes that do not exist")
        return
    reachable = np.isfinite(dist)
    missing = int((~has & reachable & ~shelter).sum())
    if missing:
        r.errors.append(f"{missing} nodes have a way to a shelter but no next node")
    useless = int((has & ~reachable).sum())
    if useless:
        r.errors.append(f"{useless} nodes have a next node but no way to a shelter")
    # the next node must be a neighbour; every walk must end at a shelter
    adjacency = _adjacency(net, "min").tocsr()
    walk = has & ~shelter
    idx = np.where(walk)[0]
    ok = np.asarray([adjacency[i, nxt[i]] > 0 for i in idx], dtype=bool) if len(idx) else np.array([], dtype=bool)
    if (~ok).any():
        r.errors.append(f"{int((~ok).sum())} next nodes are not neighbours")
        return
    f = np.where(has, nxt, np.arange(n))
    for _ in range(max(int(np.ceil(np.log2(max(n, 2)))), 1) + 1):
        f = f[f]
    stuck = int((has & ~shelter[f]).sum())
    if stuck:
        r.errors.append(f"{stuck} nodes never reach a shelter by following the next nodes")
        return
    step = np.asarray([adjacency[i, nxt[i]] for i in idx]) if len(idx) else np.array([])
    off = int((np.abs(dist[idx] - (step + dist[nxt[idx]])) > 1e-9).sum())
    r.stats["next_nodes_off_the_shortest_walk"] = off
    if off:
        r.warnings.append(f"{off} next nodes are not on a shortest walk (several links between the same two nodes?)")


def validate_case(data_dir):
    """`validate_tables` on the tables in `data_dir`."""
    t = read_tables(data_dir)
    for key in ("nodes", "links", "actions", "transitions"):
        if key not in t:
            r = Report()
            r.errors.append(f"{FILES[key]} is missing in {data_dir}")
            return r
    return validate_tables(t["nodes"], t["links"], t["actions"], t["transitions"], t.get("nextnode"), t.get("agents"))
