# -*- coding: utf-8 -*-
"""From a raw network to a case folder, in one call.

    raw network --merge short links--> network --actions/transitions, next nodes, agents--> tables --write--> cases/<name>/

The raw network comes from `evacrl.casebuild.geo` (an OSM download or a snapshot of one) or from `read_raw` (a folder written
by an earlier build). Nothing here needs more than NumPy and SciPy.
"""
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

from evacrl.casebuild import case as case_io
from evacrl.casebuild.actions import actions_and_transitions, prune_excess_links
from evacrl.casebuild.network import Network, merge_short_links
from evacrl.casebuild.population import (agents_table, start_nodes, start_nodes_per_node, start_nodes_proportional)
from evacrl.casebuild.routing import next_nodes

# What the 2024 Kochi study did, for reproducing its tables (see docs/audits/step3).
LEGACY = dict(merge="legacy", parallel="last")


@dataclass
class PopulationSpec:
    """How many agents there are and where they start.

    strategy  "uniform"       `total` agents, each start node equally likely (what the 2024 study did)
              "per_node"      `per_node` agents at every start node
              "proportional"  `total` agents in proportion to `weights(network)`, the same every time (no randomness):
                              for instance a census (see `evacrl.casebuild.geo.node_weights`)
    exclude_shelters  agents do not start at a shelter (they would count as evacuated at time 0)
    seed              of the random draw ("uniform")
    """
    strategy: str = "uniform"
    total: Optional[int] = None
    per_node: int = 1
    weights: Optional[Callable[[Network], np.ndarray]] = None
    exclude_shelters: bool = True
    seed: int = 0

    def settings(self):
        d = asdict(self)
        d["weights"] = None if self.weights is None else getattr(self.weights, "__name__", "callable")
        return d


def build_tables(raw, merge="clusters", threshold=5.0, parallel="min", routing="nearest", population=None, excess="error"):
    """The tables of a case from its raw network: a dict with `network`, `actions`, `transitions`, `nextnode`,
    `agents` (None without a `population`) and `merge_report`.

    excess  what to do with a node that has more than 10 links, the most the model can hold: "error" (default) or
            "prune" (remove the longest links there, except those into a shelter; see `prune_excess_links`; how many in
            `merge_report`)."""
    network, report = merge_short_links(raw, threshold=threshold, method=merge)
    if excess == "prune":
        network, removed = prune_excess_links(network)
        report["links_pruned_for_degree"] = len(removed)
    elif excess != "error":
        raise ValueError("excess must be 'error' or 'prune'")
    if network.shelters.size == 0:
        raise ValueError("no node is an evacuation node: no shelter lies in the area, or none was close enough to a node")
    actions, transitions = actions_and_transitions(network)
    nextnode = next_nodes(network, method=routing, parallel=parallel)
    agents = None
    if population is not None:
        rng = np.random.default_rng(population.seed)
        if population.strategy == "uniform":
            if population.total is None:
                raise ValueError("strategy 'uniform' needs `total`")
            starts = start_nodes(network, nextnode, population.total, rng, exclude_shelters=population.exclude_shelters)
        elif population.strategy == "per_node":
            starts = start_nodes_per_node(network, nextnode, population.per_node, population.exclude_shelters)
        elif population.strategy == "proportional":
            if population.weights is None or population.total is None:
                raise ValueError("strategy 'proportional' needs `weights` and `total`")
            starts = start_nodes_proportional(network, nextnode, population.weights(network), population.total,
                                              population.exclude_shelters)
        else:
            raise ValueError("strategy must be 'uniform', 'per_node' or 'proportional'")
        agents = agents_table(starts)
    return dict(network=network, actions=actions, transitions=transitions, nextnode=nextnode, agents=agents,
                merge_report=report)


def build_case(raw, case_dir, population=None, merge="clusters", threshold=5.0, parallel="min", routing="nearest",
               provenance=None, require_valid=True, excess="error", write_raw=True):
    """Make the case `case_dir` (a folder such as `cases/kochi_area2`) from a raw network.

    Writes `data/` (the tables the model reads), `raw/` (the network before the clean-up, to rebuild offline) and
    `provenance.json`. The tables are validated; with `require_valid` an invalid case raises `ValueError` (nothing is
    written then). `write_raw=False` leaves `raw/` as it is (rebuilding a case from its own `raw/`). Returns `(tables, report)`.
    """
    tables = build_tables(raw, merge, threshold, parallel, routing, population, excess)
    report = case_io.validate_tables(tables["network"].nodes, tables["network"].links, tables["actions"],
                                     tables["transitions"], tables["nextnode"], tables["agents"])
    if require_valid and not report.ok:
        raise ValueError(f"the case is not valid:\n{report}")
    case_io.write_tables(os.path.join(case_dir, "data"), tables["network"], tables["actions"], tables["transitions"],
                         tables["nextnode"], tables["agents"])
    if write_raw:
        case_io.write_raw(os.path.join(case_dir, "raw"), raw)
    info = dict(settings=dict(merge=merge, threshold=threshold, parallel=parallel, routing=routing, excess=excess,
                              population=None if population is None else population.settings()),
                merge=tables["merge_report"], validation=dict(stats=report.stats, warnings=report.warnings),
                raw_sha256={name: _sha256(os.path.join(case_dir, "raw", name)) for name in ("nodes.csv", "links.csv")})
    info["versions"] = _versions()
    info.update(provenance or {})
    case_io.write_provenance(case_dir, info)
    return tables, report


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _versions():
    """Versions of what produced the tables (the random draw of the agents is reproducible with the same NumPy)."""
    import platform
    import scipy
    try:
        from importlib.metadata import version
        evacrl = version("evacrl")
    except Exception:  # not installed, run from a checkout
        evacrl = "source tree"
    out = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, evacrl=evacrl)
    for name in ("geopandas", "osmnx", "shapely"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out

