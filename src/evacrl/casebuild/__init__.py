# -*- coding: utf-8 -*-
"""Building a case: from a road network and the places that are safe, to the tables the model runs on.

The core (this package without `geo`) needs only NumPy and SciPy:

    network_from_edges   raw network from nodes and edges
    merge_short_links    remove links that are too short, merge the nodes they join
    actions_and_transitions, next_nodes, start_nodes ...   the other tables
    validate_case        check a case folder

`evacrl.casebuild.geo` adds what needs OpenStreetMap and GeoPandas (`pip install -e ".[casebuild]"`): downloading the
network of an area, snapping shelters to it, and counting people in census cells. `evacrl.casebuild.pipeline` does
the whole sequence. See docs/manual.md, "Building a case".
"""
try:
    import scipy  # noqa: F401  (the clean-up, the shortest paths and the validator use it)
except ImportError as exc:  # pragma: no cover - depends on the installation
    raise ImportError('evacrl.casebuild needs SciPy: pip install scipy  (or pip install -e ".[casebuild]")') from exc

from evacrl.casebuild.actions import MAX_ACTIONS, actions_and_transitions, prune_excess_links
from evacrl.casebuild.case import (Report, read_raw, read_tables, validate_case, validate_tables, write_provenance,
                                   write_raw, write_tables)
from evacrl.casebuild.network import Network, merge_short_links, network_from_edges
from evacrl.casebuild.population import (agents_table, apportion, candidate_nodes, start_nodes, start_nodes_per_node,
                                         start_nodes_proportional)
from evacrl.casebuild.routing import NO_PATH, distance_to_shelter, next_nodes
from evacrl.casebuild.pipeline import LEGACY, PopulationSpec, build_case, build_tables

__all__ = [
    "LEGACY", "MAX_ACTIONS", "NO_PATH", "Network", "PopulationSpec", "Report", "build_case", "build_tables", "actions_and_transitions", "agents_table", "apportion",
    "candidate_nodes", "distance_to_shelter", "merge_short_links", "network_from_edges", "next_nodes", "prune_excess_links", "read_raw",
    "read_tables", "start_nodes", "start_nodes_per_node", "start_nodes_proportional", "validate_case",
    "validate_tables", "write_provenance", "write_raw", "write_tables",
]
