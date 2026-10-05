# -*- coding: utf-8 -*-
"""The shortest-path tables of the shipped cases with the tie-break of 0.2.0.

The first version of `evacrl.casebuild.routing.next_nodes` left the choice between equally short walks to SciPy's `dijkstra`,
which breaks ties differently in different versions: the clean-checkout audit (Step 5) rebuilt `kochi_area4` on Python 3.10
(SciPy 1.15) and got 1 entry in 2,220 that differs from the shipped table (made with SciPy 1.18). Now the next node is the one with
the lowest number among those on a shortest walk. This script rewrites `nextnode.csv` of the shipped cases, after checking for every entry
that changes that the old and the new step both lie on a shortest walk (so that no walk gets longer), and records it in `provenance.json`.

    python regenerate_nextnode.py           # report only
    python regenerate_nextnode.py --write   # and rewrite the files
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from evacrl.casebuild import read_tables  # noqa: E402
from evacrl.casebuild.case import write_provenance  # noqa: E402
from evacrl.casebuild.network import Network  # noqa: E402
from evacrl.casebuild.routing import _adjacency, distance_to_shelter, next_nodes  # noqa: E402

RULE = "0.2.0: the next node is the lowest-numbered neighbour on a shortest walk (before: left to SciPy's tie-breaking)"


def main(write):
    for name in ("kochi_area0", "kochi_area1", "kochi_area2", "kochi_area4"):
        case = os.path.join(REPO, "cases", name)
        with open(os.path.join(case, "provenance.json"), encoding="utf-8") as f:
            info = json.load(f)
        s = info["settings"]
        t = read_tables(os.path.join(case, "data"))
        net = Network(t["nodes"], t["links"])
        old, new = t["nextnode"], next_nodes(net, s["routing"], s["parallel"])
        dist = distance_to_shelter(net, s["parallel"])
        length = _adjacency(net, s["parallel"]).tolil()
        changed = np.where(old[:, 1] != new[:, 1])[0]
        for node in changed:
            for step in (old[node, 1], new[node, 1]):
                assert np.isclose(dist[step] + length[node, step], dist[node]), (name, node, step)   # both are on a shortest walk
        print(f"{name}: {len(old)} nodes, {len(changed)} entries change (all between equally short steps), "
              f"{int((new[:, 1] < 0).sum())} nodes without a path")
        if write and len(changed):
            np.savetxt(os.path.join(case, "data", "nextnode.csv"), new, delimiter=",", fmt="%d", header="node,next node")
            info["nextnode_rule"] = RULE
            write_provenance(case, info)


if __name__ == "__main__":
    main("--write" in sys.argv)
