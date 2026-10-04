# -*- coding: utf-8 -*-
"""A node of the model holds at most 10 links. cases/new_kochi had one node (3106) with 11: setActionsAndTransitions.py used to
cut the extra link from the table of that node only (its comment: "Not the best solution", Aug 2021), so the link was missing
at one end and an agent could walk it one way. This removes the longest link at every node with more than 10 links, from
both ends (`evacrl.casebuild.prune_excess_links`), renumbers the links, and rebuilds actionsdb.csv / transitionsdb.csv.

    cd cases/new_kochi && python fixExcessLinks.py          (after fixLinksDBAndNodesDB of preProcess.py; changes nothing if no node has more than 10 links)

Needs `pip install -e .` and SciPy. Done once on the shipped tables: node 3106, link 4779 (22 m, to node 4588, which keeps its other link).
"""
import numpy as np

from evacrl.casebuild import Network, prune_excess_links
from setActionsAndTransitions import setMatrices


def fixExcessLinks():
    nodes = np.loadtxt("./data/nodesdb.csv", delimiter=",", skiprows=1)
    links = np.loadtxt("./data/linksdb.csv", delimiter=",", skiprows=1)
    net, removed = prune_excess_links(Network(nodes, links))
    for k in removed:
        print("removed link", int(links[k, 0]), "between nodes", int(links[k, 1]), "and", int(links[k, 2]), "length", int(links[k, 3]))
    if removed:
        np.savetxt("./data/linksdb.csv", net.links, delimiter=",", header="number,node1,node2,length,width", fmt="%d,%d,%d,%d,%d")
    setMatrices()
    return removed


if __name__ == "__main__":
    fixExcessLinks()
