# -*- coding: utf-8 -*-
"""A1. In legacy mode the core of evacrl.casebuild reproduces the tables of the 2024 study, stage by stage; and its actions /
transitions reproduce those of tsuevac's own older cases (made by a different script). Core only: NumPy and SciPy.

    python legacy_exactness.py
"""
import numpy as np
from common import REPO, RESULTS, CASES
from evacrl.casebuild import (Network, actions_and_transitions, distance_to_shelter, merge_short_links, next_nodes)
from evacrl.tables import load_table

print("The 2024 study's tables from its own raw tables (nodes0.csv, edges0.csv), stage by stage\n")
print(f"{'case':8s} {'merge (legacy)':>15s} {'actionsdb':>10s} {'transitionsdb':>14s} {'nextnode (all pairs, last link)':>32s} {'nextnode (multi-source)':>24s}")
for c in CASES:
    d = f"{RESULTS}/{c}"
    raw = Network(load_table(f"{d}/nodes0.csv"), load_table(f"{d}/edges0.csv"))
    merged, report = merge_short_links(raw, 5, method="legacy")
    nodes, links = load_table(f"{d}/nodes.csv"), load_table(f"{d}/edges.csv")
    merge_ok = np.allclose(merged.nodes, nodes, atol=5e-7) and np.array_equal(merged.links, links)
    committed = Network(nodes, links)
    a, t = actions_and_transitions(committed)
    a_ok = np.array_equal(a, load_table(f"{d}/actionsdb.csv", dtype=int))
    t_ok = np.array_equal(t, load_table(f"{d}/transitionsdb.csv", dtype=int))
    nn = load_table(f"{d}/nextnode.csv", dtype=int)
    ap_ok = np.array_equal(next_nodes(committed, "allpairs", "last"), nn)
    ms_ok = np.array_equal(next_nodes(committed, "nearest", "last"), nn)
    print(f"{c:8s} {str(merge_ok):>15s} {str(a_ok):>10s} {str(t_ok):>14s} {str(ap_ok):>32s} {str(ms_ok):>24s}")

print("\ntsuevac's older cases: actions / transitions made by the old script vs by evacrl.casebuild\n")
for c in ("kochi", "new_kochi"):
    d = f"{REPO}/cases/{c}/data"
    nodes, links = load_table(f"{d}/nodesdb.csv"), load_table(f"{d}/linksdb.csv", dtype=int)
    A, T = load_table(f"{d}/actionsdb.csv", dtype=int), load_table(f"{d}/transitionsdb.csv", dtype=int)
    degree = np.bincount(np.concatenate([links[:, 1], links[:, 2]]), minlength=len(nodes))
    try:
        a, t = actions_and_transitions(Network(nodes, links))
        print(f"{c:10s} identical: {np.array_equal(a, A) and np.array_equal(t, T)}")
    except ValueError as e:
        print(f"{c:10s} {e}; the committed table keeps the first 10 and drops the rest "
              f"(links per node: max {degree.max()}, nodes above 10: {int((degree > 10).sum())})")
