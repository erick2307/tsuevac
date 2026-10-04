# -*- coding: utf-8 -*-
"""A6. The validator on the tables of the 2024 study, and what the clean-up changes. Core only (NumPy, SciPy).

    python validate_committed.py
"""
import numpy as np
from common import RESULTS, CASES
from evacrl.casebuild import Network, merge_short_links, next_nodes, validate_tables
from evacrl.tables import load_table

print("The validator on the committed tables of the 2024 study\n")
for c in CASES:
    d = f"{RESULTS}/{c}"
    nodes, links = load_table(f"{d}/nodes.csv"), load_table(f"{d}/edges.csv")
    r = validate_tables(nodes, links, load_table(f"{d}/actionsdb.csv", dtype=int), load_table(f"{d}/transitionsdb.csv", dtype=int),
                        load_table(f"{d}/nextnode.csv", dtype=int), load_table(f"{d}/population_1.csv", dtype=int))
    print(f"{c}: {r.stats}")
    for line in r.errors:
        print(f"   ERROR   {line}")
    for line in r.warnings:
        print(f"   warning {line}")

print("\nThe clean-up: the 2024 method (pairs) against clusters, from the same raw network\n")
print(f"{'case':8s} {'short links':>11s} | {'nodes: legacy':>13s} {'clusters':>9s} | {'orphan rows':>11s} | {'chains (3+ nodes)':>17s} | {'nodes without a way out: legacy':>32s} {'clusters':>9s}")
for c in CASES:
    d = f"{RESULTS}/{c}"
    raw = Network(load_table(f"{d}/nodes0.csv"), load_table(f"{d}/edges0.csv"))
    old, r_old = merge_short_links(raw, method="legacy")
    new, r_new = merge_short_links(raw)
    stuck = lambda net: int((next_nodes(net, parallel="last")[:, 1] == -9999).sum())
    print(f"{c:8s} {r_old['short_links']:11d} | {old.num_nodes:13d} {new.num_nodes:9d} | {r_old['orphan_rows']:11d} | {r_new['clusters_of_3_or_more']:17d} | "
          f"{stuck(old):32d} {stuck(new):9d}")
