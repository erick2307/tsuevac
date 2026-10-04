# -*- coding: utf-8 -*-
"""A2-A4. The geospatial stage from the 2024 study's stored inputs (needs `pip install -e ".[casebuild]"`; works offline).

A2  the stored OSM graph of an area, made into a network again, is the network of its committed raw tables (by OSM id)
A3  shelter points snapped to the nodes of that network give the committed shelter nodes; how far the snapping moves them
A4  the census population of each area, from the mesh: "within" (the study) and "weighted" (boundary cells by area)

    python geo_checks.py
"""
import collections

import numpy as np
import pandas as pd
from common import DATA, RESULTS
from evacrl.casebuild import geo
from evacrl.tables import load_table

AREAS = ("kochi0", "kochi1", "kochi2", "kochi3", "kochi4", "kochi42")

print("A2/A3  graph snapshot -> raw network, shelters\n")
points = geo.read_points([f"{DATA}/kochi_tsunami_evacbldg_crs4326.geojson", f"{DATA}/kochi_tsunami_shelters_crs4326.geojson"])
print(f"{'case':8s} {'nodes':>5s} {'links':>5s} {'same graph*':>11s} | {'points in box':>13s} {'shelters':>8s} {'same nodes':>10s} {'snap distance m: median / max':>31s} {'> 100 m':>8s}")
for c in AREAS:
    d = f"{RESULTS}/{c}"
    nodes0, edges0 = load_table(f"{d}/nodes0.csv"), load_table(f"{d}/edges0.csv")
    committed = pd.read_csv(f"{d}/original_nodes.csv", index_col=0)["0"].astype(np.int64).values
    graph = geo.graph_from_snapshot(f"{d}/Graph")
    raw, info = geo.raw_network(graph, points, within="bbox")
    pos = {int(o): i for i, o in enumerate(raw.osmid)}
    same_nodes = set(pos) == set(int(o) for o in committed)
    order = np.array([pos[int(o)] for o in committed])
    same_xy = np.abs(raw.nodes[order, 1:3] - nodes0[:, 1:3]).max() < 1e-3
    key = lambda ids, ends: collections.Counter((min(ids[int(a)], ids[int(b)]), max(ids[int(a)], ids[int(b)]), int(l)) for a, b, l in ends)
    same_links = key([int(o) for o in raw.osmid], raw.links[:, 1:4]) == key([int(o) for o in committed], edges0[:, 1:4])
    shelters_ours = set(int(raw.osmid[n]) for n in raw.shelters)
    shelters_theirs = set(int(o) for o in committed[nodes0[:, 3] == 1])
    med = f"{info['snap_distance_median']} / {info['snap_distance_max']}" if "snap_distance_median" in info else "-"
    print(f"{c:8s} {raw.num_nodes:5d} {raw.num_links:5d} {str(same_nodes and same_xy and same_links):>11s} | {info['points_used']:13d} "
          f"{len(shelters_ours):8d} {str(shelters_ours == shelters_theirs):>10s} {med:>31s} {info.get('snapped_over_100_m', 0):8d}")
print("* same OSM nodes, same coordinates (to 1 mm), same links with the same lengths (as a graph: the numbering differs)")

print("\nA4  census population of each area\n")
areas = geo.read_geojson(f"{DATA}/kochi-shi_tsunamievac_areas_crs4326.geojson")
census = geo.read_geojson(f"{DATA}/kochi-shi_census_crs4326.geojson")
study = {0: 2303, 1: 1078, 2: 622, 3: 240, 4: 12288}   # printed by the study's notebook
file_rows = {}
for c in ("kochi0", "kochi1", "kochi2", "kochi4", "kochi42"):
    with open(f"{RESULTS}/{c}/population_1.csv") as f:
        file_rows[c] = sum(1 for _ in f) - 1
print(f"{'area':5s} {'notebook':>9s} {'within':>7s} {'same':>5s} {'weighted':>9s} {'more':>7s}   rows of population_1.csv")
for i, g in enumerate(areas.geometry):
    polygon = geo.largest_polygon(g)
    within = geo.area_population(census, polygon, method="within")
    weighted = geo.area_population(census, polygon, method="weighted")
    rows = file_rows.get(f"kochi{i}", "-")
    print(f"{i:5d} {study[i]:9d} {within:7d} {str(within == study[i]):>5s} {weighted:9.0f} {100 * (weighted / within - 1):6.0f}%   {rows}")
print(f"kochi42 has {file_rows['kochi42']} agents")
