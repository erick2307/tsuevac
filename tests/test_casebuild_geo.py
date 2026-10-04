#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evacrl.casebuild.geo on small synthetic data (needs osmnx < 2 and geopandas: pip install -e ".[casebuild]").

The street network is five nodes near Kochi, about 100 m apart, every street in both directions:

      3
      |
  0 - 1 - 2 - 4

Run: python -m unittest discover tests
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from evacrl.casebuild import geo
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"evacrl.casebuild.geo needs the 'casebuild' extra ({exc}); pip install -e \".[casebuild]\"")

from evacrl.casebuild import PopulationSpec, build_tables, merge_short_links, validate_tables  # noqa: E402

LON0, LAT0 = 133.53, 33.56
DLON, DLAT = 0.0011, 0.0009   # about 100 m
POSITION = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (1, 1), 4: (3, 0)}
OSMID = {0: 100, 1: 101, 2: 102, 3: 103, 4: 104}
STREETS = [(0, 1), (1, 2), (1, 3), (2, 4)]


def make_graph():
    g = nx.MultiDiGraph(crs="EPSG:4326")
    for i, (cx, cy) in POSITION.items():
        g.add_node(OSMID[i], x=LON0 + cx * DLON, y=LAT0 + cy * DLAT, street_count=2)
    for a, b in STREETS:
        pa, pb = POSITION[a], POSITION[b]
        length = float(np.hypot((pa[0] - pb[0]) * DLON * 92500, (pa[1] - pb[1]) * DLAT * 111000))
        for u, v in ((a, b), (b, a)):
            g.add_edge(OSMID[u], OSMID[v], 0, osmid=900 + a, highway="residential", length=length, oneway=False)
    return g


class Graphs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.graph = make_graph()

    def tearDown(self):
        self._tmp.cleanup()

    def test_projecting_gives_metres_and_one_edge_per_street(self):
        g = geo.project_undirected(self.graph)
        self.assertEqual(str(g.graph["crs"]), "EPSG:32653")
        self.assertFalse(g.is_directed())
        self.assertEqual(g.number_of_edges(), len(STREETS))
        x = [d["x"] for _, d in g.nodes(data=True)]
        self.assertGreater(min(x), 3e5)   # UTM easting, not degrees

    def test_the_raw_network_of_a_graph(self):
        g = geo.project_undirected(self.graph)
        net = geo.network_from_graph(g, evacuation_osmids=[OSMID[4]])
        self.assertEqual(net.crs, "EPSG:32653")
        self.assertEqual(net.num_nodes, 5)
        self.assertEqual(net.num_links, 4)
        self.assertEqual(net.shelters.tolist(), [list(g.nodes).index(OSMID[4])])
        self.assertEqual(sorted(net.osmid.tolist()), sorted(OSMID.values()))
        self.assertTrue((net.links[:, 3] >= 90).all() and (net.links[:, 3] <= 110).all())

    def test_a_snapshot_rebuilds_the_graph(self):
        geo.graph_to_snapshot(self.graph, self.dir / "Graph")
        back = geo.graph_from_snapshot(self.dir / "Graph")
        self.assertEqual(sorted(back.nodes), sorted(self.graph.nodes))
        self.assertEqual(back.number_of_edges(), self.graph.number_of_edges())
        mine = sorted(d["length"] for _, _, d in self.graph.edges(data=True))
        theirs = sorted(d["length"] for _, _, d in back.edges(data=True))
        np.testing.assert_allclose(theirs, mine, rtol=1e-6)
        a = geo.network_from_graph(geo.project_undirected(self.graph), [OSMID[4]])
        b = geo.network_from_graph(geo.project_undirected(back), [OSMID[4]])
        self.assertEqual(sorted(a.links[:, 3]), sorted(b.links[:, 3]))

    def test_a_snapshot_with_list_valued_attributes_is_written(self):
        self.graph.edges[OSMID[0], OSMID[1], 0]["osmid"] = [1, 2]    # OSMnx merges streets into lists when it simplifies
        geo.graph_to_snapshot(self.graph, self.dir / "Graph")
        self.assertEqual(geo.graph_from_snapshot(self.dir / "Graph").number_of_edges(), self.graph.number_of_edges())

    def test_read_geojson_uses_the_crs_the_file_names(self):
        path = self.dir / "x.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                                    "features": [{"type": "Feature", "properties": {"a": 1}, "geometry": {"type": "Point", "coordinates": [133.5, 33.5]}}]}))
        g = geo.read_geojson(path)
        self.assertEqual(g.crs.to_epsg(), 4326)
        self.assertEqual(g.geometry.iloc[0].x, 133.5)

    def test_the_largest_part_of_a_multipolygon(self):
        small, big = box(0, 0, 1, 1), box(5, 5, 8, 8)
        self.assertEqual(geo.largest_polygon(MultiPolygon([small, big])).area, 9)
        self.assertEqual(geo.largest_polygon(small).area, 1)
        with self.assertRaises(ValueError):
            geo.largest_polygon(Point(0, 0))


class Shelters(unittest.TestCase):
    def setUp(self):
        self.net = geo.network_from_graph(geo.project_undirected(make_graph()), [])
        self.number = {int(o): i for i, o in enumerate(self.net.osmid)}

    def points(self, coords):
        return gpd.GeoDataFrame(geometry=[Point(LON0 + x * DLON, LAT0 + y * DLAT) for x, y in coords], crs="EPSG:4326")

    def test_each_point_goes_to_the_nearest_node(self):
        node, dist, kept = geo.snap_to_nodes(self.net, self.points([(0.1, 0.1), (2.9, 0.05), (1.0, 0.8)]))
        self.assertEqual([int(self.net.osmid[n]) for n in node], [100, 104, 103])
        self.assertEqual(kept.tolist(), [0, 1, 2])
        self.assertTrue((dist < 40).all())

    def test_only_points_inside_the_box_of_the_network_if_asked(self):
        pts = self.points([(1, 0.5), (10, 10), (-5, 0)])
        _, _, kept = geo.snap_to_nodes(self.net, pts)
        self.assertEqual(kept.tolist(), [0, 1, 2])
        node, _, kept = geo.snap_to_nodes(self.net, pts, within="bbox")
        self.assertEqual(kept.tolist(), [0])

    def test_points_too_far_from_every_node_can_be_left_out(self):
        pts = self.points([(1.0, 0.05), (0.0, 0.9)])   # 5 m and 90 m from the nearest node (node 1, node 0)
        _, dist, kept = geo.snap_to_nodes(self.net, pts, max_distance=50)
        self.assertEqual(kept.tolist(), [0])
        self.assertLess(dist[0], 50)

    def test_the_network_needs_a_crs_and_the_options_are_checked(self):
        pts = self.points([(0, 0)])
        bare = self.net.copy()
        bare.crs = None
        with self.assertRaises(ValueError):
            geo.snap_to_nodes(bare, pts)
        with self.assertRaises(ValueError):
            geo.snap_to_nodes(self.net, pts, within="polygon")

    def test_the_raw_network_uses_the_shelters_inside_the_box_unless_told_otherwise(self):
        pts = self.points([(3.0, 0.2), (-11, 0)])      # near node 4; far outside the network, on the side of node 0
        net, info = geo.raw_network(make_graph(), pts)
        self.assertEqual((info["points_used"], info["shelter_nodes"], info["within"]), (1, 1, "bbox"))
        self.assertEqual(len(net.shelters), 1)
        net, info = geo.raw_network(make_graph(), pts, within=None)
        self.assertEqual((info["points_used"], info["shelter_nodes"]), (2, 2))   # the far one lands on the nearest node (0), 1.1 km away
        self.assertGreater(info["snap_distance_max"], 1000)
        self.assertEqual(info["snapped_over_100_m"], 1)

    def test_marking_shelters(self):
        marked = geo.mark_shelters(self.net, [self.number[104], self.number[104], self.number[100]])
        self.assertEqual(sorted(marked.shelters.tolist()), sorted([self.number[104], self.number[100]]))
        self.assertEqual(self.net.shelters.tolist(), [])   # the original is untouched


class CommandLine(unittest.TestCase):
    """`from-snapshot` with a stored graph, shelter points, an area and a census mesh."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        geo.graph_to_snapshot(make_graph(), self.dir / "Graph")
        gpd.GeoDataFrame(geometry=[Point(LON0 + 3.0 * DLON, LAT0 + 0.2 * DLAT)], crs="EPSG:4326").to_file(self.dir / "shelters.geojson", driver="GeoJSON")
        area = box(LON0 - 0.0013, LAT0 - 0.0013, LON0 + 0.0053, LAT0 + 0.0023)   # a margin: no cell edge lies on its boundary
        gpd.GeoDataFrame({"id": [0]}, geometry=[area], crs="EPSG:4326").to_file(self.dir / "areas.geojson", driver="GeoJSON")
        cells = [box(LON0 + i * 0.002 - 0.001, LAT0 - 0.001, LON0 + (i + 1) * 0.002 - 0.001, LAT0 + 0.002) for i in range(3)]
        gpd.GeoDataFrame({"M_TOTPOP_H": [10, 20, 30]}, geometry=cells, crs="EPSG:4326").to_file(self.dir / "census.geojson", driver="GeoJSON")

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *extra):
        import contextlib, io
        from evacrl.casebuild import cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["from-snapshot", str(self.dir / "Graph"), str(self.dir / "case"), "--shelters", str(self.dir / "shelters.geojson")]
                            + [str(e) for e in extra])
        return code, out.getvalue()

    def test_a_case_from_a_snapshot_and_a_shelter_file(self):
        code, out = self.run_cli("--agents", 25)
        self.assertEqual(code, 0, out)
        info = json.loads((self.dir / "case" / "provenance.json").read_text())
        self.assertEqual(info["shelters"]["shelter_nodes"], 1)
        self.assertEqual(info["shelters"]["within"], "bbox")
        self.assertEqual(list(info["shelters"]["files"]), ["shelters.geojson"])
        self.assertLess(info["shelters"]["snap_distance_max"], 30)
        self.assertEqual(len((self.dir / "case" / "data" / "agentsdb.csv").read_text().splitlines()), 26)   # header + 25

    def test_the_census_gives_the_number_of_agents(self):
        code, out = self.run_cli("--areas", self.dir / "areas.geojson", "--census", self.dir / "census.geojson", "--census-method", "within")
        self.assertEqual(code, 0, out)
        info = json.loads((self.dir / "case" / "provenance.json").read_text())
        self.assertEqual(info["census"]["method"], "within")
        self.assertEqual(info["census"]["agents"], int(round(info["census"]["area_total"])))
        self.assertGreater(info["census"]["agents"], 0)

    def test_the_areas_and_the_census_may_be_in_other_crs_than_the_network(self):
        wgs = self.run_cli("--areas", self.dir / "areas.geojson", "--census", self.dir / "census.geojson", "--census-method", "within")
        self.assertEqual(wgs[0], 0, wgs[1])
        expected = json.loads((self.dir / "case" / "provenance.json").read_text())["census"]["agents"]
        for name in ("areas", "census"):
            frame = gpd.read_file(self.dir / f"{name}.geojson").to_crs("EPSG:32653")
            frame.to_file(self.dir / f"{name}.gpkg", driver="GPKG")          # a projected layer: GeoJSON could not name its CRS
        # GeoJSON files naming a projected CRS, as older tools write them
        for name in ("areas", "census"):
            data = json.loads(gpd.read_file(self.dir / f"{name}.gpkg").to_json())
            data["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32653"}}
            (self.dir / f"{name}_utm.geojson").write_text(json.dumps(data))
        shutil.rmtree(self.dir / "case")
        code, out = self.run_cli("--areas", self.dir / "areas_utm.geojson", "--census", self.dir / "census_utm.geojson", "--census-method", "within")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.dir / "case" / "provenance.json").read_text())["census"]["agents"], expected)

    def test_proportional_needs_the_census(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--strategy", "proportional", "--agents", 10)

    def test_a_shelter_too_far_away_can_be_left_out(self):
        gpd.GeoDataFrame(geometry=[Point(LON0 + 3.0 * DLON, LAT0 + 0.9 * DLAT)], crs="EPSG:4326").to_file(self.dir / "shelters.geojson", driver="GeoJSON")
        with self.assertRaisesRegex(ValueError, "no node is an evacuation node"):
            self.run_cli("--agents", 5, "--max-snap-distance", 20)


class Census(unittest.TestCase):
    """A mesh of 2 x 2 cells of 0.01 degrees; the area covers the two cells of the left column and half of the right column."""

    def setUp(self):
        x0, y0, s = 133.50, 33.50, 0.01
        cells = [box(x0 + i * s, y0 + j * s, x0 + (i + 1) * s, y0 + (j + 1) * s) for j in range(2) for i in range(2)]
        self.mesh = gpd.GeoDataFrame({"M_TOTPOP_H": [100, 200, 300, 400]}, geometry=cells, crs="EPSG:4326")   # (0,0) (1,0) (0,1) (1,1)
        self.area = box(x0 - 0.001, y0 - 0.001, x0 + 0.015, y0 + 0.021)  # all of column 0, half of column 1, a margin around

    def test_within_counts_only_the_cells_inside(self):
        self.assertEqual(geo.area_population(self.mesh, self.area, method="within"), 100 + 300)

    def test_weighted_counts_the_part_of_each_cell_inside(self):
        # column 0: 100 + 300 whole; column 1: half of (200 + 400)
        self.assertAlmostEqual(geo.area_population(self.mesh, self.area, method="weighted"), 400 + 300, delta=1.0)

    def test_the_method_is_checked(self):
        with self.assertRaises(ValueError):
            geo.area_population(self.mesh, self.area, method="x")

    def network(self):
        # nodes: two in cell (0,0), one in cell (0,1), one in cell (1,0) (inside the area), none in cell (1,1)
        graph = nx.MultiDiGraph(crs="EPSG:4326")
        for osmid, (lon, lat) in {1: (133.502, 33.502), 2: (133.507, 33.505), 3: (133.504, 33.514), 4: (133.5121, 33.505)}.items():
            graph.add_node(osmid, x=lon, y=lat, street_count=1)
        graph.add_edge(1, 2, 0, length=50.0)
        graph.add_edge(2, 1, 0, length=50.0)
        return geo.network_from_graph(geo.project_undirected(graph), [])

    def test_a_cells_people_are_shared_by_the_nodes_inside_it(self):
        net = self.network()
        w, info = geo.node_weights(net, self.mesh, self.area, method="within")
        by_osm = {int(net.osmid[i]): w[i] for i in range(net.num_nodes)}
        self.assertEqual(by_osm, {1: 50.0, 2: 50.0, 3: 300.0, 4: 0.0})       # cell (0,0): 100 over nodes 1, 2; (0,1): 300 on node 3
        self.assertEqual(info["people"], 400)
        self.assertEqual(w.sum(), 400)

    def test_weighted_adds_the_boundary_cell_and_a_cell_without_nodes_goes_to_the_nearest(self):
        net = self.network()
        w, info = geo.node_weights(net, self.mesh, self.area, method="weighted")
        # cell (1,0): half inside, node 4 (lon 133.5121) lies inside that half: 100 of its 200 people; cell (1,1): half of 400,
        # no node in it -> the nearest node (3 or 4), 200
        self.assertAlmostEqual(w.sum(), geo.area_population(self.mesh, self.area, method="weighted"), delta=1.0)
        self.assertEqual(info["cells_without_a_node"], 1)
        by_osm = {int(net.osmid[i]): w[i] for i in range(net.num_nodes)}
        self.assertAlmostEqual(by_osm[1] + by_osm[2], 100, delta=0.5)

    def test_the_weights_drive_a_proportional_population(self):
        net = self.network()
        # make it a case: node 4 is the shelter, 3 and 1, 2 can reach it only if linked: link them all
        from evacrl.casebuild import Network
        links = [[0, 0, 1, 50, 3], [1, 1, 3, 90, 3], [2, 3, 2, 60, 3], [3, 2, 0, 70, 3]]   # node numbers in network order
        order = {int(o): i for i, o in enumerate(net.osmid)}
        links = [[k, order[a], order[b], l, 3] for k, (a, b, l) in enumerate([(1, 2, 50), (2, 3, 90), (3, 4, 60)])]
        raw = Network(net.nodes.copy(), links, osmid=net.osmid, crs=net.crs)
        raw.nodes[order[4], 3] = 1
        weights = lambda n: geo.node_weights(n, self.mesh, self.area, method="within")[0] + 1e-9
        t = build_tables(raw, population=PopulationSpec(strategy="proportional", total=400, weights=weights))
        self.assertEqual(len(t["agents"]), 400)
        self.assertTrue(validate_tables(t["network"].nodes, t["network"].links, t["actions"], t["transitions"],
                                        t["nextnode"], t["agents"]).ok)


if __name__ == "__main__":
    unittest.main()
