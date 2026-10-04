# -*- coding: utf-8 -*-
"""What needs OpenStreetMap and GeoPandas: the network of an area, the shelters on it, the people in it.

    pip install -e ".[casebuild]"      (osmnx < 2, geopandas, shapely)

Everything here is optional machinery around the core of `evacrl.casebuild`: a network that was downloaded once is kept as a
snapshot (`graph_to_snapshot`) and rebuilt from it (`graph_from_snapshot`), and from there the tables are made offline.
"""
import json
import os

import numpy as np

from evacrl.casebuild.network import Network, attach_shelters, network_from_edges

_HINT = 'pip install -e ".[casebuild]"'


def _require():
    try:
        import geopandas as gpd
        import osmnx as ox
        import shapely
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise ImportError(f"evacrl.casebuild.geo needs osmnx (< 2) and geopandas: {_HINT}") from exc
    return gpd, ox, shapely


# ------------------------------------------------------------------------------------------------------- reading
def read_geojson(path, crs="EPSG:4326"):
    """A GeoJSON file as a GeoDataFrame, read with `json` (GDAL's reader fails on some property values that
    OSMnx writes). `crs` is used when the file does not name one."""
    gpd, _, _ = _require()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    named = (data.get("crs") or {}).get("properties", {}).get("name")
    if named and "CRS84" in named:
        named = "EPSG:4326"
    return gpd.GeoDataFrame.from_features(data["features"], crs=named or crs)


def largest_polygon(geometry):
    """A Polygon from a Polygon or MultiPolygon (the largest part, as the 2024 study did)."""
    _, _, shapely = _require()
    merged = shapely.ops.unary_union(geometry)
    if merged.geom_type == "Polygon":
        return merged
    if merged.geom_type == "MultiPolygon":
        return max(merged.geoms, key=lambda p: p.area)
    raise ValueError(f"expected an area, got {merged.geom_type}")


# ------------------------------------------------------------------------------------------------------- network
def download_network(polygon, network_type="all"):
    """The street network inside `polygon` (WGS84), from OpenStreetMap: a directed OSMnx graph, simplified."""
    _, ox, _ = _require()
    return ox.graph_from_polygon(polygon, network_type=network_type, simplify=True)


def project_undirected(graph):
    """The graph in a metric CRS (the UTM zone of its centre) and with the two directions of a street merged."""
    _, ox, _ = _require()
    return ox.project_graph(graph).to_undirected()


def network_from_graph(graph, evacuation_osmids=(), length_rounding="floor"):
    """The raw `Network` of a projected, undirected OSMnx graph (nodes numbered in the order of `graph.nodes`)."""
    osmid = list(graph.nodes)
    xy = [(d["x"], d["y"]) for _, d in graph.nodes(data=True)]
    edges = [(u, v, d["length"]) for u, v, d in graph.edges(data=True)]
    crs = graph.graph.get("crs")
    return network_from_edges(osmid, xy, edges, evacuation_osmids, length_rounding=length_rounding,
                              crs=None if crs is None else str(crs))


def graph_to_snapshot(graph, folder):
    """Keep a downloaded (directed, WGS84) graph: `Gnodes.geojson` and `Gedges.geojson`, the format of the 2024 study."""
    gpd, ox, _ = _require()
    os.makedirs(folder, exist_ok=True)
    nodes, edges = ox.graph_to_gdfs(graph)
    keep_nodes = [c for c in ("street_count", "highway") if c in nodes.columns]
    nodes = nodes[keep_nodes + ["x", "y", "geometry"]].reset_index()
    edges = edges.reset_index()
    for column in edges.columns:
        if column not in ("u", "v", "key", "length", "geometry") and edges[column].map(lambda v: isinstance(v, (list, tuple, dict))).any():
            edges[column] = edges[column].map(lambda v: str(v) if isinstance(v, (list, tuple, dict)) else v)
    for name, frame in (("Gnodes", nodes), ("Gedges", edges)):
        with open(os.path.join(folder, f"{name}.geojson"), "w", encoding="utf-8") as f:
            f.write(frame.to_json(na="null", drop_id=True))


def graph_from_snapshot(folder):
    """The directed WGS84 graph kept by `graph_to_snapshot` (or by the 2024 study's `download_nwk`)."""
    _, ox, _ = _require()
    nodes = read_geojson(os.path.join(folder, "Gnodes.geojson")).set_index("osmid")
    edges = read_geojson(os.path.join(folder, "Gedges.geojson")).set_index(["u", "v", "key"])
    return ox.graph_from_gdfs(nodes, edges)


# ------------------------------------------------------------------------------------------------------ shelters
def snap_to_nodes(network, points, within=None, max_distance=None):
    """For each point, the number of the nearest node of `network`.

    network      a `Network` with `crs` set (its x, y are metres in that CRS)
    points       GeoDataFrame of points, in any CRS
    within       None: every point; "bbox": only points inside the bounding box of the network (what the 2024 study did,
                 so a shelter just outside the area is snapped to the nearest node at its edge)
    max_distance metres; points farther than this from every node are left out

    Returns `(nodes, distances, kept)`: the node of every point that was kept, the distance to it, and the positions
    (in `points`) of those points.
    """
    gpd, _, _ = _require()
    from scipy.spatial import cKDTree
    if network.crs is None:
        raise ValueError("the network has no CRS; its coordinates cannot be compared with the points")
    p = points.to_crs(network.crs)
    xy = np.column_stack([p.geometry.x.values, p.geometry.y.values])
    keep = np.ones(len(xy), dtype=bool)
    if within == "bbox":
        # the box of the network in the CRS of the points (longitude, latitude for the Kochi data), not the box in metres
        nodes = gpd.GeoSeries(gpd.points_from_xy(network.nodes[:, 1], network.nodes[:, 2]), crs=network.crs).to_crs(points.crs)
        x0, y0, x1, y1 = nodes.total_bounds
        raw = points.geometry
        keep &= ((raw.x >= x0) & (raw.x <= x1) & (raw.y >= y0) & (raw.y <= y1)).values
    elif within is not None:
        raise ValueError("within must be None or 'bbox'")
    dist, node = cKDTree(network.nodes[:, 1:3]).query(xy)
    if max_distance is not None:
        keep &= dist <= max_distance
    kept = np.where(keep)[0]
    return node[kept], dist[kept], kept


def mark_shelters(network, node_numbers):
    """A copy of `network` with these nodes as shelters."""
    out = network.copy()
    for node in np.unique(node_numbers):
        out.nodes[int(node), 3] = 1
        out.nodes[int(node), 4] = 1000
    return out


# ------------------------------------------------------------------------------------------------------ population
def area_population(mesh, area, column="M_TOTPOP_H", method="within"):
    """People in `area` (a polygon in the CRS of the mesh) according to a census mesh (GeoDataFrame of cells).

    method="within"    the cells that lie entirely inside the area (what the 2024 study did: the cells on the boundary
                       are lost, so the total is too low)
    method="weighted"  every cell that touches the area, in proportion to the part of it that lies inside
    """
    gpd, _, _ = _require()
    if method == "within":
        return int(mesh[mesh.geometry.within(area)][column].sum())
    if method != "weighted":
        raise ValueError("method must be 'within' or 'weighted'")
    touching = mesh[mesh.geometry.intersects(area)]
    projected = gpd.GeoSeries(touching.geometry).to_crs(gpd.GeoSeries([area], crs=mesh.crs).estimate_utm_crs())
    clipped = projected.intersection(gpd.GeoSeries([area], crs=mesh.crs).to_crs(projected.crs).iloc[0])
    share = (clipped.area / projected.area).values
    return float((touching[column].values * share).sum())


def node_weights(network, mesh, area, column="M_TOTPOP_H", method="within", exclude_shelters=True):
    """The census population of the area (a polygon in the CRS of `mesh`) spread over the nodes of `network`: `(weights, info)`.

    Each cell's people are shared equally by the nodes inside it (a cell on the boundary: inside the part within the
    area, and `method="weighted"` counts only that part of its people). A cell with people but no node goes to the
    node nearest its centre. `weights.sum()` is the number of people placed.

    exclude_shelters   shelters are not places where people start (the default of `PopulationSpec`), so they get no
                       share: a cell's people go to its other nodes, and to the nearest one that is not a shelter if
                       it has none. Without this, the people of a cell would be lost to the shelter nodes in it (and a
                       cell holding only an attached shelter would lose them all).
    """
    gpd, _, _ = _require()
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    if network.crs is None:
        raise ValueError("the network has no CRS")
    cells = mesh.to_crs(network.crs)
    boundary = gpd.GeoSeries([area], crs=mesh.crs).to_crs(network.crs).iloc[0]
    if method == "within":
        cells = cells[cells.geometry.within(boundary)]
        people = cells[column].values.astype(float)
        clipped = cells.geometry
    elif method == "weighted":
        cells = cells[cells.geometry.intersects(boundary)]
        clipped = cells.geometry.intersection(boundary)
        people = cells[column].values.astype(float) * (clipped.area / cells.geometry.area).values
    else:
        raise ValueError("method must be 'within' or 'weighted'")
    usable = np.where(network.nodes[:, 3] != 1)[0] if exclude_shelters else np.arange(network.num_nodes)
    weights = np.zeros(network.num_nodes)
    if len(usable) == 0:
        return weights, dict(method=method, cells=int((people > 0).sum()), cells_without_a_node=0, people=0.0)
    xy = network.nodes[usable, 1:3]
    tree = cKDTree(xy)
    empty = 0
    points = gpd.GeoSeries([Point(p) for p in xy])
    index = points.sindex
    for geometry, n in zip(clipped.values, people):
        if n <= 0:
            continue
        inside = index.query(geometry, predicate="intersects")
        if len(inside):
            weights[usable[inside]] += n / len(inside)
        else:
            weights[usable[tree.query([geometry.centroid.x, geometry.centroid.y])[1]]] += n
            empty += 1
    return weights, dict(method=method, cells=int((people > 0).sum()), cells_without_a_node=empty,
                         people=float(people.sum()))


def read_points(paths):
    """The point layers of several GeoJSON files (shelters, evacuation buildings) as one GeoDataFrame in WGS84."""
    gpd, _, _ = _require()
    import pandas as pd
    layers = [read_geojson(p).to_crs("EPSG:4326")[["geometry"]] for p in paths]
    return gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), crs="EPSG:4326")


def raw_network(graph, shelter_points, within="bbox", max_distance=None, length_rounding="floor", shelters="attach"):
    """The raw `Network` of a downloaded (directed, WGS84) graph with its shelters: `(network, info)`.

    The points of `shelter_points` that lie inside the box of the network (`within`) and not farther than `max_distance`
    from a node become shelters in one of two ways (`shelters`):

    "attach"  each shelter is a node of its own, at the point, joined to the nearest street node by a link as long as
              the distance between them (see `attach_shelters`): the walk to the shelter counts, and no shelter is
              moved onto the edge of the network
    "snap"    each shelter is its nearest node (what the 2024 study did): a shelter far from every node is placed
              on the edge of the network, and the walk from there to the building is not part of the evacuation

    `info` says what was done, in particular how far the shelters are from the network."""
    if shelters not in ("attach", "snap"):
        raise ValueError("shelters must be 'attach' or 'snap'")
    net = network_from_graph(project_undirected(graph), [], length_rounding=length_rounding)
    node, dist, kept = snap_to_nodes(net, shelter_points, within=within, max_distance=max_distance)
    info = dict(mode=shelters, points=len(shelter_points), points_used=len(kept), within=within, max_distance=max_distance)
    if shelters == "snap":
        info["shelter_nodes"] = len(set(node.tolist()))
        if len(dist):
            info.update(snap_distance_median=round(float(np.median(dist)), 1), snap_distance_max=round(float(dist.max()), 1),
                        snapped_over_100_m=int((dist > 100).sum()))
        return mark_shelters(net, node), info
    xy = shelter_points.iloc[kept].to_crs(net.crs).geometry
    attached, counts = attach_shelters(net, np.column_stack([xy.x.values, xy.y.values]), length_rounding=length_rounding)
    counts.pop("points")
    info.update(shelter_nodes=counts.pop("shelters"), **counts)
    return attached, info
