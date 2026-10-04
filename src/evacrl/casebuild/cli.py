# -*- coding: utf-8 -*-
"""Command line: build, rebuild and check cases.

    python -m evacrl.casebuild from-raw RAW_DIR CASE_DIR --agents N        rebuild offline from a folder written by an earlier build
    python -m evacrl.casebuild from-snapshot GRAPH_DIR CASE_DIR ...        from a stored OSM graph (Gnodes/Gedges.geojson) + shelters + census
    python -m evacrl.casebuild from-osm CASE_DIR --areas F --index I ...   the same, downloading the network of an area first
    python -m evacrl.casebuild validate CASE_DIR                           check the tables of a case

`from-snapshot` and `from-osm` need `pip install -e ".[casebuild]"`; `from-raw` and `validate` only NumPy and SciPy.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from evacrl.casebuild import case as case_io
from evacrl.casebuild.pipeline import LEGACY, PopulationSpec, build_case


def _population_args(p):
    g = p.add_argument_group("population")
    g.add_argument("--strategy", choices=("uniform", "per_node", "proportional"), default="uniform",
                   help="uniform: --agents at random start nodes (the 2024 study); per_node: --per-node at each; "
                        "proportional: --agents spread by the census (needs --census)")
    g.add_argument("--agents", type=int, help="number of agents (default with --census: the census population of the area)")
    g.add_argument("--per-node", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--include-shelters", action="store_true", help="agents may start at a shelter (the 2024 study did)")


def _clean_args(p):
    g = p.add_argument_group("clean-up")
    g.add_argument("--merge", choices=("clusters", "legacy"), default="clusters")
    g.add_argument("--threshold", type=float, default=5.0, help="links shorter than this (m) are merged away")
    g.add_argument("--excess", choices=("error", "prune"), default="error",
                   help="a node with more than 10 links: stop (default) or remove the longest links there")
    g.add_argument("--legacy", action="store_true", help="the 2024 study's choices throughout (merge, parallel links, snapped shelters, agents at shelters)")


def _geo_args(p):
    g = p.add_argument_group("area, shelters and census")
    g.add_argument("--areas", help="GeoJSON of the areas (needed to download, and for the census)")
    g.add_argument("--index", type=int, default=0, help="which area of --areas")
    g.add_argument("--shelters", nargs="+", required=True, help="GeoJSON point files of shelters / evacuation buildings")
    g.add_argument("--shelters-as", choices=("attach", "snap"), default="attach", dest="shelters_as",
                   help="attach (default): a node at each shelter, joined to the nearest street node by a link as long as the "
                        "distance; snap: the shelter is the nearest node (the 2024 study)")
    g.add_argument("--no-bbox", action="store_true", help="use shelters outside the box of the network too")
    g.add_argument("--max-snap-distance", type=float,
                   help="metres: leave out shelters farther than this from every node (the longest access link, with attach)")
    g.add_argument("--census", help="GeoJSON of the census mesh")
    g.add_argument("--census-column", default="M_TOTPOP_H")
    g.add_argument("--census-method", choices=("within", "weighted"), default="within",
                   help="within: cells entirely inside the area (the 2024 study); weighted: every cell, by the part inside")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evacrl.casebuild", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    raw = sub.add_parser("from-raw", help="rebuild a case from its raw/ folder")
    raw.add_argument("raw_dir")
    raw.add_argument("case_dir")
    _clean_args(raw)
    _population_args(raw)

    snap = sub.add_parser("from-snapshot", help="build a case from a stored OSM graph")
    snap.add_argument("graph_dir", help="folder with Gnodes.geojson and Gedges.geojson")
    snap.add_argument("case_dir")
    _geo_args(snap)
    _clean_args(snap)
    _population_args(snap)

    osm = sub.add_parser("from-osm", help="download the network of an area and build a case")
    osm.add_argument("case_dir")
    _geo_args(osm)
    _clean_args(osm)
    _population_args(osm)

    val = sub.add_parser("validate", help="check the tables of a case")
    val.add_argument("case_dir", help="a case folder (cases/<name>) or its data folder")
    args = parser.parse_args(argv)

    if args.command == "validate":
        data = os.path.join(args.case_dir, "data") if os.path.isdir(os.path.join(args.case_dir, "data")) else args.case_dir
        report = case_io.validate_case(data)
        print(report)
        return 0 if report.ok else 1

    options = dict(merge=args.merge, threshold=args.threshold, parallel="min", excess=args.excess)
    exclude = not args.include_shelters
    if args.legacy:
        options.update(LEGACY)
        exclude = False
        if hasattr(args, "shelters_as"):
            args.shelters_as = "snap"
    if options["merge"] == "legacy" and getattr(args, "shelters_as", "snap") == "attach":
        parser.error("--merge legacy reproduces the 2024 clean-up, which loses a shelter attached by a link under the threshold: "
                     "use it with --shelters-as snap (or --legacy)")
    given = argv if argv is not None else sys.argv[1:]
    # file names without the folders they happened to be in, so that the record does not depend on the machine
    shown = [os.path.basename(a.rstrip("/")) if os.sep in a else a for a in given]
    provenance = dict(command=" ".join(["python -m evacrl.casebuild"] + shown))

    write_raw = True
    if args.command == "from-raw":
        raw_network = case_io.read_raw(args.raw_dir)
        population = _population(args, exclude, total=args.agents, weights=None)
        if os.path.abspath(args.raw_dir) == os.path.abspath(os.path.join(args.case_dir, "raw")):
            # rebuilding a case from its own raw/: keep that folder, and the record of where the case came from
            write_raw = False
            previous = os.path.join(args.case_dir, "provenance.json")
            if os.path.exists(previous):
                with open(previous, encoding="utf-8") as f:
                    before = json.load(f)
                provenance = {k: before[k] for k in ("command", "census", "shelters") if k in before}
                provenance["rebuilt_with"] = " ".join(["python -m evacrl.casebuild"] + shown)
    else:
        from evacrl.casebuild import geo
        graph = geo.graph_from_snapshot(args.graph_dir) if args.command == "from-snapshot" else None
        area = None
        if args.areas:
            # the polygon in WGS84 (OSM, and the census below, are asked in it)
            area = geo.largest_polygon(geo.read_geojson(args.areas).to_crs("EPSG:4326").geometry.iloc[args.index])
        if graph is None:
            if area is None:
                parser.error("from-osm needs --areas")
            graph = geo.download_network(area)
        points = geo.read_points(args.shelters)
        raw_network, snap = geo.raw_network(graph, points, within=None if args.no_bbox else "bbox",
                                            max_distance=args.max_snap_distance, shelters=args.shelters_as)
        provenance["shelters"] = dict(files=_hashes(args.shelters), **snap)
        weights, total = None, args.agents
        if args.census:
            mesh = geo.read_geojson(args.census).to_crs("EPSG:4326")
            if area is None:
                parser.error("--census needs --areas")
            census_total = geo.area_population(mesh, area, args.census_column, args.census_method)
            total = args.agents if args.agents is not None else int(round(census_total))
            provenance["census"] = dict(file=_hashes([args.census]), column=args.census_column, method=args.census_method,
                                        area_total=census_total, agents=total)
            if args.strategy == "proportional":
                weights = lambda network: geo.node_weights(network, mesh, area, args.census_column, args.census_method,
                                                           exclude_shelters=exclude)[0]
        population = _population(args, exclude, total=total, weights=weights)

    tables, report = build_case(raw_network, args.case_dir, population=population, provenance=provenance,
                                write_raw=write_raw, **options)
    print(f"{args.case_dir}: {tables['network'].num_nodes} nodes, {tables['network'].num_links} links, "
          f"{int(tables['network'].nodes[:, 3].sum())} shelters, "
          f"{0 if tables['agents'] is None else len(tables['agents'])} agents")
    print(report)
    return 0


def _population(args, exclude, total, weights):
    if args.strategy == "uniform" and total is None:
        raise SystemExit("--agents is needed (or --census, to take the population of the area from it)")
    if args.strategy == "proportional" and weights is None:
        raise SystemExit("--strategy proportional needs --census")
    return PopulationSpec(strategy=args.strategy, total=total, per_node=args.per_node, weights=weights,
                          exclude_shelters=exclude, seed=args.seed)


def _hashes(paths):
    out = {}
    for path in paths:
        with open(path, "rb") as f:
            out[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
    return out


if __name__ == "__main__":
    sys.exit(main())
