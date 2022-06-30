#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely as shp
from lib import createLinksAndNodes as cln
from lib import getPopulation as gp
from lib import setActionsAndTransitions as actrans


def createFolders():
    tempfolder = Path("./tmp")
    if not Path.exists(tempfolder):
        Path.mkdir(tempfolder)
    datafolder = Path("./data")
    if not Path.exists(datafolder):
        Path.mkdir(datafolder)


def get_area_from_geojson(aos_file):
    gdf = gpd.read_file(aos_file)
    bbox = gdf.total_bounds
    area = {"north": bbox[3], "south": bbox[1], "east": bbox[2], "west": bbox[0]}
    return area


def createBoundingBox(
    area={"north": 33.58, "south": 33.53, "east": 133.58, "west": 133.52}
):
    # create a Bounding Box
    # returns a shapely geometry
    bbox = shp.geometry.box(area["west"], area["south"], area["east"], area["north"])
    return bbox


def createGraph(
    area={"north": 33.58, "south": 33.53, "east": 133.58, "west": 133.52},
    crs="EPSG:6690",
    ntype="drive",
    plot=False,
    simplify=False,
):
    # NEW AREA (Aug, 2021)
    maprange = area  # Degree 1.0 = 111km. 2.22km x 4.44km square.
    # Obtain the roadmap data from OpenStreetMap by using OSMNX
    G = ox.graph_from_bbox(
        maprange["north"],
        maprange["south"],
        maprange["east"],
        maprange["west"],
        network_type=ntype,
        simplify=simplify,
    )
    G_projected = ox.project_graph(G, to_crs=crs)
    if simplify:
        # Simplify topology
        G_out = ox.simplification.consolidate_intersections(
            G_projected,
            tolerance=10,
            rebuild_graph=True,
            dead_ends=False,
            reconnect_edges=True,
        )
        print(
            f"""
        The graph was saved as 'graph.gpkg' with projection {crs}.
        This is a SIMPLIFIED {ntype} type OSM network.
        """
        )
    else:
        G_out = G_projected
        print(
            f"""
        The graph was saved as 'graph.gpkg' with projection {crs}.
        This is an {ntype} type OSM network.
        """
        )
    # save the OSM data as Geopackage
    ox.io.save_graph_geopackage(G_out, filepath="./tmp/graph.gpkg")
    # Draw a map
    if plot:
        ox.plot_graph(G_out, bgcolor="white", node_color="red", edge_color="black")
    # Create edges file
    edges = gpd.read_file("./tmp/graph.gpkg", layer="edges")
    edges.to_file("./tmp/edges.shp")
    nedges = G.number_of_edges()
    print(f"{nedges} edges in graph. The edges file was saved as ESRI shapefile")
    # create database of links and edges
    cln.main()
    print("Database created!")
    if plot:
        cln.plotNetwork()
    return G_out


def getPrefShelters(pref_code=39, crs="EPSG:6690", filter=True):
    rootfolder = "/Volumes/Pegasus32/data"
    datafolder = "PAREA_Hazard_2018/data/世界測地系"
    areafile = f"{pref_code:02d}/PHRP{pref_code:02d}18.shp"
    path = os.path.join(rootfolder, datafolder, areafile)
    shelters_gc = gpd.read_file(path, encoding="shift_jis")
    shelters = shelters_gc.to_crs(crs)
    if filter:
        shelters = shelters[shelters["TUNAMI"] == 1]
    return shelters


def getAreaShelters(
    area={"north": 33.58, "south": 33.53, "east": 133.58, "west": 133.52},
    pref_code=39,
    crs="EPSG:6690",
):
    bbox = createBoundingBox(area)
    bbox_gdf = gpd.GeoSeries(bbox)
    bbox_gdf.set_crs("EPSG:4326", inplace=True)
    bbox_gdf = bbox_gdf.to_crs(crs)
    poly = shp.geometry.shape(bbox_gdf[0])
    shelters = getPrefShelters(pref_code, crs)
    gs = gpd.GeoSeries(shelters.geometry)
    sh_in_area = shelters[gs.within(poly)]
    return sh_in_area


def getPrefEvacBldgs(crs="EPSG:4326"):
    rootfolder = "/volumes/Pegasus32/kochi/evacuation"
    filename = "Kochi_EvacBldg_20211220.csv"
    path = Path(rootfolder, filename)
    bldgs_pd = pd.read_csv(path)
    bldgs = gpd.GeoDataFrame(
        bldgs_pd, geometry=gpd.points_from_xy(bldgs_pd["lon"], bldgs_pd["lat"])
    )
    bldgs.set_crs("EPSG:4326", inplace=True)
    bldgs.to_crs(crs)
    return bldgs


def getAreaEvacBldgs(
    area={"north": 33.58, "south": 33.53, "east": 133.58, "west": 133.52},
    crs="EPSG:6690",
):
    bbox = createBoundingBox(area)
    bbox_gdf = gpd.GeoSeries(bbox)
    bbox_gdf.set_crs("EPSG:4326", inplace=True)
    bbox_gdf = bbox_gdf.to_crs(crs)
    poly = shp.geometry.shape(bbox_gdf[0])
    bldgs = getPrefEvacBldgs(crs)
    gs = gpd.GeoSeries(bldgs.geometry)
    bldgs_in_area = bldgs[gs.within(poly)]
    return bldgs_in_area


def pointsWithinPolygon(poly):
    # Get the nodes within a polygon
    df = pd.read_csv("./data/nodesdb.csv")
    nodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.coord_x, df.coord_y))
    gs = gpd.GeoSeries(nodes.geometry)
    ninarea = nodes[gs.within(poly)]
    return ninarea


def fixLinksDBAndNodesDB(shelters):
    # fixes the nodesdb to add shelters as nodes
    # requires 'shelters' geodataframe
    # the 'nodesdb.csv' was created with 'createLinksAndNodes.py'
    nodesnp = np.loadtxt("./tmp/nodesdb0.csv", delimiter=",")
    linksdb = np.loadtxt("./tmp/linksdb0.csv", delimiter=",")
    # create a numpy array for nodesdb
    nodesdb = np.zeros((nodesnp.shape[0], nodesnp.shape[1] + 2))
    nodesdb[:, :3] = nodesnp[:, :3]
    # create a pandas then numpy for shelter coords
    sheltersdb = pd.DataFrame()
    for i, g in zip(shelters.index, shelters["geometry"]):
        x, y = g.coords.xy
        sheltersdb.loc[i, "x"], sheltersdb.loc[i, "y"] = x[0], y[0]
    sheltersdb = sheltersdb.to_numpy()
    for i in range(sheltersdb.shape[0]):
        x0, y0 = sheltersdb[i, :]
        dist = ((nodesnp[:, 1] - x0) ** 2 + (nodesnp[:, 2] - y0) ** 2) ** 0.5
        indx = np.argmin(dist)
        nodesdb[indx, 3] = 1
    nodesdb[:, 4] += 1
    # correcting links length = 0 to = 2
    linksdb[:, 3][np.where(linksdb[:, 3] == 0)] = 2
    np.savetxt(
        "./data/nodesdb.csv",
        nodesdb,
        delimiter=",",
        header="number,coord_x,coord_y,evacuation,reward",
        fmt="%d,%.6f,%.6f,%d,%d",
    )
    np.savetxt(
        "./data/linksdb.csv",
        linksdb,
        delimiter=",",
        header="number,node1,node2,length,width",
        fmt="%d,%d,%d,%d,%d",
    )
    return


def appendAgents(agentsdb, pop, index, poly):
    # Get a polygon
    poly_pop = pop.TotalPop.to_list()[index]
    ninarea = pointsWithinPolygon(poly)
    if ninarea.shape[0] == 0:
        return agentsdb
    pop_per_node = int(poly_pop / ninarea.shape[0])
    from_row = np.trim_zeros(agentsdb[:, 4], "b").shape[0]
    to_row = from_row + ninarea.shape[0] * pop_per_node  # +1?
    n = ninarea["# number"].to_list()
    nr = np.repeat(n, pop_per_node)
    agentsdb[from_row:to_row, 4] = nr
    return agentsdb


def copyDataFolder(case="kochi"):
    shutil.copytree(Path("./data"), Path(f"../{case}/data"), dirs_exist_ok=True)


if __name__ == "__main__":
    # Set Prefecture / Area of Interest / working CRS
    # case = 'kochi'
    # pref_code = 39
    # area = {'north': 33.58, 'south': 33.53, 'east': 133.58, 'west': 133.52}
    # crs = 'EPSG:6690'

    # Set Prefecture / Area of Interest / working CRS
    case = "arahama"
    pref_code = 4
    area = get_area_from_geojson("../input/arahama_aos.geojson")
    crs = "EPSG:6691"
    before311 = True
    # create Folders
    createFolders()
    # Get a Bounding Box of the area
    bbox = createBoundingBox(area=area)
    aos = gpd.GeoSeries([bbox]).to_json()
    # Get Population in the Area of Study
    pop = gp.getPopulationArea(
        pref_code=pref_code, aos=aos, crs=crs, before311=before311
    )
    # Create a Graph object
    G = createGraph(area=area, crs=crs, ntype="drive", plot=False, simplify=False)
    # Create a Shelter GeoDataframe
    shelters = getAreaShelters(area=area, pref_code=pref_code, crs=crs)
    if pref_code == 39:
        bldgs = getAreaEvacBldgs(area=area, crs=crs)
    # Create  Links and nodes?
    cln.main()
    # Fix the nodesdb
    fixLinksDBAndNodesDB(shelters)
    actrans.setMatrices()
    # Create the agentsdb
    agentsdb = np.zeros((pop.TotalPop.sum().astype("int64"), 5))
    for i, g in enumerate(pop.geometry.to_list()):
        agentsdb = appendAgents(agentsdb, pop=pop, index=i, poly=g)
    last = np.trim_zeros(agentsdb[:, 4], "b").shape[0]
    agentsdb = agentsdb[:last, :]
    np.savetxt(
        "./data/agentsdb.csv",
        agentsdb,
        delimiter=",",
        header="age,gender,hhType,hhId,Node",
        fmt="%d,%d,%d,%d,%d",
    )
    copyDataFolder(case)
