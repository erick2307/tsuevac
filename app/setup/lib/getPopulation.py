#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

pref = {
    1: "北海道",
    2: "青森県",
    3: "岩手県",
    4: "宮城県",
    5: "秋田県",
    6: "山形県",
    7: "福島県",
    8: "茨城県",
    9: "栃木県",
    10: "群馬県",
    11: "埼玉県",
    12: "千葉県",
    13: "東京都",
    14: "神奈川県",
    15: "新潟県",
    16: "富山県",
    17: "石川県",
    18: "福井県",
    19: "山梨県",
    20: "長野県",
    21: "岐阜県",
    22: "静岡県",
    23: "愛知県",
    24: "三重県",
    25: "滋賀県",
    26: "京都府",
    27: "大阪府",
    28: "兵庫県",
    29: "奈良県",
    30: "和歌山県",
    31: "鳥取県",
    32: "島根県",
    33: "岡山県",
    34: "広島県",
    35: "山口県",
    36: "徳島県",
    37: "香川県",
    38: "愛媛県",
    39: "高知県",
    40: "福岡県",
    41: "佐賀県",
    42: "長崎県",
    43: "熊本県",
    44: "大分県",
    45: "宮崎県",
    46: "鹿児島県",
    47: "沖縄県",
}


def getPopulation(pref_code=39):
    """
    Returns a GeoDataFrame of Census Data
    Args:
    pref_code (int, optional): Prefecture code.
    Defaults to 39 (Kochi Pref.).
    """
    rootfolder = "/Volumes/Pegasus32/data"
    # GetAreaCodes
    datafolder = "PAREA_Town_2018/Shape形式/Shape形式/世界測地系"
    areafile = f"{pref_code:02d}/A{pref_code:02d}24POL.shp"
    path = os.path.join(rootfolder, datafolder, areafile)
    area = gpd.read_file(path, encoding="shift_jis")
    # GetPopulationData
    datafolder = "PAREA_StatAllforTown_2018/PAREA-Stat.Population"
    headerfile = "ヘッダーファイル/JC0401S0000.csv"
    popfile = f"data/{pref_code:02d}/S{pref_code:02d}4JC0401S0000.csv"
    pathfile = os.path.join(rootfolder, datafolder, popfile)
    pathheader = os.path.join(rootfolder, datafolder, headerfile)
    pop_h = pd.read_csv(pathheader)
    pop = pd.read_csv(pathfile, names=pop_h.columns)
    # Change field names to english
    colnames = {
        "行政コード": "Val_GovCod",
        "図形有無F": "FigureF",
        "分割合算F": "DivRatioF",
        "秘匿処理F": "ConcealF",
        "リザーブ1": "Reserve1",
        "リザーブ2": "Reserve2",
        "指標数": "NoIndicators",
        "総数(人口)": "TotalPop",
        "男(人口)": "Male",
        "女(人口)": "Female",
        "世帯数": "NoHouseholds",
        "面積(Ｋ㎡)": "AreaKm2",
        "人口密度(Ｋ㎡あたり人口)": "PopDensity",
    }
    pop.rename(columns=colnames, inplace=True)
    # Merge numerical and spatial data
    data = pd.merge(pop, area, on="Val_GovCod")
    # Change to Geodataframe
    gdata = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    return gdata


def getPopulation_before2011(pref_code=39):
    """
    Returns a GeoDataFrame of Census Data
    Args:
    pref_code (int, optional): Prefecture code.
    Defaults to 39 (Kochi Pref.).
    """
    rootfolder = "/Volumes/Pegasus32/data"
    # GetAreaCodes
    datafolder = "Census_2010/平成22年国勢調査100mメッシュ推計データ"
    areafile = f"{pref_code:02d}{pref[pref_code]}/メッシュ地図2010WH_{pref_code:02d}.shp"
    path = os.path.join(rootfolder, datafolder, areafile)
    area = gpd.read_file(path, encoding="shift_jis")
    area.rename(columns={"メッシュコ": "mesh", "geometry": "geometry"}, inplace=True)
    area["mesh"] = area["mesh"].astype("int64")
    # GetPopulationData
    popfile = f"{pref_code:02d}{pref[pref_code]}/国勢調査2010メッシュ100M人口{pref_code:02d}.csv"
    pathfile = os.path.join(rootfolder, datafolder, popfile)
    pop = pd.read_csv(pathfile)
    pop.rename(columns={"1": "mesh", "8": "TotalPop"}, inplace=True)
    # Merge numerical and spatial data
    data = pd.merge(pop, area, on="mesh")
    # Change to Geodataframe
    gdata = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    return gdata


def getPopulationArea(pref_code, aos, crs=6690, before311=False):
    if before311:
        gdf1 = getPopulation_before2011(pref_code)
    else:
        gdf1 = getPopulation(pref_code)
    gdf1.to_crs(crs, inplace=True)
    gdf2 = gpd.read_file(aos)
    gdf2.to_crs(crs, inplace=True)
    gdf_int = gpd.overlay(gdf1, gdf2, how="intersection")
    # gdf_int = gpd.overlay(gdf1, gdf2, keep_geom_type=False, how='intersection')
    return gdf_int


def plotPopulation(gdata, bck=False):
    fig, ax = plt.subplots(1, 1)
    if bck:
        import contextily as ctx

        gdata.to_crs(3857, inplace=True)
    gdata.plot(
        column="TotalPop",
        ax=ax,
        legend=True,
        legend_kwds={"loc": 4},
        cmap="OrRd",
        scheme="quantiles",
    )
    if bck:
        ctx.add_basemap(ax, zoom=12)
    plt.show()


def plotFolium(gdata):
    import folium

    gdata.to_crs(4326, inplace=True)
    lonlat = gdata.geometry.total_bounds[0:2]
    loc = list([lonlat[1], lonlat[0]])
    map = folium.Map(location=loc, tiles="OpenStreetMap", zoom_start=14)
    for _, r in gdata.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])  # .simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j, style_function=lambda x: {"fillColor": "orange"}
        )
        geo_j.add_to(map)
    return map


if __name__ == "__main__":
    # plotPopulation(getPopulation(4))
    aos = "../../input/arahama_aos.geojson"
    crs = "EPSG:6691"
    gdf_int = getPopulationArea(4, aos=aos, crs=crs, before311=True)
    map = plotFolium(gdf_int)
    map.save("arahama.html")
