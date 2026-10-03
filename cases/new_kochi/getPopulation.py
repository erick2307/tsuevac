# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import os

pref = {1: u'北海道', 2: u'青森県', 3: u'岩手県', 4: u'宮城県', 5: u'秋田県',
        6: u'山形県', 7: u'福島県', 8: u'茨城県', 9: u'栃木県', 10: u'群馬県',
        11: u'埼玉県', 12: u'千葉県', 13: u'東京都', 14: u'神奈川県', 15: u'新潟県',
        16: u'富山県', 17: u'石川県', 18: u'福井県', 19: u'山梨県', 20: u'長野県',
        21: u'岐阜県', 22: u'静岡県', 23: u'愛知県', 24: u'三重県', 25: u'滋賀県',
        26: u'京都府', 27: u'大阪府', 28: u'兵庫県', 29: u'奈良県', 30: u'和歌山県',
        31: u'鳥取県', 32: u'島根県', 33: u'岡山県', 34: u'広島県', 35: u'山口県',
        36: u'徳島県', 37: u'香川県', 38: u'愛媛県', 39: u'高知県', 40: u'福岡県',
        41: u'佐賀県', 42: u'長崎県', 43: u'熊本県', 44: u'大分県', 45: u'宮崎県',
        46: u'鹿児島県', 47: u'沖縄県'}


def getPopulation(pref_code=39):
    """
    Returns a GeoDataFrame of Census Data
    Args:
    pref_code (int, optional): Prefecture code. 
    Defaults to 39 (Kochi Pref.).
    """
    rootfolder = "/Volumes/Pegasus32/data" #"/Users/erick/ReGID Dropbox/zDATA"
    # GetAreaCodes
    datafolder = u"PAREA_Town_2018/Shape形式/Shape形式/世界測地系"
    areafile = f"{pref_code:02d}/A{pref_code:02d}24POL.shp"
    path = os.path.join(rootfolder, datafolder, areafile)
    area = gpd.read_file(path, encoding='shift_jis')
    # GetPopulationData
    datafolder = u"PAREA_StatAllforTown_2018/PAREA-Stat.Population"
    headerfile = u"ヘッダーファイル/JC0401S0000.csv"
    popfile = f"data/{pref_code:02d}/S{pref_code:02d}4JC0401S0000.csv"
    pathfile = os.path.join(rootfolder, datafolder, popfile)
    pathheader = os.path.join(rootfolder, datafolder, headerfile)
    pop_h = pd.read_csv(pathheader)
    pop = pd.read_csv(pathfile, names=pop_h.columns)
    # Change field names to english
    colnames = {'行政コード': "Val_GovCod", '図形有無F': "FigureF", '分割合算F': "DivRatioF",
                '秘匿処理F': "ConcealF", 'リザーブ1': "Reserve1", 'リザーブ2': "Reserve2",
                '指標数': "NoIndicators", '総数(人口)': "TotalPop", '男(人口)': "Male",
                '女(人口)': "Female", '世帯数': "NoHouseholds", '面積(Ｋ㎡)': "AreaKm2",
                '人口密度(Ｋ㎡あたり人口)': "PopDensity"}
    pop.rename(columns=colnames, inplace=True)
    # Merge numerical and spatial data
    data = pd.merge(pop, area, on="Val_GovCod")
    # Change to Geodataframe
    gdata = gpd.GeoDataFrame(data, geometry=data['geometry'], crs="EPSG:4326")
    return gdata


def getPopulationArea(pref_code, aos, crs=6690):
    gdf1 = getPopulation(pref_code)
    gdf1.to_crs(crs, inplace=True)
    gdf2 = gpd.read_file(aos)
    gdf2.to_crs(crs, inplace=True)
    gdf_int = gpd.overlay(gdf1, gdf2, how='intersection')
    #gdf_int = gpd.overlay(gdf1, gdf2, keep_geom_type=False, how='intersection')
    return gdf_int


def plotPopulation(gdata, crs=3857, bck=False):
    fig, ax = plt.subplots(1, 1)
    if bck:
        import contextily as ctx
        gdata.to_crs(crs, inplace=True)
    ccrs = gdata.crs
    gdata.plot(column='TotalPop', ax=ax, legend=True,
               legend_kwds={'loc': 4},
               cmap='OrRd', scheme='quantiles')
    if bck:
        ctx.add_basemap(ax, zoom=12)
    plt.show()
    print(f"Plotted in CRS:{ccrs}")
    return


def plotFolium(gdata, crs=4326):
    import folium
    gdata.to_crs(crs, inplace=True)
    lonlat = gdata.geometry.total_bounds[0:2]
    loc = list([lonlat[1], lonlat[0]])
    map = folium.Map(location=loc, tiles='OpenStreetMap', zoom_start=14)
    for _, r in gdata.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                               style_function=lambda x: {'fillColor': 'orange'})
        geo_j.add_to(map)
    print(f"Data in CRS:{crs}")
    return map


if __name__ == '__main__':
    #plotPopulation(getPopulation(4))
    getPopulationArea()
