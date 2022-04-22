import networkx as nx
import folium
import osmnx as ox
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import rasterio
from rasterio.plot import show
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

class Environment(object):
    def __init__(self, bbox, nw_type='drive', verb=False):
        # bbox is a dict of north,south,east,west lat,lon edges of target area
        self.e_bbox = {'north': bbox['north'], 'south': bbox['south'],
                       'east': bbox['east'], 'west': bbox['west']}
        self.e_get_network(nw_type, verb)

    def e_get_network(self, nw_type='drive', verb=False):
        ox.config(use_cache=True)
        # Obtain the roadmap data from OpenStreetMap by using OSMNX
        self.e_G = ox.graph_from_bbox(self.e_bbox['north'], self.e_bbox['south'],
                                      self.e_bbox['east'], self.e_bbox['west'],
                                      network_type=nw_type)
        if verb:
            print('Network downloaded')

    def e_network_to_shp(self, filepath='./tests/.', verb=False):
        # Export the graph as a shapefile
        ox.io.save_graph_shapefile(self.e_G, filepath)
        if verb:
            print(f'Network exported to {filepath}')

    def e_project_network(self, crs=None, verb=False):
        # Project network
        if crs:
            self.e_Gp = ox.project_graph(self.e_G, to_crs=crs)
        else:
            self.e_Gp = ox.project_graph(self.e_G)
        if verb:
            print('Network projected')

    def e_get_nodes(self, verb=False):
        self.e_nodes, self.e_edges = ox.graph_to_gdfs(self.e_G)
        # self.e_nodes.reset_index(inplace=True)
        # self.e_edges.reset_index(inplace=True)
        if verb:
            print('Nodes and Edges created')
        return self.e_nodes

    def e_get_edges(self, verb=False):
        self.e_nodes, self.e_edges = ox.graph_to_gdfs(self.e_G)
        # self.e_nodes.reset_index(inplace=True)
        # self.e_edges.reset_index(inplace=True)
        if verb:
            print('Nodes and Edges created')
        return self.e_edges

    def e_get_nearest_nodes(self, gdA, gdB, verb=False):
        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=False)
        gdf = pd.concat([gdA.reset_index(drop=True), gdB_nearest, pd.Series(dist, name='dist')], axis=1)
        return gdf

    def e_add_nodes_to_graph(self, gdf, verb=False):
        last = self.e_G.number_of_nodes()
        num = gdf.shape[0]
        for i in range(num):
            key = last + i
            self.e_G.add_node(key, geometry=gdf.iloc[i]['geometry'])
        if verb:
            print(f'{num} nodes added. Before:{last}, After:{self.e_G.number_of_nodes()}.')

    def e_add_edges_to_graph(self, gdf, verb=False):
        last = self.e_G.number_of_edges()
        last_nodes = self.e_G.number_of_nodes()
        num = gdf.shape[0]
        for i in range(num):
            self.e_G.add_edge(last_nodes + i, gdf.iloc[i]['osmid'])
        if verb:
            print(f'{num} edges added. Before:{last}, After:{self.e_G.number_of_edges()}.')
            print(f'Nodes:{self.e_G.number_of_nodes()}.')

    def e_load_tsunami(self, tsunami_file, verb=False):
        # open a raster (e.g. TIF) file and return the raster object
        # raster_file is global variable (see '__init__.py')
        self.e_tsunami = rasterio.open(tsunami_file)
        self.e_tsunami_crs = self.e_tsunami.crs
        if verb:
            print(f'Raster {tsunami_file} read with projection {self.e_tsunami_crs}')

    def e_show_tsunami(self, ax=None, vmin=1, vmax=10, verb=False):
        if ax is None:
            show(self.e_tsunami, cmap='Blues', vmin=vmin, vmax=vmax)
        else:
            show(self.e_tsunami, cmap='Blues', ax=ax, vmin=vmin, vmax=vmax)
        if verb:
            print('Change default values as vmin=1, vmax=10')

    def e_set_tsunami_in_nodes(self, verb=False):
        tsu = []
        for row in self.e_nodes.iterrows():
            x = row[1].x
            y = row[1].y
            row, col = self.e_tsunami.index(x, y)
            try:
                tsu.append(self.e_tsunami.read(1)[row, col])
            except Exception as e:
                if verb:
                    print(e)
                tsu.append(float("nan"))

        for i, v in enumerate(tsu):
            if v == -99.:
                tsu[i] = float("nan")

        self.e_nodes['depth'] = tsu
        nx.set_node_attributes(self.e_G, values=self.e_nodes['depth'], name="depth")
        if verb:
            print('Tsunami in nodes done!')
            print(self.e_get_nodes_attribute_list())

    def e_plot_tsunami_in_network(self, width=16, height=8, colorbar=False, verb=False):
        nc = ox.plot.get_node_colors_by_attr(G=self.e_G, attr='depth', cmap='Reds',
                                             start=0, stop=3, na_color='none', equal_size=True)
        ncr = ox.plot.get_node_colors_by_attr(G=self.e_G, attr='depth', num_bins=5, cmap='Reds',
                                              start=0, stop=3, na_color='none', equal_size=False)
        cmap = plt.cm.get_cmap('Reds')
        norm = plt.Normalize(vmin=0, vmax=3)
        # (vmin=nodes['tsu'].min(), vmax=nodes['tsu'].max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig, ax = ox.plot_graph(G=self.e_G, ax=None, figsize=(width, height), bgcolor='w', node_size=0.5,
                                node_color=nc, node_alpha=0.8, node_edgecolor=nc, node_zorder=2,
                                edge_color='gray', show=False, close=False)
        if colorbar:
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
        return fig, ax

    def e_load_aos(self, aos_file, explore=False, verb=False):
        # to load the geometry of the area of study (aos)
        self.e_aos = gpd.read_file(aos_file)
        if explore:
            self.e_aos.explore(style_kwds=dict(fill=False, color="black"))
        if verb:
            print('Geometry of area of study loaded')

    def e_get_network_projection(self, graph=None, verb=False):
        # to check the projection of the graph
        if graph:
            self.e_G_crs = graph.graph["crs"]
        else:
            self.e_G_crs = self.e_G.graph["crs"]
        if verb:
            print(f"Your graph has this projection: {self.e_G_crs}")

    def e_get_nodes_attribute_list(self, verb=False):
        return set([k for n in self.e_G.nodes for k in self.e_G.nodes[n].keys()])

    def e_plot(self, width=16, height=8, show=False, close=False, verb=False):
        # returning a fig and ax
        if verb:
            print('==' * 10)
        return ox.plot_graph(self.e_G, figsize=(width, height), bgcolor='w', node_color='k',
                             node_alpha=0.1, edge_color=(0, 0, 0, 0.5), show=show,
                             close=close)

    def e_map(self, verb=False):
        if verb:
            print('==' * 10)
        return ox.plot_graph_folium(self.e_G, tiles='OpenStreetMap')
