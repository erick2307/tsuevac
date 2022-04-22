print(f'Setting default parameters from {__name__}')
import numpy as np
import random
import os, os.path
import osmnx as ox
import networkx as nx
import cv2
from pathlib import Path
import datetime
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import textwrap
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from ipywidgets import FloatProgress
import rasterio
from rasterio.plot import show
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from matplotlib.colors import LinearSegmentedColormap
cmap_triage = LinearSegmentedColormap.from_list('triage',['g','y','r','k'])
mpl.style.use('seaborn')

seed = 12345
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng(seed)
case = os.path.basename(os.getcwd())
print(f'Setting {case} case')

# *********************************************\
# YOU MAY CHANGE FILENAMES OR 'None' IF NOT AVAILABLE

shelters_file = f'medabm/data/{case}_hospitals.csv'
aos_file = f'medabm/data/{case}_aos.geojson'
tsunami_file = f'medabm/data/{case}_tsunami.tif'

# *********************************************\
if aos_file != None:
    gdf = gpd.read_file(aos_file)
    bbox = gdf.total_bounds
    aos_bbox = {'north': bbox[3], 'south': bbox[1],
        'east': bbox[2], 'west': bbox[0]}