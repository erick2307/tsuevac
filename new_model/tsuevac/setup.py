from pathlib import Path
import random
import numpy as np
import geopandas as gpd

print(f'Setting default parameters from {__name__}')
# INPUTS *********************************************\
CASE = 'arahama'
AOS_FILE = Path('./data/arahama_aos.geojson')
TSUNAMI_FOLDER = Path('./data/tsunami')
SHELTERS_FILE = Path('./data/shelters.csv')
REWARD = -1
NUM_EPISODES = 100
CONGESTION_LEVELS = [0, 1, 2]

# DEFAULT PARAMETERS *********************************************\

REPLICATION = False
# If 'False' decide a value for the SEED
SEED = 12345
