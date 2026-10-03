from tsuevac import Environment
from tsuevac import Agent
from tsuevac import Evacuee
from tsuevac import Node
from tsuevac import Shelter
from tsuevac import setup
import random
import sys
import numpy as np
from pathlib import Path
import geopandas as gpd

class Model(object):
    """A Model class"""
    def __init__(self, case=setup.CASE, verb=False):
        """Initialize the model"""
        self.clean()
        if setup.REPLICATION:
            random.seed(setup.SEED)
            np.random.seed(setup.SEED)
            self.rng = np.random.default_rng(setup.SEED)
        self.id = case
        p = Path(f'./cases/{self.id}')
        try:
            p.mkdir(parents=True, exist_ok=False)
            print('Running Model.setup()')
            self.overwrite = True
            self.setup(verb)
        except FileExistsError:
            print('The directory already exists.')
            overwrite = input('Overwrite? ([y]/n):')
            if overwrite == 'y':
                print('Overwriting. Running Model.setup()')
                self.overwrite = True
                self.setup(verb)
            else:
                print('Nothing done. Aborting.')
                self.overwrite = False
        return

    def setup(self, verb=False):
        """Setup the model."""
        self.bbox = self.load_aos_file(setup.AOS_FILE)
        self.tsunami = setup.RUN_TSUNAMI
        self.Env = Environment(self.bbox, setup.NETWORK_TYPE, verb)
        self.Env.e_get_network_projection(self.Env.e_G, verb)
        self.nodes = self.Env.e_get_nodes(verb)
        self.edges = self.Env.e_edges
        self.create_node_agents()
        self.Env.e_load_aos(aos_file=setup.AOS_FILE, verb=verb)
        self.step = 0

    def load_aos_file(self, aos_file):
        if aos_file is not None:
            gdf = gpd.read_file(aos_file)
            bbox = gdf.total_bounds
            aos_bbox = {'north': bbox[3], 'south': bbox[1], 'east': bbox[2], 'west': bbox[0]}
        return aos_bbox

    def go(self):
        while self.step != setup.NUM_EPISODES:
            self.step += 1
            self.call_evacuees()
            self.call_replay_memory()
            flag = self.check_terminal_state()
            if flag:
                break
        return

    def call_evacuees(self):
        pass

    def call_replay_memory(self):
        pass

    def check_terminal_state(self):
        pass

    def create_node_agents(self):
        pass

    def initial_condition(self):
        pass

    def read_shelters_from_file(self):
        pass

    def clean(self):
        try:
            Agent.count = 0
            Agent.agents = []
            Evacuee.count = 0
            Evacuee.evacuees = []
            Node.count = 0
            Node.nodes = []
            Shelter.count = 0
            Shelter.shelters = []
        except Exception as e:
            print("Error:", e)
