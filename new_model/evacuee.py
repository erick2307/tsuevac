import networkx as nx

class Agent(object):
    """An agent class"""
    count = 0
    agents = []

    def __init__(self, verb=False):
        self.a_uid = self.count  # unique id
        self.a_home = None  # initial location
        self.a_current_coordinate = None  # x,y of a_current
        self.a_dead = False  # to check if it is out of the model
        self.a_state = 0  # code for states
        self.a_current = None  # current location in osmid (node)
        self.a_destination = None  # next location in osmid (node)
        self.a_velocity = (0, 1)  # needs to be adjusted based on units of the model
        self.a_g_obj = None  # target object (a shelter)
        self.a_task = 0  # code for tasks
        self.a_mypath = []  # current path to target
        self.a_mytime = None  # time for next move (scheduling)
        self.__class__.count += 1
        Agent.agents.append(self)
        if verb:
            print(f'Agent {self.a_uid} created')

    def find_path(self, model, G, source, target, verb=False):
        "finds the path from source to target and saves the total time to a_mypath variable"
        try:
            self.a_mypath = nx.shortest_path(G, source, target)
            self.find_path_time(model, G, source, target, verb=verb)
        except Exception as e:
            if verb:
                print(e)
            self.a_mypath = None
            self.a_mytime = None
        if verb:
            print(f'{model.step}> A{self.a_uid} source: {source}')
            print(f'{model.step}> A{self.a_uid} target: {target}')
            print(f'{model.step}> A{self.a_uid} path: {self.a_mypath}')
            print(f'{model.step}> A{self.a_uid} time: {self.a_mytime}s')
        return self.a_mypath

    def find_path_length(self, G, source, target, weight='length', method='dijkstra', verb=False):
        "finds the total length along a path from source to target"
        return nx.shortest_path_length(G, source, target, weight, method)

    def find_path_time(self, model, G, source, target, weight='length', method='dijkstra', verb=False):
        "finds the total time along the whole path and saves it to a_mytime variable"
        distance = self.find_path_length(G, source, target, weight, method, verb=verb)
        veloc = self.speed()
        if verb:
            print(f'{model.step}> A{self.a_uid}:D={distance}m ;V={veloc}m/s')
        if veloc != 0:
            self.a_mytime = int((distance / veloc) + model.step)
        else:
            self.a_mytime = 0

    def speed(self):
        "calculates the speed from velocity vector"
        vx = self.a_velocity[0]
        vy = self.a_velocity[1]
        s = (vx ** 2 + vy ** 2) ** 0.5
        return s

    # def check_arrived(self, model, verb=False):
    #     if self.a_mytime == model.step:
    #         if self.a_injury is not None:  # carrying patient, arrived hospital ??
    #             self.arrived_to_hospital(model, verb=verb)
    #         else:  # no patient yet, arrived to patient
    #             self.arrived_to_patient(model, verb=verb)

    def check_move(self, model, verb=False):
        "checks if agent should move based on his time to next target and model time step"
        if self.a_mytime == model.step:
            self.a_state = 0
            self.a_mypath = [self.a_mypath[-1]]
            self.a_mytime = None
            if verb:
                print(f'{model.step}> A{self.a_uid} arrived!')

    def move(self, model, verb=False):
        "to move agents"
        self.a_current = self.a_mypath[0]
        self.a_mypath.pop(0)
        s = model.nodes[model.nodes.index == self.a_current]
        self.a_current_coordinate = (float(s['x']), float(s['y']))
        if verb:
            print(f'{model.step}> A{self.a_uid} moving to {self.a_current}')
