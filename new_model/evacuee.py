import Agent
import networkx as nx
class Evacuee(Agent):
    """An Evacuee agent inherited from Agent class"""
    count = 0
    evacuees = []

    def __init__(self, verb=False):
        """Initialize attributes of the parent class."""
        super().__init__(self, verb=False)
        self.e_uid = self.count  # unique evacuee id
        self.e_origin = None  # current node in osmid code
        self.e_destination = None  # next node in osmid code
        self.e_speed = None  # needs to be adjusted based on units of the model
        self.e_target = None  # target (a shelter)
        self.e_path = []  # current path to target
        self.e_evac_time = None
        self.e_behavior = None
        self.e_scheduled_time = None  # time for next move (scheduling)
        Evacuee.evacuees.append(self)
        if verb:
            print(f'Evacuee {self.e_uid} created')

    def get_shortest_path(self, G, source, target, verb=False):
        """Find the shortest path from source to target."""
        try:
            route = nx.shortest_path(G, source, target)
        except Exception as e:
            print(e)
        if verb:
            print(f'source: {source}')
            print(f'target: {target}')
            print(f'path: {route}')
        return route
    
    def get_path_to_shelter(self, G, agent):
        """Find the path from origin to shelter location."""
        target = self.get_shelter_location(agent)
        return self.get_shortest_path(G, self.e_origin, target)
    
    def get_shelter_location(self, agent):
        """find the node location of a shelter."""
        return agent.get_node_location()

    def get_shortest_path_length(self, G, source, target, weight='length', method='dijkstra'):
        """Find the total length along a path from source to target."""
        return nx.shortest_path_length(G, source, target, weight, method)

    def get_shortest_path_time(self, G, source, target, weight='length', method='dijkstra', verb=False):
        """Find the total time along a path from source to target."""
        distance = self.get_shortest_path_length(G, source, target, weight, method, verb)
        return int(distance / self.e_speed)

    def reset_path(self):
        """Resets the [e_path] to None."""
        self.e_path = None
        return
        
    def update_e_path(self):
        self.e_path = self.get_shortest_path()
        return

    def update_schedule(self, G, source, target, weight='length', method='dijkstra', verb=False):
        """Updates the variable [e_scheduled_time]."""
        if self.e_speed != 0:
            self.e_scheduled_time = self.get_shortest_path_time(G, source, target, weight, method, verb)
        else:
            self.e_scheduled_time = 0
        return

    def check_schedule(self, model):
        """Check if model time is equal to evacuee's scheduled time."""
        if self.e_scheduled_time == model.step:
            return True
        else:
            return False

    def move(self):
        """Move agent forward"""
        pass

    def set_evac_time(self):
        """Sets a time for decision of starting evacuation in [e_evac_time]."""
        pass

    def check_evac_time(self):
        """Check if model time is equal to the evacuee's evacuation time."""
        pass

    def set_behavior(self):
        """Sets a behavior of evacuation in [e_behavior].
        0 : Shortest path - Non compliance of guidance sytem
        1 : Reinforcement Learning - Compliance of guidance system
        """
        pass
    
    def get_closest_agent(self, agent_list):
        """Finds the closest agent from a list of agents with (x,y) coordinates."""
        pass
    
    
    