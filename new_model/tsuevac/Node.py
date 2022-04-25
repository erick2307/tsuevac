from tsuevac import Agent
from tsuevac import setup

class Node(Agent):
    """A Node agent inherited from Agent class."""
    count = 0
    nodes = []

    def __init__(self, verb=False):
        """Initialize attributes of the parent class."""
        super().__init__(self, verb)
        self.n_uid = self.count  # unique node id
        self.n_num_evacuees = None  # number of evacuees in node
        self.n_num_edges = self.get_number_of_edges()  # number of edges in node
        self.n_State = np.zeros(self.n_num_edges)
        # The state vector of a node consists on congestion conditions
        # for each edge connected to the node.
        # i.e. [ congestion ] * n_num_edges
        self.n_Action_space = np.linspace(0, self.n_num_edges - 1, self.n_num_edges).astype(np.int64)
        # The possible actions are the directions towards the next node
        # through each connected edge. i.e. [0, 1, 2, ...]
        self.n_State_space = list(it.combinations_with_replacement(CONGESTION_LEVELS, len(self.n_Action_space)))
        # The state space is the combiation of all possible values of congestion
        # at each edge.
        self.n_Action = None
        self.n_Reward = REWARD
        self.n_Qvalue = None
        self.n_Qtable = np.empty([len(self.n_State_space), len(self.n_Action_space), NUM_EPISODES])
        self.n_lag_time = None
        # The lag time is the time an agent need to spent in the node to account for interactions at intersections.
        # Can be a fix number or the outcome of a surrogate model.
        self.__class__.count += 1
        Node.nodes.append(self)
        if verb:
            print(f'Node {self.n_uid} created')

    def get_number_of_edges(self):
        """Returns the number of edges connected to the calling node."""
        pass

    def get_next_node(self, edge):
        """Returns the node at the end of the given edge."""
        pass

    def get_current_state(self):
        """Returns the congestion condition of edges at the node."""
        pass

    def update_state(self):
        """Updates the variable [n_State]."""
        self.n_State = self.get_current_state()
        return

    def get_next_action(self):
        """Returns the next action from the QTable."""
        pass

    def update_action(self):
        """Updates the variable [n.Action] to the next suggested action."""
        self.n_Action = self.get_next_action()
        pass
