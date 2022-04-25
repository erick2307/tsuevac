from tsuevac import setup

class Agent(object):
    """An agent class"""
    count = 0
    agents = []

    def __init__(self, verb=False):
        self.who = self.count  # unique agent id
        self.color = 'k'
        self.heading = 0.0
        self.xcor = 0.0
        self.ycor = 0.0
        self.shape = "default"
        self.label = ""
        self.label_color = 'k'
        self.breed = None
        self.hidden = False
        self.size = 1
        self.pen_size = 1
        self.pen_mode = "up"
        self.__class__.count += 1
        Agent.agents.append(self)
        if verb:
            print(f'Agent {self.a_uid} created')
