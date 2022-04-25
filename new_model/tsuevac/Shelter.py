from tsuevac import Agent
from tsuevac import setup

class Shelter(Agent):
    """A shelter class"""
    count = 0
    shelters = []

    def __init__(self, verb=False):
        super().__init__(self, verb)
        self.s_uid = self.count
        self.s_name = ""
        self.s_type = None
        self.s_heliport = None
        self.s_capacity = 0  # a number of max evacuees
        self.s_damage = 0  # level or flag of damage 0 is None
        self.__class__.count += 1
        Shelter.shelters.append(self)
