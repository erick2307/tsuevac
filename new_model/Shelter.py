from medabm import *

class Hospitals(object):
    """A hospital class"""
    count = 0
    hosps = []
    def __init__(self,verb=False):
        self.h_uid = self.count
        self.h_name = ""
        self.h_type = None
        self.h_heliport = None
        self.h_capacity = 0 #a number of max in-patients
        self.h_home = None #location of the building
        self.h_home_coordinate = None
        self.h_state = 0 #situation of behavior e.g. 0:available; 1:unavailable
        self.h_staff = 0 #number of doctors
        self.h_damage = 0 #level or flag of damage 0 is none
        self.h_inpatients = 0 #number of patients
        self.h_inpatientsList = []
        #to control uid and an aggregated list
        self.__class__.count +=1
        Hospitals.hosps.append(self)
