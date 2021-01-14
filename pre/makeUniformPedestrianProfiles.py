#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:32:29 2020

@author: luismoya
"""

import numpy as np

nodesdb= np.loadtxt("nodesdbV2.csv", delimiter= ",", skiprows=1)

numPedPerNode= 10
pedprof= []

for i in range(nodesdb.shape[0]):
    if i == 2877:
        continue
    if not nodesdb[i,3]:
        for j in range(numPedPerNode):
            pedprof.append([18,1,0,0,nodesdb[i,0]])

np.savetxt("pedestrianProfiles_UniformDistributed_%dPerNode.csv" % (numPedPerNode), 
           np.array(pedprof), header= "age,gender,hhType,hhId,Node", delimiter= ",", fmt="%d") 