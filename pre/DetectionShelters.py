#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:55:05 2020

@author: luismoya
"""
import numpy as np
import matplotlib.pyplot as plt

nodesdb= np.loadtxt("nodesdb.csv", delimiter= ",", skiprows=0)
shelterdb= np.loadtxt("ShelterCoordinates.csv", delimiter=",", usecols=(0,1), skiprows=1) 
nodesdbV2= np.zeros(( nodesdb.shape[0] , nodesdb.shape[1]+2 ))
#print(nodesdb[:,1:3])
print(nodesdbV2.shape)
nodesdbV2[:,:3] = nodesdb[:,:3]

print("Closest node for shelters")
for i in range(shelterdb.shape[0]):
    x0, y0= shelterdb[i,:]
    # print(x0, y0)
    dist= ( (nodesdb[:,1] - x0)**2 + (nodesdb[:,2] - y0)**2 )**0.5
    indx= np.argmin(dist)
    # print(indx)
    # print(nodesdb[indx, :])
    nodesdbV2[indx,3] = 1

nodesdbV2[:,4] += 1

np.savetxt("nodesdbV2.csv", nodesdbV2, delimiter=",", header="number,coord_x,coord_y, evacuation, reward",
           fmt="%d,%.6f,%.6f,%d,%d" )

# plt.figure(num="plotNodes")
# plt.scatter(nodesdb[:,1], nodesdb[:,2])
# plt.scatter(shelterdb[:,0], shelterdb[:,1], facecolor="None", edgecolor='red', linewidths=2)
# plt.show()