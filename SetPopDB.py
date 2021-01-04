#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:02:03 2020

@author: luismoya
"""

import numpy as np
import glob
import os

def bldClosestNode(bDB, nodesDB):
    bDBNode= np.zeros( (bDB.shape[0],3) )
    
    for i, b in enumerate(bDB):
        # print(i, b)
        coordBld= b[1:3]
        dist= (nodesDB[:,1] - coordBld[0])**2 + (nodesDB[:,2] - coordBld[1])**2
        indx= np.argmin(dist)
        bDBNode[i,:] = [i, indx, dist[indx]]
    return bDBNode


def setPopDB():
    nodesDB= np.loadtxt("KochiNodesDB.csv", delimiter=",")
    print(nodesDB)
    popPaths= glob.glob( os.path.join("Population_database","Pop_Code*.csv") )
    # print(bldPaths)
    
    popAllAreas= []
    
    for pfp in popPaths:
        # print(pfp)
        pDB= np.loadtxt(pfp, delimiter= ",", skiprows= 1, dtype= np.int)
        codeArea= pfp.split("_")[-1]
        hfp= os.path.join( "Household_database" , "HH_" + codeArea )
        hDB= np.loadtxt(hfp, delimiter= ",", skiprows= 1, dtype= np.int)
        # print(hDB)
        bfp= os.path.join( "CensusAndBuildingDatabase", "BldDb_" + codeArea )
        bDB= np.loadtxt(bfp, delimiter= ",", skiprows= 1)
        if (len(bDB.shape) == 1) and (bDB.shape[0] == 5):
            bDB = bDB.reshape((1,5))
        
        print(pfp)
        print(hfp)
        print(bfp)
        print(pDB.shape, hDB.shape, bDB.shape)
        
        if not pDB.shape[0]:
            continue
        
        bDBNode= bldClosestNode(bDB, nodesDB)
        
        
        hhIndx= pDB[:,-1]
        bldIndx= hDB[hhIndx, -1]
        pedCoord= bDBNode[bldIndx-1, :]
        print(pDB.shape)
        print(hhIndx.shape)
        print(bldIndx.shape)
        print(pedCoord.shape)
        print(pedCoord)
        
        # age,gender,hhType,hhId,Node
        for j, pc in enumerate(pDB):
            if pedCoord[j,2] > 500:
                continue
            popAllAreas.append( [ pc[1] , pc[2] , pc[3] , pc[4] , pedCoord[j,1] , pedCoord[j,2] ] )
    
        # break
    np.savetxt("pedProfilesKochi.csv", np.array(popAllAreas), delimiter=",", 
               header="age,gender,hhType,hhId,Node")
    return

if __name__ == "__main__":
    setPopDB()