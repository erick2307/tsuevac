#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:02:03 2020

@author: luismoya
"""

import numpy as np
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing evacrl
from evacrl import paths

def bldClosestNode(bDB, nodesDB):
    bDBNode= np.zeros( (bDB.shape[0],3) )
    
    for i, b in enumerate(bDB):
        # print(i, b)
        coordBld= b[1:3]
        dist= (nodesDB[:,1] - coordBld[0])**2 + (nodesDB[:,2] - coordBld[1])**2
        indx= np.argmin(dist)
        bDBNode[i,:] = [i, indx, dist[indx]]
    return bDBNode


def setPopDB(area="kochi"):
    # Builds cases/<area>/data/agentsdb.csv from the census databases (datasets/census) and the
    # nodes of the same case; the census data describe the old Kochi area.
    nodesDB= np.loadtxt(paths.case_path(area, "data", "nodesdb.csv"), delimiter=",", skiprows= 1)
    print(nodesDB)
    popPaths= sorted(glob.glob( os.path.join(paths.CENSUS_DIR, "Population_database", "Pop_Code*.csv") ))  # sorted: glob order is filesystem dependent
    # print(bldPaths)
    
    popAllAreas= []
    
    for pfp in popPaths:
        # print(pfp)
        pDB= np.loadtxt(pfp, delimiter= ",", skiprows= 1, dtype= int)
        codeArea= pfp.split("_")[-1]
        hfp= os.path.join( paths.CENSUS_DIR, "Household_database" , "HH_" + codeArea )
        hDB= np.loadtxt(hfp, delimiter= ",", skiprows= 1, dtype= int)
        # print(hDB)
        bfp= os.path.join( paths.CENSUS_DIR, "CensusAndBuildingDatabase", "BldDb_" + codeArea )
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
            popAllAreas.append( [ pc[1] , pc[2] , pc[3] , pc[4] , pedCoord[j,1] ] ) 
            # , pedCoord[j,2] ] )
    
        # break
    np.savetxt(paths.case_path(area, "data", "agentsdb.csv"), np.array(popAllAreas), delimiter=",", fmt='%d',
               header="age,gender,hhType,hhId,Node")
    return

if __name__ == "__main__":
    setPopDB()