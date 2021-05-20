#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:27:09 2020
"""

import numpy as np
import os
from sarsa import *
import time

def createVideo(filename, foldername, area="kochi",timeSimulation= 30*60, meandeparture=15*60):
    #setup
    t0 = time.time() 
    fn = os.path.join(area,foldername,filename)
    videoNamefile = f"sarsa_{area}_{filename[:-4]}.avi" 
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = meandeparture

    #load files
    agentsProfileName= os.path.join(area,"data","agentsdb.csv")
    nodesdbFile= os.path.join(area,"data","nodesdb.csv")
    linksdbFile= os.path.join(area,"data", "linksdb.csv")
    transLinkdbFile= os.path.join(area,"data", "actionsdb.csv")
    transNodedbFile= os.path.join(area,"data", "transitionsdb.csv")
    
    #initiate class
    case = SARSA(agentsProfileName = agentsProfileName , 
                nodesdbFile= nodesdbFile,
                linksdbFile= linksdbFile, 
                transLinkdbFile= transLinkdbFile, 
                transNodedbFile= transNodedbFile,
                meanRayleigh = meanRayleighTest,
                discount= 0.9,
                folderStateNames= foldername)

    #input policy
    case.loadStateMatrixFromFile(namefile = fn)
    
    #output population initial condition
    outnamefile = os.path.join(area,"results","agents_startcondition.csv")
    case.exportAgentDBatTimet(outnamefile)
    
    #setup canvas
    case.setFigureCanvas()
    
    #start simulation
    for t in range( int(min(case.pedDB[:,9])) , timeSimulation  ):
        case.initEvacuationAtTime()
        case.stepForward()
        optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
        case.checkTarget(ifOptChoice = optimalChoice)
        if not t % 10: 
            print(t) 
            case.getSnapshotV2()
            case.computePedHistDenVelAtLinks()
            case.updateVelocityAllPedestrians()

    #output population condition
    outnamefile = os.path.join(area,"results","agents_finalcondition.csv")
    case.exportAgentDBatTimet(outnamefile)   
    
    #output population path and time
    fname = os.path.join("results","agents_experience.csv")
    np.savetxt(fname,case.expeStat,delimiter=',')
    
    case.makeVideo(nameVideo = videoNamefile)
    case.destroyCanvas()
    #case.deleteFigures()
    case= None
    print("\n***** Video created (%.2f seconds) *****" % (time.time() - t0)) 
    return

def main():      
    filename= "sim_000001526.csv" #name of state matrix to load
    area = "kochi"
    foldername= "state" #folder where state matrices are saved
    timeSimulation= 30*60 #total time of simulation
    meandeparture = 15*60 #this is the actual evacuation behavior (not necessary the trained behavior)
    createVideo(filename= filename, foldername= foldername, area=area,timeSimulation= timeSimulation, meandeparture=meandeparture)
    return 

if __name__ == "__main__":    
    main()
    
    
    
    