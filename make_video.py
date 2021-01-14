#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:27:09 2020

@author: luismoya
"""

import numpy as np
import os
from sarsa import *
#from AgentsEvacuation_MonteCarlo_Aug2020 import *
import time

def SARSA_simulationVideo(filename, foldername, timeSimulation= 60*60):
    t0 = time.time() 
    fn = os.path.join(foldername, filename)
    videoNamefile = "sarsa_kochi_%s.avi" % (filename[:-4])  
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = 15*60
    
    agentsProfileName= os.path.join("data","agentsdb.csv")
    nodesdbFile= os.path.join("data","nodesdb.csv")
    linksdbFile= os.path.join("data", "linksdb.csv")
    transLinkdbFile= os.path.join("data", "actionsdb.csv")
    transNodedbFile= os.path.join("data", "transitionsdb.csv")
    
    sarsaSimulator = SARSA(agentsProfileName = agentsProfileName , 
                nodesdbFile= nodesdbFile,
                linksdbFile= linksdbFile, 
                transLinkdbFile= transLinkdbFile, 
                transNodedbFile= transNodedbFile,
                meanRayleigh = meanRayleighTest)
    
    # arahama = SARSA(agentsProfileName = agentsProfileName , meanRayleigh = meanRayleighTest)
    sarsaSimulator.loadStateMatrixFromFile(namefile = fn)
    sarsaSimulator.setFigureCanvas()
    
    for t in range( int(min(sarsaSimulator.pedDB[:,9])) , timeSimulation  ):
        sarsaSimulator.initEvacuationAtTime()
        sarsaSimulator.stepForward()
        optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
        sarsaSimulator.checkTarget(ifOptChoice = optimalChoice)
        if not t % 1: 
            print(t) 
            sarsaSimulator.getSnapshotV2()
            sarsaSimulator.computePedHistDenVelAtLinks()
            sarsaSimulator.updateVelocityAllPedestrians()
    
    sarsaSimulator.makeVideo(nameVideo = videoNamefile)
    sarsaSimulator.destroyCanvas()
    sarsaSimulator.deleteFigures()
    sarsaSimulator= None
    print("\n***** Video created (%.2f seconds) *****" % (time.time() - t0)) 
    return

def MC_simulationVideo(filename, foldername, timeSimulation= 120*60):
    t0 = time.time() 
    fn = os.path.join(foldername, filename)
    videoNamefile = "montecarloV2_%s.avi" % (filename[:-4])
    agentsProfileName = "IPF_AgentsCoordV2.csv"
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = 14*60
    
    arahama = pedestrianMonteCarlo(agentsProfileName = agentsProfileName , meanRayleigh = meanRayleighTest) 
    arahama.loadStateMatrixFromFile(namefile = fn)
    arahama.setFigureCanvas()
    
    for t in range( int(min(arahama.pedDB[:,9])) , timeSimulation  ):
        arahama.initEvacuationAtTime()
        arahama.stepForward()
        optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
        arahama.checkTarget(ifOptChoice = optimalChoice)
        if not t % 5:
            print(t)
            arahama.getSnapshotV2()
            arahama.computePedHistDenVelAtLinks()
            arahama.updateVelocityAllPedestrians() 
    arahama.makeVideo(nameVideo = videoNamefile)
    arahama.destroyCanvas()
    arahama.deleteFigures()
    arahama = None  
    print("\n***** Video created (%.2f seconds) *****" % (time.time() - t0)) 
    return

def main():
    # filename= "sim_7627.csv"
    # foldername= "Arahama_MC_2020Oct08"
    # timeSimulation= 90*60
    # MC_simulationVideo(filename= filename, foldername= foldername, timeSimulation= timeSimulation) 
    
    filename= "sim_000001526.csv"
    foldername= "state"
    timeSimulation= 30*60  
    SARSA_simulationVideo(filename= filename, foldername= foldername, timeSimulation= timeSimulation)
    return 

if __name__ == "__main__":    
    main()
    
    
    
    