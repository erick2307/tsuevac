#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:52:07 2020

@author: luismoya
"""
import numpy as np
import os
from sarsa import *

def run(numSim0=0, numBlocks= 5, simPerBlock= 1000):
    t0 = time.time()
    agentsProfileName= os.path.join("data","agentsdb.csv")
    nodesdbFile= os.path.join("data","nodesdb.csv")
    linksdbFile= os.path.join("data", "linksdb.csv")
    transLinkdbFile= os.path.join("data", "actionsdb.csv")
    transNodedbFile= os.path.join("data", "transitionsdb.csv")
    folderStateNames = "state"
    meanRayleighTest = 15*60
    simulTime = 30*60
    survivorsPerSim= []
    
    if numSim0 == 0:
        randomChoiceRate = 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        kochi = SARSA(agentsProfileName = agentsProfileName , 
                      nodesdbFile= nodesdbFile,
                      linksdbFile= linksdbFile, 
                      transLinkdbFile= transLinkdbFile, 
                      transNodedbFile= transNodedbFile,
                      meanRayleigh = meanRayleighTest)
        
        for t in range( int(min(kochi.pedDB[:,9])) , int(min(max(kochi.pedDB[:,9]) , simulTime)) ):
            print(t)
            kochi.initEvacuationAtTime()
            kochi.stepForward()
            optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
            kochi.checkTarget(ifOptChoice = optimalChoice)   
            if not t % 10:
                kochi.computePedHistDenVelAtLinks()
                kochi.updateVelocityAllPedestrians()
        outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim0)
        kochi.exportStateMatrix(outnamefile = outfile)
        print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim0, (time.time()-t0)/60. ))
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print("survived pedestrians: %d" % np.sum(kochi.pedDB[:,10] == 1) )
        survivorsPerSim.append([numSim0, np.sum(kochi.pedDB[:,10] == 1)])
        kochi= None
    
    numSim= numSim0 +1
    for b in range(numBlocks):
        for s in range(simPerBlock):
            randomChoiceRate = 1.0/(0.015*s + 1.0)
            optimalChoiceRate = 1.0 - randomChoiceRate
            kochi = SARSA(agentsProfileName = agentsProfileName , 
                          nodesdbFile= nodesdbFile,
                          linksdbFile= linksdbFile, 
                          transLinkdbFile= transLinkdbFile, 
                          transNodedbFile= transNodedbFile,
                          meanRayleigh = meanRayleighTest)
            namefile = os.path.join(folderStateNames , "sim_%09d.csv" % (numSim-1) )
            kochi.loadStateMatrixFromFile(namefile = namefile)
            for t in range( int(min(kochi.pedDB[:,9])) , int(min(max(kochi.pedDB[:,9]) , simulTime)) ):
                kochi.initEvacuationAtTime()
                kochi.stepForward()
                optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
                kochi.checkTarget(ifOptChoice = optimalChoice)
                if not t % 10:
                    kochi.computePedHistDenVelAtLinks()
                    kochi.updateVelocityAllPedestrians()
            outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim)
            kochi.exportStateMatrix(outnamefile = outfile)
            print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim , (time.time()-t0)/60. ))
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print("survived pedestrians: %d" % np.sum(kochi.pedDB[:,10] == 1) )
            survivorsPerSim.append([numSim, np.sum(kochi.pedDB[:,10] == 1)])
            kochi= None
            numSim += 1
    outSurvivors= os.path.join(folderStateNames, "survivorsPerSim.csv")
    np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter= ",", fmt= "%d" )
    return 

def main():  
    numSim0= 0
    numBlocks= 5
    simPerBlock= 1000
    run(numSim0= numSim0, numBlocks= numBlocks, simPerBlock= simPerBlock) 
    return  

def test():
    t0 = time.time()
    agentsProfileName= os.path.join("data","agentsdb.csv")
    nodesdbFile= os.path.join("data","nodesdb.csv")
    linksdbFile= os.path.join("data", "linksdb.csv")
    transLinkdbFile= os.path.join("data", "actionsdb.csv")
    transNodedbFile= os.path.join("data", "transitionsdb.csv")
    folderStateNames = "state"
    meanRayleighTest = 14*60
    simulTime = 30*60
    
    kochi = SARSA(agentsProfileName = agentsProfileName , 
                nodesdbFile= nodesdbFile,
                linksdbFile= linksdbFile, 
                transLinkdbFile= transLinkdbFile, 
                transNodedbFile= transNodedbFile,
                meanRayleigh = meanRayleighTest)
    
    print(kochi.pedDB)
    print(kochi.pedDB[32649,:])
    return
    
if __name__ == "__main__":
    # test()
    main()
