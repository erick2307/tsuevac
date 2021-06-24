#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from mc import MonteCarlo
plt.ioff() 

def simulationShortestPath():
    simulTime = 67*60
    sendai = MonteCarlo(agentsProfileName = "IPF_AgentsCoordV2.csv")
    sendai.loadShortestPathDB(namefile= "nextnode.csv")
    sendai.setFigureCanvas()
    survivedAgents = np.zeros((simulTime,3))
    for t in range( int(min(sendai.pedDB[:,9])) , max( min( int(max(sendai.pedDB[:,9])), simulTime), simulTime) ):
        sendai.initEvacuationAtTime()
        sendai.stepForward()
        if not t % 60:
            print(t)
            sendai.getSnapshotV2()
        sendai.checkTargetShortestPath()
        survivedAgents[t,0] = np.sum( np.isin(sendai.pedDB[:,8], sendai.evacuationNodes[0]) )
        survivedAgents[t,1] = np.sum( np.isin(sendai.pedDB[:,8], sendai.evacuationNodes[1]) )
        survivedAgents[t,2] = np.sum( np.isin(sendai.pedDB[:,8], sendai.evacuationNodes[2]) )
    sendai.makeVideo(nameVideo="Simulation20191211_shortestPath_OriginalCensus.avi") 
    sendai.destroyCanvas()  
    sendai.deleteFigures()
    sendai = None
    np.savetxt("agentsAtEvacuationNodesVsTime_shortesPath_OriginalCensus.csv", survivedAgents, delimiter=",")
    return

def ArahamaMTRL_20191220_SeqSim():
    t0 = time.time()
    simulTime = 67*60   #T*60 sec of tsunami arrival time
    agentsProfileName = "IPF_AgentsCoordV2.csv"
    folderStateNames = "Arahama_20191220"
    numMaxSim = 5 #20
    optimalChoiceRate = 0.9
    randomChoiceRate = 1.0 - optimalChoiceRate
    survivedAgentsPerSimName = "survivedAgents_Arahama20191220.csv"
    meanRayleighTest = 20*60
    
    survivedAgents = np.zeros(numMaxSim)
    arahama = MonteCarlo(agentsProfileName = agentsProfileName, meanRayleigh = meanRayleighTest)
    
    numSim= 0
    for t in range( int(min(arahama.pedDB[:,9])) , int(min(max(arahama.pedDB[:,9]) , simulTime))  ):
        arahama.initEvacuationAtTime()
        arahama.stepForward()
        arahama.checkTarget()
    arahama.updateValueFunctionDB()
    survivedAgents[0] = np.sum( np.isin(arahama.pedDB[:,8], arahama.evacuationNodes) )
    outfile = os.path.join(folderStateNames,"sim_%04d.csv" % numSim)
    outfilepedDB = os.path.join(folderStateNames, "ped_%04d.csv" % numSim)
    arahama.exportStateMatrix(outnamefile = outfile)
    arahama.exportAgentDBatTimet(outnamefile = outfilepedDB)
    arahama = None 
    
    for s in range(1,numMaxSim):
        print("simulation number %d , t = %.1f" % ( s , time.time()-t0 )) 
        arahama = MonteCarlo(agentsProfileName = agentsProfileName, meanRayleigh = meanRayleighTest)
        namefile = os.path.join(folderStateNames , "sim_%04d.csv" % (s-1) )
        arahama.loadStateMatrixFromFile(namefile = namefile)
        
        for t in range( int(min(arahama.pedDB[:,9])) , simulTime  ):
            arahama.initEvacuationAtTime()
            arahama.stepForward()
            optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
            arahama.checkTarget(ifOptChoice = optimalChoice)
        arahama.updateValueFunctionDB()
        survivedAgents[s] = np.sum( np.isin(arahama.pedDB[:,8], arahama.evacuationNodes) ) 
        outfile = os.path.join(folderStateNames , "sim_%04d.csv" % s)
        outfilepedDB = os.path.join(folderStateNames , "ped_%04d.csv" % s)
        arahama.exportStateMatrix(outnamefile = outfile)
        arahama.exportAgentDBatTimet(outnamefile = outfilepedDB)
    np.savetxt(survivedAgentsPerSimName, survivedAgents, delimiter=",") 
    
    QFun, VFun, policy  = arahama.computeAction_Value_Policy()  #computeAction_Value_Policy
    return

def ArahamaMTRL_20191220_Video(): 
    folderStateNames = "Arahama_20191220"
    stateSimFile = "sim_0498.csv"
    namefile = os.path.join(folderStateNames, stateSimFile) 
    fileNameAgentsAtEvacNodevsTime = "agentsAtEvacuationNodesVsTime.csv"
    videoNamefile = "ArahamaTest2020Oct06.avi"
    meanRayleighTest = 20*60
    simulTime = 67*60
    agentsProfileName = "IPF_AgentsCoordV2.csv"
    optimalChoiceRate = 0.9
    randomChoiceRate = 1.0 - optimalChoiceRate
    
    survivedAgents = np.zeros((simulTime,3))
    
    arahama = MonteCarlo(agentsProfileName = agentsProfileName , meanRayleigh = meanRayleighTest)
    arahama.loadStateMatrixFromFile(namefile = namefile)
    arahama.setFigureCanvas()
    
    for t in range( int(min(arahama.pedDB[:,9])) , simulTime  ):
        arahama.initEvacuationAtTime()
        arahama.stepForward()
        optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
        arahama.checkTarget(ifOptChoice = optimalChoice)
        if not t % 5:
            print(t)
            arahama.getSnapshotV2()
            arahama.computePedHistDenVelAtLinks()
            arahama.updateVelocityAllPedestrians() 
        survivedAgents[t,0] = np.sum( np.isin(arahama.pedDB[:,8], arahama.evacuationNodes[0]) )
        survivedAgents[t,1] = np.sum( np.isin(arahama.pedDB[:,8], arahama.evacuationNodes[1]) )
        survivedAgents[t,2] = np.sum( np.isin(arahama.pedDB[:,8], arahama.evacuationNodes[2]) )
    
    arahama.makeVideo(nameVideo = videoNamefile)
    arahama.destroyCanvas()
    arahama.deleteFigures()
    arahama = None 
    np.savetxt(fileNameAgentsAtEvacNodevsTime, survivedAgents, delimiter=",")
    return 
    
def SurvivedAgentsPerEvacuationNode():
    fileNameAgentsAtEvacNodevsTime = "agentsAtEvacuationNodesVsTime.csv"
    rlDb = np.loadtxt(fileNameAgentsAtEvacNodevsTime, delimiter=',')
    
    plt.figure(num="comparison")
    plt.subplot(1,4,1)
    plt.plot(rlDb[:,0], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 1")
    plt.grid()
    plt.legend()
    
    plt.subplot(1,4,2)
    plt.plot(rlDb[:,1], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 2")
    plt.grid()
    
    plt.subplot(1,4,3)
    plt.plot(rlDb[:,2], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 3")
    plt.grid()
    
    plt.subplot(1,4,4)
    plt.plot(np.sum(rlDb , axis=1), color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at nodes")
    plt.title("All evacuation nodes")
    plt.grid()
    
    plt.show()
    return

if __name__ == "__main__":
    #SurvivedAgentsPerEvacuationNode()
    # ArahamaMTRL_20191220_Video()
    ArahamaMTRL_20191220_SeqSim()
