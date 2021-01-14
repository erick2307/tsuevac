#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:46:53 2020

@author: luismoya
"""

import numpy as np
import os
from sarsa import *
#from AgentsEvacuation_MonteCarlo_Aug2020 import *
import time
import matplotlib.pyplot as plt

def survivorsVsTime(numfiles,simtime=30,pop=35930,allfiles=True):
    if allfiles:
        simNum=np.arange(0,numfiles)            
    else:
        simNum= np.arange(0,numfiles,250)
        simNum= np.append(simNum,numfiles)
    
    #folder= "Arahama_SARSA_2020Oct19"
    timeSimulation= simtime*60
    agentsProfileName = "data/agentsdb.csv"
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = 15*60
    
    
    sM = np.zeros(( int(timeSimulation) , len(simNum)+1 )) 
    
    for i , sn in enumerate(simNum):
        fileName= "state/sim_%09d.csv" % sn
        #fn = os.path.join(folder, fileName)
        print(fileName)
        
        arahama = SARSA(agentsProfileName = agentsProfileName , meanRayleigh = meanRayleighTest)
        arahama.loadStateMatrixFromFile(namefile = fileName)
        # j= 0
        
        for t in range( int(min(arahama.pedDB[:,9])) , timeSimulation  ):
            arahama.initEvacuationAtTime()
            arahama.stepForward()
            optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
            arahama.checkTarget(ifOptChoice = optimalChoice)
            if not t % 1:
                arahama.computePedHistDenVelAtLinks()
                arahama.updateVelocityAllPedestrians()
                sM[int(t),0] = arahama.time
                sM[int(t),i+1] = np.sum( arahama.pedDB[:,10] == 1 )
                # j+=1
            if not t % 100:
                print(t) 
        arahama= None
    np.savetxt("survivor-time_v3.csv", sM, delimiter=",", fmt= "%d")
    return

def plotSurvivors(numfiles,simtime=30,pop=35930,allfiles=True):    
    if allfiles:
        simNum=np.arange(0,numfiles)            
    else:
        simNum= np.arange(0,numfiles,250)
        simNum= np.append(simNum,numfiles)

    db= np.loadtxt("survivor-time_v3.csv", delimiter=",")
    maxEvac=np.amax(db)
    maxCase=np.argmax(db[-1,1:])
    print(simNum[4],maxEvac)

    plt.figure(num="survivors")
    for i,sn in enumerate(simNum):
        if sn == 0:
            color= "b"
            lineW= 2
        elif i == maxCase: #sn == numfiles - 1:
            color= "r"
            lineW= 2
        else:
            color= "gray"
            lineW= 1
    
        plt.plot(db[:,0]/60., db[:,i+1]/pop*100, c= color, lw= lineW)
        # plt.text(db[-1,0], db[-1,i+1], "%d" % (sn))
        
    # plt.legend()
    plt.vlines(x = simtime, ymin = 0, ymax = 100, color = 'darkblue', linewidth=1, linestyle='--')
    plt.text(simtime + 0.5,25, "Tsunami Arrival time", size=12, color='darkblue',rotation=90);
    plt.xlabel("Time (min)")
    plt.ylabel("Rate of evacuation")
    plt.ylim(0,100)
    plt.savefig("NumberOfEvacuatedPedestrians_v3.png")
    
    plt.figure(num="survTotal")
    plt.plot(simNum, db[-1,1:]/pop*100, c="k")
    plt.scatter(simNum, db[-1,1:]/pop*100, c="r")
    plt.ylim(0,100)
    plt.xlabel("Episode number")
    plt.ylabel("Rate of total evacuation")
    plt.savefig("NumberOfEvacuatedPedestriansPerEpisode_v3.png")

    
    return

if __name__ == "__main__":
    population = 35930
    simtime = 30 #simulation time in minutes
    numfiles = 1526
    #survivorsVsTime(numfiles,simtime,population,False)
    plotSurvivors(numfiles,simtime,population,False)
    