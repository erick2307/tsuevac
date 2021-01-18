#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:46:53 2020
"""

import numpy as np
import os
from sarsa import *
from mc import *
import matplotlib.pyplot as plt

def survivorsVsTime(numfiles,simtime=30,pop=35930,meandeparture=15,allfiles=True,blocks=250):
    if allfiles:
        simNum=np.arange(0,numfiles)            
    else:
        simNum= np.arange(0,numfiles,blocks)
        simNum= np.append(simNum,numfiles)
    
    simulTime= simtime*60
    agentsProfileName = os.path.join("data","agentsdb.csv")
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = meandeparture*60
    
    
    sM = np.zeros(( int(simulTime) , len(simNum)+1 )) 
    
    for i , sn in enumerate(simNum):
        fileName= os.path.join("state","sim_%09d.csv" % sn)
        print(fileName)
        
        case = SARSA(agentsProfileName = agentsProfileName , meanRayleigh = meanRayleighTest)
        case.loadStateMatrixFromFile(namefile = fileName)
        
        for t in range( int(min(case.pedDB[:,9])) , simulTime  ):
            case.initEvacuationAtTime()
            case.stepForward()
            optimalChoice = bool(np.random.choice(2, 1, p=[randomChoiceRate , optimalChoiceRate]))
            case.checkTarget(ifOptChoice = optimalChoice)
            if not t % 10:
                print(t) 
                case.computePedHistDenVelAtLinks()
                case.updateVelocityAllPedestrians()
                sM[int(t),0] = case.time
                sM[int(t),i+1] = np.sum( case.pedDB[:,10] == 1 )
        case= None
    folder=os.path.join("results","survivor-time.csv")
    np.savetxt(folder, sM, delimiter=",", fmt= "%d")
    return

def plotSurvivors(numfiles,simtime=30,pop=35930,allfiles=True,blocks=250):    
    if allfiles:
        simNum=np.arange(0,numfiles)            
    else:
        simNum= np.arange(0,numfiles,blocks)
        simNum= np.append(simNum,numfiles)

    folder=os.path.join("results","survivor-time.csv")
    db= np.loadtxt(folder, delimiter=",")
    maxEvac=np.amax(db)
    print(f"The max number of evacuees was {maxEvac}.")
    maxCase=np.argmax(db[-1,1:])

    plt.figure(num="survivors")
    for i,sn in enumerate(simNum):
        if sn == 0:
            color= "b"
            lineW= 2
        #plotting the maximum case in red. Not necessary the latest.
        #BEFORE: sn == numfiles - 1:
        elif i == maxCase: 
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
    fout=os.path.join("results","evacuation_rate.png")
    plt.savefig(fout)
    
    plt.figure(num="survTotal")
    plt.plot(simNum, db[-1,1:]/pop*100, c="k")
    plt.scatter(simNum, db[-1,1:]/pop*100, c="r")
    plt.ylim(0,100)
    plt.xlabel("Episode number")
    plt.ylabel("Rate of total evacuation")    
    fout=os.path.join("results","evacuation_per_episode.png")
    plt.savefig(fout)
    return

if __name__ == "__main__":
    numfiles = 25
    simtime = 30 #simulation time in minutes
    population = 3593
    meandeparture = 15 #this is the actual evacuation behavior (not necessary the trained behavior)
    allfiles = True #to plot all (True) or by blocks (False)
    blocks=250 #size of blocks from the total number of files
    
    survivorsVsTime(numfiles,simtime,population,meandeparture,allfiles,blocks)
    plotSurvivors(numfiles,simtime,population,allfiles,blocks)
    