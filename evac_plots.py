#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from sarsa import *
from mc import *
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
import time

def survivorsVsTime(numfiles,startfile=0,simtime=30,pop=35930,meandeparture=15,
                    allfiles=True,blocks=250,casealias='case1',statefolder = "case_u30min"):
    if allfiles:
        simNum=np.arange(startfile,startfile+numfiles)            
    else:
        simNum= np.arange(startfile,startfile+numfiles,blocks)
        simNum= np.append(simNum,startfile+numfiles)
    
    simulTime= simtime*60
    agentsProfileName = os.path.join("data","agentsdb.csv")
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = meandeparture*60
    
    
    sM = np.zeros(( int(simulTime) , len(simNum)+1 )) 
    
    for i , sn in enumerate(simNum):
        fileName= os.path.join(f"state_{statefolder}","sim_%09d.csv" % sn)
        print(fileName)
        
        case = SARSA(agentsProfileName = agentsProfileName , 
                     meanRayleigh = meanRayleighTest, folderStateNames=statefolder)
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
            sM[int(t),0] = int(t) #case.time
            sM[int(t),i+1] = np.sum(case.pedDB[:,10] == 1 )
        case= None
    folder=os.path.join("results",f"survivor-time_{casealias}.csv")
    np.savetxt(folder, sM, delimiter=",", fmt= "%d")
    return

def plotSurvivors(numfiles,simtime=30,pop=35930,meandeparture=15,allfiles=True,blocks=250,casealias="case1",cls='b--'):    
    if allfiles:
        simNum=np.arange(startfile,startfile+numfiles)            
    else:
        simNum= np.arange(startfile,startfile+numfiles,blocks)
        simNum= np.append(simNum,startfile+numfiles)

    folder=os.path.join("results",f"survivor-time_{casealias}.csv")
    db= np.loadtxt(folder, delimiter=",")
    #maxCase=np.argmax(db[-1,1:])
    maxCase=pd.DataFrame(db[:,1:]).idxmax(axis=1).mode().item()
    maxEvac=db[-1,maxCase]
    print(f"The max number of evacuees was {maxEvac} in Case {maxCase}.")    

    plt.figure(num="survivors")
    for i,sn in enumerate(simNum):
        if sn == 0:
            color= "b"
            lineW= 2
            alf=1.0
        #plotting the maximum case in red. Not necessary the latest.
        #BEFORE: sn == numfiles - 1:
        elif i == maxCase: 
            color= "r"
            lineW= 2
            alf=1.0
        else:
            color= "gray"
            lineW= 0.5
            alf=0.6
    
        plt.plot(db[:,0]/60., db[:,i+1]/pop, c= color, lw= lineW,alpha=alf)
        # plt.text(db[-1,0], db[-1,i+1], "%d" % (sn))
        
    # plt.legend()
    plt.vlines(x = simtime, ymin = 0, ymax = 1, color = 'darkblue', linewidth=1, linestyle='--')
    plt.text(simtime + 0.5,0.25, "Tsunami Arrival time", size=12, color='darkblue',rotation=90);
    plt.xlabel("Time (min)")
    plt.ylabel("Rate of evacuation (%)")
    plt.xlim(0,simtime+5)
    plt.ylim(0,1)
    
    scale = 1
    loc = 0
    sc = meandeparture
    x = np.linspace(0,simtime,1000)
    plt.plot(x, scale*rayleigh.cdf(x,loc,sc),cls, lw=2,alpha=0.9, label='Fast')    
    fout=os.path.join("results",f"evacuation_rate_{casealias}.png")
    plt.savefig(fout)

    
    plt.figure(num="survTotal")
    plt.plot(simNum, db[-1,1:]/pop, c="k")
    plt.scatter(simNum, db[-1,1:]/pop, c="r")
    plt.ylim(0,1)
    plt.xlabel("Episode number")
    plt.ylabel("Rate of total evacuation (%)")    
    fout=os.path.join("results",f"evacuation_per_episode_{casealias}.png")
    plt.savefig(fout)
    plt.close("all")
    return maxEvac

if __name__ == "__main__":
    t0 = time.time()
    numfiles = 1000
    startfile  = 0
    simtime = 67 #simulation time in minutes
    population = 2723 #Arahama #3593 #Kochi
    meandeparture = [ 5,15,30 ] #this is the actual evacuation behavior (not necessary the trained behavior)
    colors = [ 'g--','b--','r--' ]
    allfiles = False #to plot all (True) or by blocks (False)
    blocks=50 #size of blocks from the total number of files (i.e. numfiles/blocks will be simulated)
    casealias = ["case_u30min_5min","case_u30min_15min","case_u30min_30min"]
    statefolder = "case_u30min"
    
    for casealias,meandeparture,cls in zip(casealias,meandeparture,colors):
        survivorsVsTime(numfiles,startfile,simtime,population,meandeparture,allfiles,blocks,casealias,statefolder = "case_u30min")
        maxEvac = plotSurvivors(numfiles,simtime,population,meandeparture,allfiles,blocks,casealias,cls)
        print(f"Time: {time.time()-t0} s.")
        # os.system(f"osascript -e 'Tell application \"System Events\" "
                #   f"to display dialog \"Max Evacuees:{maxEvac}\"'")
    