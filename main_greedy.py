#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:52:07 2020
"""
from mc import MonteCarlo
import numpy as np
import os
from sarsa import *

def run_sarsa(area="kochi",simtime=30, meandeparture=15, numSim0=0, numBlocks= 5, simPerBlock= 1000,name='r'):
    t0 = time.time()
    agentsProfileName= os.path.join(area,"data","agentsdb.csv")
    nodesdbFile= os.path.join(area,"data","nodesdb.csv")
    linksdbFile= os.path.join(area,"data", "linksdb.csv")
    transLinkdbFile= os.path.join(area,"data", "actionsdb.csv")
    transNodedbFile= os.path.join(area,"data", "transitionsdb.csv")
    folderStateNames = os.path.join(area,f"state_{name}")
    if not os.path.exists(folderStateNames):
        os.mkdir(folderStateNames)
    meanRayleighTest = meandeparture*60
    simulTime = simtime*60
    survivorsPerSim= []
    
    if numSim0 == 0:
        randomChoiceRate = 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        case = SARSA(agentsProfileName = agentsProfileName , 
                      nodesdbFile= nodesdbFile,
                      linksdbFile= linksdbFile, 
                      transLinkdbFile= transLinkdbFile, 
                      transNodedbFile= transNodedbFile,
                      meanRayleigh = meanRayleighTest,
                      discount=0.9,
                      folderStateNames= folderStateNames)
        
        #for t in range( int(min(case.pedDB[:,9])) , int(min(max(case.pedDB[:,9]) , simulTime)) ):
        for t in range( int(min(case.pedDB[:,9])) , simulTime ):
            case.initEvacuationAtTime()
            case.stepForward()
            optimalChoice = bool(np.random.choice(2, 1, p=[randomChoiceRate , optimalChoiceRate]))
            case.checkTarget(ifOptChoice = optimalChoice)   
            if not t % 10:
                case.computePedHistDenVelAtLinks()
                case.updateVelocityAllPedestrians()
                
        outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim0)
        case.exportStateMatrix(outnamefile = outfile)
        print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim0, (time.time()-t0)/60. ))
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print("survived pedestrians: %d" % np.sum(case.pedDB[:,10] == 1) )
        survivorsPerSim.append([numSim0, np.sum(case.pedDB[:,10] == 1)])
        case= None
    
    numSim= numSim0 +1
    for b in range(numBlocks):
        for s in range(simPerBlock):
            randomChoiceRate = (simPerBlock - s - 1.0)/(simPerBlock - s + 1.0) #1.0/(0.015*s + 1.0)
            optimalChoiceRate = 1.0 - randomChoiceRate
            case = SARSA(agentsProfileName = agentsProfileName , 
                          nodesdbFile= nodesdbFile,
                          linksdbFile= linksdbFile, 
                          transLinkdbFile= transLinkdbFile, 
                          transNodedbFile= transNodedbFile,
                          meanRayleigh = meanRayleighTest,
                          folderStateNames = folderStateNames)
            
            namefile = os.path.join(folderStateNames , "sim_%09d.csv" % (numSim-1) )
            case.loadStateMatrixFromFile(namefile = namefile)
            
            #for t in range( int(min(case.pedDB[:,9])) , int(min(max(case.pedDB[:,9]) , simulTime)) ):
            for t in range( int(min(case.pedDB[:,9])) , simulTime ):
                case.initEvacuationAtTime()
                case.stepForward()
                optimalChoice = bool(np.random.choice(2, 1, p=[randomChoiceRate , optimalChoiceRate]))
                case.checkTarget(ifOptChoice = optimalChoice)
                if not t % 10:
                    case.computePedHistDenVelAtLinks()
                    case.updateVelocityAllPedestrians()
                    
            outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim)
            case.exportStateMatrix(outnamefile = outfile)
            print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim , (time.time()-t0)/60. ))
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print("survived pedestrians: %d" % np.sum(case.pedDB[:,10] == 1) )

            #evaluate survivors in simulation
            survivorsPerSim.append([numSim, np.sum(case.pedDB[:,10] == 1)])
            fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}csv"
            outSurvivors= os.path.join(folderStateNames, fname)
            np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter= ",", fmt= "%d" )            
            
            case= None
            numSim += 1

    return 

def run_mc(area="kochi",simtime=30, meandeparture=15, numSim0=0, numBlocks= 5, simPerBlock= 1000,name='r'):
    t0 = time.time()
    agentsProfileName= os.path.join(area,"data","agentsdb.csv")
    nodesdbFile= os.path.join(area,"data","nodesdb.csv")
    linksdbFile= os.path.join(area,"data", "linksdb.csv")
    transLinkdbFile= os.path.join(area,"data", "actionsdb.csv")
    transNodedbFile= os.path.join(area,"data", "transitionsdb.csv")
    folderStateNames = os.path.join(area,f"state_{name}")
    if not os.path.exists(folderStateNames):
        os.mkdir(folderStateNames)
    meanRayleighTest = meandeparture*60
    simulTime = simtime*60
    survivorsPerSim= []

    if numSim0 == 0:
        randomChoiceRate = 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        case = MonteCarlo(agentsProfileName = agentsProfileName , 
                      nodesdbFile= nodesdbFile,
                      linksdbFile= linksdbFile, 
                      transLinkdbFile= transLinkdbFile, 
                      transNodedbFile= transNodedbFile,
                      meanRayleigh = meanRayleighTest,
                      folderStateNames= folderStateNames)
        
        for t in range( int(min(case.pedDB[:,9])) , simulTime ):
            case.initEvacuationAtTime()
            case.stepForward()
            optimalChoice = bool(np.random.choice(2, 1, p=[randomChoiceRate , optimalChoiceRate]))
            case.checkTarget(ifOptChoice = optimalChoice)   
            if not t % 10:
                case.computePedHistDenVelAtLinks()
                case.updateVelocityAllPedestrians()
        case.updateValueFunctionDB()        
        outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim0)
        outfilepedDB = os.path.join(folderStateNames , "ped_%09d.csv" % numSim0)
        case.exportStateMatrix(outnamefile = outfile)
        case.exportAgentDBatTimet(outnamefile = outfilepedDB)
        print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim0, (time.time()-t0)/60. ))
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print("survived pedestrians: %d" % np.sum(case.pedDB[:,10] == 1) )
        survivorsPerSim.append([numSim0, np.sum(case.pedDB[:,10] == 1)])
        fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}csv"
        outSurvivors= os.path.join(folderStateNames, fname)
        np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter= ",", fmt= "%d" )  

        case= None
    
    numSim= numSim0 +1
    for b in range(numBlocks):
        for s in range(simPerBlock):
            randomChoiceRate = (simPerBlock - s - 1.0)/(simPerBlock - s + 1.0) #1.0/(0.015*s + 1.0)
            optimalChoiceRate = 1.0 - randomChoiceRate
            case = MonteCarlo(agentsProfileName = agentsProfileName , 
                          nodesdbFile= nodesdbFile,
                          linksdbFile= linksdbFile, 
                          transLinkdbFile= transLinkdbFile, 
                          transNodedbFile= transNodedbFile,
                          meanRayleigh = meanRayleighTest,
                          folderStateNames = folderStateNames)
            
            namefile = os.path.join(folderStateNames , "sim_%09d.csv" % (numSim-1) )
            case.loadStateMatrixFromFile(namefile = namefile)
            
            #for t in range( int(min(case.pedDB[:,9])) , int(min(max(case.pedDB[:,9]) , simulTime)) ):
            for t in range( int(min(case.pedDB[:,9])) , simulTime ):
                case.initEvacuationAtTime()
                case.stepForward()
                optimalChoice = bool(np.random.choice(2, 1, p=[randomChoiceRate , optimalChoiceRate]))
                case.checkTarget(ifOptChoice = optimalChoice)
                if not t % 10:
                    case.computePedHistDenVelAtLinks()
                    case.updateVelocityAllPedestrians()
                    
            case.updateValueFunctionDB()
            outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim)
            outfilepedDB = os.path.join(folderStateNames , "ped_%09d.csv" % numSim)
            case.exportStateMatrix(outnamefile = outfile)
            case.exportAgentDBatTimet(outnamefile = outfilepedDB)
            print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim , (time.time()-t0)/60. ))
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print("survived pedestrians: %d" % np.sum(case.pedDB[:,10] == 1) )
            
            #evaluate survivors in simulation
            survivorsPerSim.append([numSim, np.sum(case.pedDB[:,10] == 1)])
            fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}csv"
            outSurvivors= os.path.join(folderStateNames, fname)
            np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter= ",", fmt= "%d" )            
            case= None
            numSim += 1

    #QFun, VFun, policy  = case.computeAction_Value_Policy()  #computeAction_Value_Policy

    return 

def main():  
    simtime=67 #min
    meandeparture=7 #min
    
    # numSim0= 4390
    # numBlocks= 1
    # simPerBlock= 610

    # name="case_mc_arahama_67_7"
    area="arahama"
    
    # run_mc(area=area,simtime=simtime, meandeparture=meandeparture, 
    #    numSim0=numSim0, numBlocks=numBlocks, simPerBlock=simPerBlock, name=name) 

    simtime=67 #min
    meandeparture=7 #min
    
    numSim0= 4390
    numBlocks= 1
    simPerBlock= 610

    name="case_sarsa_arahama_67_7"

    run_sarsa(area=area,simtime=simtime, meandeparture=meandeparture, 
        numSim0=numSim0, numBlocks=numBlocks, simPerBlock=simPerBlock, name=name) 

    # os.system("osascript -e 'Tell application \"System Events\" to display dialog \"Python run  finished!\"'")
    return 

if __name__ == "__main__":
    main()
