#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def setMatrices():
    nodesdb = np.loadtxt("./data/nodesdb.csv", delimiter=',', skiprows=1)
    linksdb = np.loadtxt("./data/linksdb.csv", delimiter=',', skiprows=1)
    numNodes = nodesdb.shape[0]
    numLinks = linksdb.shape[0]
    
    actionsdb = np.zeros((numNodes, 20), dtype=np.int64)
    transitionsdb = np.zeros((numNodes, 20), dtype=np.int64)
    probdb = np.zeros((numNodes, 20), dtype=np.int64)
    rewarddb = np.zeros((numNodes, 20), dtype=np.int64)
    
    # print(numNodes) #19257
    
    for i in range(numNodes):
        actionsdb[i,0] = nodesdb[i,0]
        transitionsdb[i,0] = nodesdb[i,0]
        rewarddb[i,0] = nodesdb[i,0]
        
        if nodesdb[i,3]:
            actionsdb[i,1] = 1
            actionsdb[i,2] = -1
            transitionsdb[i,1] = 1
            transitionsdb[i,2] = actionsdb[i,0]
            rewarddb[i,1] = 1
            rewarddb[i,2] = 0
            continue
            
        tmpLinksdb1 = linksdb[linksdb[:,1] == nodesdb[i,0]]
        tmpLinksdb2 = linksdb[linksdb[:,2] == nodesdb[i,0]]
        numlinks1 = tmpLinksdb1.shape[0]
        numlinks2 = tmpLinksdb2.shape[0]
        
        actionsdb[i,1] = numlinks1 + numlinks2
        
        
        transitionsdb[i,1] = numlinks1 + numlinks2
        
        
        rewarddb[i,1] = numlinks1 + numlinks2
        
        # print(i)
        # print(numlinks1,numlinks2)
        if numlinks1:
            actionsdb[i, 2: 2+numlinks1] = tmpLinksdb1[:,0]
            transitionsdb[i,2:2 + numlinks1] = tmpLinksdb1[:,2]
            rewarddb[i, 2:2+numlinks1] = -tmpLinksdb1[:,3]
        
        if numlinks2:
            actionsdb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = tmpLinksdb2[:,0]
            transitionsdb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = tmpLinksdb2[:,1]
            rewarddb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = -tmpLinksdb2[:,3]

    ind = np.argmax(actionsdb[:,1])
#    print(actionsdb[ind])
#    print(transitionsdb[ind])
    
    probdb[:,0:2] = actionsdb[:,0:2]
#    rewarddb[:,0:2] = actionsdb[:,0:2]
    
    for i in range(numNodes):
        numActions = actionsdb[i,1]
        if numActions:
            probdb[i,2:2+numActions] = np.ones(numActions)
    
#    print(probdb[ind])
    np.savetxt('./data/actionsdb.csv', actionsdb, delimiter=',', fmt='%d')
    np.savetxt('./data/transitionsdb.csv', transitionsdb, delimiter=',', fmt='%d')
    # np.savetxt('probdb.csv', probdb, delimiter=',', fmt='%d')
    # np.savetxt('rewarddb.csv', rewarddb, delimiter=',', fmt='%d')
    return


if __name__ == "__main__":
    setMatrices()