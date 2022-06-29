#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
plt.ioff()


class QLearning:
    def __init__(self, agentsProfileName="kochi/data/agentsdb.csv",
                 nodesdbFile="kochi/data/nodesdb.csv",
                 linksdbFile="kochi/data/linksdb.csv",
                 transLinkdbFile="kochi/data/actionsdb.csv",
                 transNodedbFile="kochi/data/transitionsdb.csv",
                 meanRayleigh=7*60,
                 discount=0.9,
                 folderStateNames="state"):
        # setting the rewards for survive or dead
        self.surviveReward = 100000
        self.deadReward = -1000
        self.stepReward = -1
        # store the discount parameter to compute the returns:
        self.discount = discount
        # nodes database in utm coordinates [number, coordX, coordY, evacuactionCode]; 
        # evacuationCode equals 1 if the node is an evacuation node; otherwise, is zero. 
        # On 2020August28 we decided to include a new column that will store the reward in each node
        # That is, a reward will be assigned every time an agent arrives a node:
        # new structure: [number, coordX, coordY, evacuationCode, rewardCode]
        # 2020Oct08: the reward node is replaced by reward every step
        self.nodesdb = np.loadtxt(nodesdbFile, delimiter=',') 
        # 2020Oct07: An additional column must be added to store the link width
        # Thus, this is the new format: [number, node1, node2, length, width]
        self.linksdb = np.loadtxt(linksdbFile, delimiter=',', dtype=np.int) 
        self.populationAtLinks = np.zeros((self.linksdb.shape[0], 2)) # number of agents at links [linkNumber, numberOfAgentsAtLink, density]
        self.populationAtLinks[:,0] = self.linksdb[:,0]
        # Parameters to construct histograms of polations at every link: (1) unit length, (2) number of units
        self.popAtLink_HistParam = np.zeros((self.linksdb.shape[0], 2))
        self.popAtLink_HistParam[:,1] = np.ceil(self.linksdb[:,3] / 2. ) # assuming length units of about 2 meters
        self.popAtLink_HistParam[:,0] = self.linksdb[:,3] / self.popAtLink_HistParam[:,1] 
        # Separating memory for histogram information
        # the number of columns contains the larges number of segments of all the links
        # 2021Jan08 an error of out of bound in axis 1 was 'solved' by adding +1 to these three arrays         
        self.popHistPerLink = np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+1))
        # 2020Oct07: Memory for density array at links+1
        self.denArrPerLink= np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+1))
        # 2020Oct07: Memory for velocity array at links
        self.speArrPerLink= np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+1))
    
        # for p in self.popHistPerLink: print(p)
        self.transLinkdb = np.loadtxt(transLinkdbFile, delimiter=',', dtype=np.int) # database of actions [currentNode, numberOfNodesTarget, linkConnectingNode1, linkConnectingNode2,...]
        self.transNodedb = np.loadtxt(transNodedbFile, delimiter=',', dtype=np.int) # database with possible transitions between nodes [currentNode, numberOfNodesTarget, nodeTarget1, nodeTarget2,...]
        # identifying evacuation nodes
        self.evacuationNodes = self.nodesdb[self.nodesdb[:,3] == 1,0].astype(np.int)
        self.pedProfiles = np.loadtxt(agentsProfileName, delimiter=',', dtype=np.int) # agents profile [age, gender, householdType, householdId, closestNodeNumber]
        self.numPedestrian = self.pedProfiles.shape[0]
        self.errorLoc = 2.0   # acceptable error between coordinate of a node and a coordinate of a pedestrian
        self.snapshotNumber = 0
        
        # setting agent experienced  states
        # each comoponent is a list (initialized as None) of the nodes a 
        # pedestrian pass during a simulation
        # 2020Aug28: We decided to save also the time in which the pedestrian arrives a node
        self.expeStat = [None] * self.numPedestrian
        
        # setting the pedestrian database
        # columns denote:
        # [(0)x0, (1)y0, (2)xtarget,(3)ytarget,(4)vx,(5)vy, (6)currentLink, (7)nodeTgt, (8)lastNode, (9)StartTimeEvacuation]
        # On 2020August28 we included an additional column to save info wheteher a pedestrian arrived an evacuation point
        # new structure:
        # [(0)x0, (1)y0, (2)xtarget,(3)ytarget,(4)vx,(5)vy, (6)currentLink, 
        #  (7)nodeTgt, (8)lastNode, (9)StartTimeEvacuation, (10)IfAlreadyEvacuated]
        self.pedDB = np.zeros((self.numPedestrian,11)) 
        # Assigning initial node
        self.pedDB[:,8] = self.pedProfiles[:,4]  
        # 2020Oct06: Before initiate evacuation, pedestrians do not have link
        # we assign a value of -2, which means, a pedestrian are not in a link
        self.pedDB[:,6] -= 2
        # Pedestrian already located in evacuation node:
        indxAlEv = np.isin(self.pedDB[:,8] , self.evacuationNodes)
        self.pedDB[ indxAlEv , 10] += 1
        # Assigning coordinates
        self.pedDB[:,0:2] = self.nodesdb[ self.pedProfiles[:,4] , 1:3 ] 
        # Assigning randomly the next node target (Book recommend first node target to be random)
        # 2020Oct06: This block has been moved to the function "initEvacuationAtTime"
        # to have the consistency with the variable populationAtLinks
        # lets keep the code lines (commented) for some weeks till confirm the new code runs fine
        #-------init-------
        # for i in range(self.pedDB.shape[0]):
        #     node0 = int(self.pedDB[i,8]) # current node
        #     indxTgt = np.random.choice( int(self.transNodedb[node0,1]) ) # random choise for the next node
        #     nodeTgt = self.transNodedb[node0, 2+indxTgt] # next number node
        #     link = self.transLinkdb[node0, 2+indxTgt] # next link code
        #     self.pedDB[i,7] = nodeTgt
        #     self.pedDB[i,6] = link
        #     self.pedDB[i, 2:4] = self.nodesdb[nodeTgt, 1:3] # coordinates of the next target (next node)
        #     # 2020Aug28: [1 slot for the state code, 1 slot for the agent choice, 1 slot para el tiempo de arrivo]
        #     # 2020Aug28: We reserve three slots now
        #     firstState = np.zeros(3, dtype = np.int)
        #     # we only update the chosen link (first action)
        #     # firstState[0] = None
        #     firstState[1] = int(indxTgt)
        #     self.expeStat[i] = [firstState] 
        #-------end-------
    
        # setting initial evacuation time for each pedestrian
        scaleRayleigh = meanRayleigh * (2/np.pi)**0.5
        self.pedDB[:,9] = np.round( np.random.rayleigh(scale = scaleRayleigh, size = self.pedDB.shape[0]) , decimals = 0 ).astype(np.int)
        # Updating initial time for simulation. That is, simulation starts with
        # lowest evacuation time of an arbitrary pedestrian
        self.time = min(self.pedDB[:,9])
        # Setting the state and action-value functions. Its components are:
        # [nodeInit, 10 slots for links density code, 10 slots for action-value functions, \n
        # 10 slots for counting the observations of this states]
        self.stateMat = np.zeros(( self.nodesdb.shape[0] , 31 ))
        # As initial observed states, we report the nodes with empty links:
        # Note we assign 0.5 as action-value (it will be updated during the learning process)
        self.stateMat[ : , 0 ] = self.nodesdb[ : , 0 ]
        for i in range(self.nodesdb.shape[0]):
            self.stateMat[ i , 11:11+int(self.transLinkdb[i,1]) ] = 0.5*np.ones(self.transLinkdb[i,1])
            self.stateMat[ i , 21:21+int(self.transLinkdb[i,1]) ] = np.ones(self.transLinkdb[i,1])
        
        # setting database with the shortestpath, geometrically speaking, database
        # Included in order to assess RL approach with the simplest shortest path
        self.shortestPathDB = None
    
    ########## Set environment ##########
    def setTime(self, t):
        """
        function for the user to modify the time-step of a simulation
        """
        self.time = t
        return
    
    def computePedHistDenVelAtLinks(self):
        """
        2020Oct05:
        Function created to compute the histogram, density, and velocity at every link:
        """
        emptyLinksIndx= np.where( self.populationAtLinks[:,1] == 0 )[0]
        for ind in emptyLinksIndx:
            self.popHistPerLink[ind,:] = np.zeros( self.popHistPerLink.shape[1] )
            self.denArrPerLink[ind,:] = np.zeros( self.popHistPerLink.shape[1] )
            self.speArrPerLink[ind,:] = 1.19*np.ones( self.popHistPerLink.shape[1] )
        occupLinksIndx= np.where( self.populationAtLinks[:,1] > 0 )[0]
        for ind in occupLinksIndx:
            unitL=  self.popAtLink_HistParam[ind,0] 
            numComp= int( self.popAtLink_HistParam[ind,1] )
            lengthL= self.linksdb[ind, 3]
            width= self.linksdb[ind, 4]
            n0L= self.linksdb[ind, 1]
            x0L, y0L= self.nodesdb[n0L,1] , self.nodesdb[n0L,2]
            dbPed_tmp= self.pedDB[ self.pedDB[:,6] == ind , :]
            dist= ( (dbPed_tmp[:,0] - x0L)**2 + (dbPed_tmp[:,1] - y0L)**2 )**0.5
            dist = np.clip(dist, 0, lengthL)
            
            hist, bin_edges= np.histogram(dist, bins= numComp, range= (0, lengthL) )

            self.popHistPerLink[ind, :numComp ] = hist 
            self.denArrPerLink[ind, :numComp] = hist / (unitL * width)
            self.speArrPerLink[ind, :numComp] = 1.388 - 0.396 * self.denArrPerLink[ind, :numComp]
            self.speArrPerLink[ind, :numComp] = np.clip( self.speArrPerLink[ind, :numComp] , 0.2 , 1.19 )
        return
    
    def computeWeightsAtLinks(self):
        filename="w_%09d.csv" % self.time
        fout=os.path.join("weights",filename)
        np.savetxt(fout,self.populationAtLinks,delimiter=",",fmt="%d")
        return
        
    def getPedHistAtLink(self, codeLink):
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.popHistPerLink[codeLink, :numComp] 
    
    def getDenArrAtLink(self, codeLink):
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.denArrPerLink[codeLink, :numComp]
    
    def getVelArrAtLink(self, codeLink):
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.speArrPerLink[codeLink, :numComp]
    
    def computeDensityLevel(self, codeLink, linkWidth = 2.):
        """
        Computes the pedestrian-density level at specific link. 
        The current function has only three levels of pedestrian-density
        """
        density = float(self.populationAtLinks[codeLink,1]) /(linkWidth * self.linksdb[codeLink,3])
        
        if density <= 0.5:
            denLvl = 0
        elif density <= 3.0:
            denLvl = 1
        else:
            denLvl = 2
        return denLvl
    
    def getStateIndexAtNode(self, codeNode):
        """
        It identify the location of current state at node "codeNode" in the matrix "stateMat"
        """
        #reading state at codeNode
        state_arr = np.zeros(31)
        state_arr[0] = codeNode
        linksdb = self.transLinkdb[codeNode,:]
        for l in range(2,2+linksdb[1]):
            state_arr[l-1] = self.computeDensityLevel(linksdb[l])
        #checking current state at "stateMat" variable
        indx = np.where(self.stateMat[:,0] == codeNode)
        # If the state at node "codeNode" already exist, it will return the index:
        for i in indx[0]:
            if np.array_equal(state_arr[:11],self.stateMat[i,:11]):
                return i
        # If the state at node "codeNode" does not exist, it will create a new state at botttom
        state_arr[11:11+linksdb[1]] = 0.5*np.ones(linksdb[1])
        state_arr[21:21+linksdb[1]] = np.ones(linksdb[1])
        self.stateMat = np.concatenate((self.stateMat, state_arr.reshape(1,31)), axis=0)
        return self.stateMat.shape[0] - 1
    
    def exportStateMatrix(self, outnamefile="stateMatrix.csv"):
        """
        Export the matrix "stateMat" to the file "outnamefile" in csv format
        """
        np.savetxt(outnamefile, self.stateMat, delimiter=',', fmt=["%d"  ,"%d"  ,  "%d",  "%d",  "%d",  "%d",  "%d",  "%d",  "%d",  "%d","%d",
                                                                   "%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f",
                                                                   "%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"  ,"%d"])
        return
    
    def exportAgentDBatTimet(self,outnamefile):
        """
        Exports the matrix "pedDb" in the file "outnamefile" in csv format

        """
        # Recall the meaning of the components:
        # cols: [(0)x0, (1)y0, (2)xtarget,(3)ytarget,(4)vx,(5)vy,
        # (6)currentLink, (7)nodeTgt, (8)lastNode, (9)StartTimeEvacuation, (10)IfAlreadyEvacuated]        
        np.savetxt(outnamefile, self.pedDB, delimiter=',', fmt=["%.6f","%.6f","%.6f","%.6f","%.3f","%.3f","%d","%d","%d","%d","%d"])
        return
        
    def loadStateMatrixFromFile(self, namefile):
        """
        Updates the matrix "stateMat" from a file. Useful to the learning process,
        in which we want the stateMat from previous simulations
        """
        self.stateMat = np.loadtxt(namefile, delimiter=",")
        return
    
    def getPopulationAtLinks(self):
        """
        return the matrix "populationAtLinks", the number of pedestrian at links.
        """
        return self.populationAtLinks
    
    def resizePedestrianDB(self, size):
        indx= np.random.choice( self.pedDB.shape[0] , size= size )
        self.pedDB = self.pedDB[indx, :]
        return
    
    #new function to create diff size of population
    def resizePedDB(self, size):
        cs = self.pedDB.shape[0]
        print(cs)
        if cs < size:
            fillnum = size - cs
            newindx = np.linspace(0,size-1,size)
            indx = np.random.choice(self.pedDB.shape[0],size=fillnum)
            self.pedDB = np.concatenate((self.pedDB,self.pedDB[indx,:]), axis=0)
            self.pedDB[:,0]=newindx
        else:
            fillnum = cs - size + 1
            newindx = np.linspace(0,size-1,size)
            indx= np.random.choice(self.pedDB.shape[0],size=fillnum)
            self.pedDB = self.pedDB[indx, :]
            self.pedDB[:,0]=newindx
        return   
    
    ######### Move a unit step in the simulation #########
    def stepForward(self, dt=1):
        """
        Moves the simulation one step forward. That is, 
        the pedestrian moves according to their velocity for a period of time "dt".
        it also updates the variable "time".
        """
        self.pedDB[ : , 0:2 ] += dt * self.pedDB[ : , 4:6 ]
        self.time += dt
        return
    
    def checkTarget(self, ifOptChoice = False):
        """
        Verifies if the pedestrians arrived to their node target.
        For those who already arrived, the function assign the next target.
        """
        # Computes the distance of the pedestrian coordinates to the node trget coordinates:
        error = ( (self.pedDB[ : , 0 ] - self.pedDB[ : , 2 ])**2 + (self.pedDB[ :  , 1 ] - self.pedDB[ : , 3 ])**2 )**0.5
        # Identifies the pedestrians that are closer than the threshold "errorLoc"
        # Then the new target is assigned
        indx = np.where(error <= self.errorLoc)[0]
        for i in indx:
            # 2020Aug28: we use the new column in pedDB:
            if self.pedDB[i,10]:
                continue
            self.updateTarget(i, ifOptChoice = ifOptChoice)
        return
    
    #def updateSpeed(self, codeLink, linkWidth = 2.):
        #"""
        #Computes the speed at link "codeLink" considering its actual pedestrian-density.
        #2021Jan06(E) we need to consider vehicles instead of pedestrians.
        #A new function was copied with different speeds.
        #"""
        ## Computes the density at link
        #density = self.populationAtLinks[codeLink,1] /(linkWidth * self.linksdb[codeLink,3])
        ## Assign a speed according the the density
        ## Check paper reference, where these values come from
        #if density <= 0.5:
            #return 12.5 + np.random.randn()*0.1
        #elif density <= 3.0:
##            print("moderate density of %.4f at %d" % (density,codeLink))
            #return 5.56 + np.random.randn()*0.1
        #else:
##            print("high density of %.4f at %d" % (density,codeLink))
            #return 1.39 + np.random.randn()*0.01 
        
    #### COMMENTED ON 2021 JAN 06 ######    
    def updateSpeed(self, codeLink, linkWidth = 2.):
        """
        Computes the speed at link "codeLink" considering its actual pedestrian-density.
        2021Jan06(E) we need to consider vehicles instead of pedestrians.
        A new function was copied with different speeds.
        """
        # Computes the density at link
        density = self.populationAtLinks[codeLink,1] /(linkWidth * self.linksdb[codeLink,3])
        # Assign a speed according the the density
        # Check paper reference, where these values come from
        if density <= 0.5:
            return 1.19 + np.random.randn()*0.1
        elif density <= 3.0:
#            print("moderate density of %.4f at %d" % (density,codeLink))
            return 0.695 + np.random.randn()*0.1
        else:
#            print("high density of %.4f at %d" % (density,codeLink))
            return 0.20 + np.random.randn()*0.01     
    ######################################
    
    def updateVelocityV1(self, pedIndx, linkWidth = 2.):
        """
        Updates the velocity of a pedestrian "pedIndx" according to the link the pedestrian is located.
        """
        codeLink= int( self.pedDB[pedIndx, 6] )

        # Compute speed:
        speed = self.updateSpeed(codeLink = codeLink, linkWidth = linkWidth)
        # Unit vector pointing to the node target
        unitDir = (self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2]) / np.linalg.norm(self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2])
        # Compute and assign new velocity:
        vel_arr = speed * unitDir
        self.pedDB[pedIndx, 4:6] = vel_arr
        return
    
    def updateVelocityV2(self, pedIndx):
        """
        Update the velocity of a pedestrian "pedIndx" according to a histogram of density in the link.
        """
        codeLink= int( self.pedDB[pedIndx, 6] )
        
        if codeLink == -1:
            return
        else:
            n0L= self.linksdb[codeLink, 1]
            x0L, y0L= self.nodesdb[n0L,1] , self.nodesdb[n0L,2]
            dist= ( (self.pedDB[pedIndx,0] - x0L)**2 + (self.pedDB[pedIndx,1] - y0L)**2 )**0.5
            unitL= self.popAtLink_HistParam[codeLink,0]
            xAtLink= int( np.floor( dist / unitL) ) 
            speed= self.speArrPerLink[codeLink, xAtLink] + np.random.rand()*0.02 - 0.01  
            unitDir = (self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2]) / np.linalg.norm(self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2])
            vel_arr = speed * unitDir
            self.pedDB[pedIndx, 4:6] = vel_arr
        return
    
    def updateVelocityAllPedestrians(self):
        indxAtEvaPoi= self.pedDB[:,10] == 0
        indxIniEva= self.time > self.pedDB[:,9] 
        indxBoth = np.where( np.logical_and(indxAtEvaPoi, indxIniEva) )[0]
        
        for ind in indxBoth:
            self.updateVelocityV2(ind)
        return
            
            
    
    def initEvacuationAtTime(self):
        """
        this function updates the velocity of the agents according to their initial evacuation time.
        That is, the function will identify the agents whose initial evacuation time equals the current parameter "self.time".
        Then, the velocity of these agents are updated (because they have velocity zero at first).
        furthermore, here the first experienced state is recorded.
        """
        # Find pedestrian initiating evacuation
        indxPed = np.where(self.pedDB[:,9] == self.time)[0]
        # Check if there are pedestrians starting evacuation
        if len(indxPed) != 0:
            for i in indxPed:
                node0 = int(self.pedDB[i,8]) # current node
                indxTgt = np.random.choice( int(self.transNodedb[node0,1]) ) # random choice for the next node
                nodeTgt = self.transNodedb[node0, 2+indxTgt] # next number node
                link = self.transLinkdb[node0, 2+indxTgt] # next link code
                self.pedDB[i,7] = nodeTgt
                self.pedDB[i,6] = link
                self.pedDB[i, 2:4] = self.nodesdb[nodeTgt, 1:3] # coordinates of the next target (next node)
                # 2020Aug28: [1 slot for the state code, 1 slot for the agent choice, 1 slot for the arrival time]
                # 2020Aug28: We reserve three slots now
                firstState = np.zeros(3, dtype = np.int)
                firstState[1] = int(indxTgt)
                #-----end-----
                # Update velocity
                self.updateVelocityV2(i)  
                #previous: self.updateVelocity(i, int(self.pedDB[i,6]))
                
                # Get state code at starting node
                indxStat = self.getStateIndexAtNode(int(self.pedDB[i,8]))
                
                firstState[0]= int(indxStat)
                firstState[2]= int(self.time)
                self.expeStat[i] = [firstState] 
                # Save the first state code experienced by the pedestrian "i" at the list "expeStat".
                # Note we only save the state code, the action (target node) chosen was assigned in the initiation:
                # self.expeStat[i][0][0] = int(indxStat)
                # Save also the starting time
                # self.expeStat[i][0][2] = int(self.time)
                # Report a new pedestrian enter link
                # But first we check if the pedestrian arrived an evacuation-node
                if int(self.pedDB[i,6]) >= 0:
                    self.populationAtLinks[int(self.pedDB[i,6]), 1] += 1
        return
    
    def updateTarget(self, pedIndx, ifOptChoice = False):
        """
        Updates the node target of pedestrian pedIndx.
        It considers whether the pedestrian uses optimal (exploting) 
        or random (exploring) approach. 
        """
        # 2020Aug28: using new column of pedDBid ped is in evacuation node:
        if self.pedDB[pedIndx,10]:
            return
        # Otherwise, we assign new target node:
        else:
            # New start node is now previous target node:
            node0 = int(self.pedDB[pedIndx , 7])
            x0_arr = np.array([ self.nodesdb[node0,1], self.nodesdb[node0,2] ])
            ######### Here, please update later the new approach to choose a link!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Get state at new start node 
            stateIndx = self.getStateIndexAtNode(node0)
            # Get all possible action-values according to the state:
            Qval_arr = self.stateMat[stateIndx , 11:11+int(self.transLinkdb[node0,1])]
            # If optimal choice, it selects the action with the largest action-value:
            if ifOptChoice:
                # In case there are more than one action with the largest action-value,
                # one is chosen randomly
                indxTgt = np.random.choice( np.where( Qval_arr == max(Qval_arr) )[0] )
            # If not optimal choice, then we select randomly, 
            # but using a distribution based on the current action-values
            else:
                indxTgt = np.random.choice(int(self.transNodedb[node0,1]))
            # Get chosen link and new target node:
            link = self.transLinkdb[node0, 2+indxTgt]
            nodeTgt = self.transNodedb[node0, 2+indxTgt]
            # If new start node equals the new target node,
            # it means pedestrian arrived evacuation node. Therefore, we change 
            # velocity to zero, remove pedestrian from its last link
            # (2020Aug28) and report in matrix "pedDB" that the pedestrian arrived
            # and evacuation
            if nodeTgt == node0:
                xTgt_arr = x0_arr
                vel_arr = np.array([0,0])
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.pedDB[pedIndx,10] = 1
            # If new target node is not an evacuation node, then velocity is
            # updated:
            else:
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.populationAtLinks[link, 1] += 1
                xTgt_arr = np.array([ self.nodesdb[nodeTgt,1], self.nodesdb[nodeTgt,2] ])
                unitDir = (xTgt_arr - x0_arr) / np.linalg.norm(xTgt_arr - x0_arr)
                # 2020Oct07: we modified the speed
                speed= self.speArrPerLink[link,0]
                # speed = self.updateSpeed(link)
                vel_arr = speed * unitDir
            # Save experienced state and action taken in an array:
            # 2020Aug28: We now save also the time
            expeStatAndVal = np.array([stateIndx, indxTgt, self.time], dtype=np.int)
            # Update matrix "pedDB":
            self.pedDB[pedIndx , :9] = np.array( [x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0] )
            # Record state and action experienced by the pedestrian:
            self.expeStat[pedIndx].append(expeStatAndVal) 
            # delete this
            
            # 2020Oct08: Apply here TDControl
            self.tdControl(pedIndx)
        return
    
    def tdControl(self, pedIndx, alpha= 0.05):
        """
        This funciton represents the main change between SARSA and MonteCarlo.
        Here we update the variable "stateMat" during the episode, rather than at the end.

        """
        trackPed = np.array(self.expeStat[pedIndx])
        # current and previous states
        current_S= trackPed[-1,0]
        pre_S= trackPed[-2,0]
        # current and previous actions
        current_A= trackPed[-1,1]
        pre_A= trackPed[-2,1]
        # current and pre time
        current_t = trackPed[-1,2]
        pre_t = trackPed[-2,2]
        
        if self.pedDB[pedIndx,10]:
            currentReward= self.surviveReward
            self.stateMat[current_S, 11 + current_A] += alpha * (currentReward - self.stateMat[current_S, 11 + current_A])
            self.stateMat[current_S, 21 + current_A] += 1
        
        preReward= self.stepReward * (current_t - pre_t)
        self.stateMat[pre_S, 11 + pre_A] += alpha * (preReward + self.discount * self.stateMat[current_S, 11 + current_A] - self.stateMat[pre_S, 11 + pre_A])
        self.stateMat[pre_S, 21 + pre_A] += 1
        return
    
    ########## functions to use shortest path
    
    def loadShortestPathDB(self, namefile):
        self.shortestPathDB = np.loadtxt(namefile, delimiter=",", skiprows=1, dtype=np.int)
        print(self.shortestPathDB.size)
        return
    
    def updateTargetShortestPath(self, pedIndx):
        if self.pedDB[pedIndx, 6] == -1:
            return
        else:
            node0 = int(self.pedDB[pedIndx, 7])
            x0_arr = np.array([self.nodesdb[node0, 1], self.nodesdb[node0, 2]])
            nodeTgt = self.shortestPathDB[node0, 1]
            numNodesLinked = self.transNodedb[node0, 1]
            nodesLinked = self.transNodedb[node0, 2 : 2 + numNodesLinked]
            indxTgt = np.where(nodesLinked == nodeTgt)[0][0]
            link = self.transLinkdb[node0, 2 + indxTgt]
            if nodeTgt == node0:
                xTgt_arr = x0_arr
                vel_arr = np.array([0, 0])
                self.populationAtLinks[int(self.pedDB[pedIndx, 6]), 1] -= 1
            else:
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.populationAtLinks[link, 1] += 1
                xTgt_arr = np.array([ self.nodesdb[nodeTgt, 1], self.nodesdb[nodeTgt, 2]])
                unitDir = (xTgt_arr - x0_arr) / np.linalg.norm(xTgt_arr - x0_arr)
                speed = self.updateSpeed(link)
                vel_arr = speed * unitDir
            self.pedDB[pedIndx, :9] = np.array([x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0])
        return
    
    def checkTargetShortestPath(self):
        error = ((self.pedDB[:, 0] - self.pedDB[:, 2]) ** 2 + (self.pedDB[:, 1] - self.pedDB[:, 3]) ** 2) ** 0.5
        indx = np.where(error <= self.errorLoc)[0]
        for i in indx:
            self.updateTargetShortestPath(i)
        return
    
    ########## Update state matrix (stateMat) on a MC fashion ##########
    def updateValuefunctionByAgent(self, pedIndx):
        """
        The matrix "stateMat" is updated for each pedestrian
        """
        if self.pedDB[pedIndx,8] in self.evacuationNodes:
            reward = self.surviveReward #1.
        else:
            reward = self.deadReward #0.
        expSta = np.array(self.expeStat[pedIndx])
        #updating first experience
        self.stateMat[int(expSta[0,0]), int(11 + expSta[0,1])] += (reward - self.stateMat[int(expSta[0,0]), int(11 + expSta[0,1])])/(self.stateMat[int(expSta[0,0]), 21 + int(expSta[0,1])] + 1)
        self.stateMat[int(expSta[0,0]), 21 + int(expSta[0,1])] += 1
        if len(expSta) > 1:
            for i in range(1,expSta.shape[0]):
                if expSta[i,0] in expSta[:i,0]:
                    continue
                self.stateMat[int(expSta[i,0]), 11 + int(expSta[i,1])] += (reward - self.stateMat[int(expSta[i,0]), 11 + int(expSta[i,1])])/(self.stateMat[int(expSta[i,0]), 21 + int(expSta[i,1])] + 1)
                self.stateMat[int(expSta[i,0]), 21 + int(expSta[i,1])] += 1    
        return 
    
    def updateValuefunctionByAgentV2(self, pedIndx, ifConstStepSize= True, alpha= 0.05):
        """
        2020Aug28:
        Updated version to compute the returns.
        This version considers discount and better consideration of the time 
        between consecutive nodes
        
        2020Oct06:
        We included the option to use a constant step size (alpha):
        V = V + alpha (G - V)
        Previous version only considered average updating:
        V = V + (G - V)/(N+1) ; where N is the number of times a particular state has occurred
        """
        expSta = np.array(self.expeStat[pedIndx])
        state = expSta[-1,0]
        choice = expSta[-1,1]
        node = int(self.stateMat[state,0])
        
        G = 0
        
        if node in self.evacuationNodes:
            R = self.surviveReward
        else:
            R = self.deadReward
        
        G = R
        
        if ifConstStepSize:
            self.stateMat[state, 11 + choice] += alpha*(G - self.stateMat[state, 11 + choice])
            self.stateMat[state, 21 + choice] += 1
        else:
            self.stateMat[state, 11 + choice] += (G - self.stateMat[state, 11 + choice])/(self.stateMat[state, 21 + choice] + 1)
            self.stateMat[state, 21 + choice] += 1
        
        if len(expSta) > 1:
            
            for i in range(expSta.shape[0]-2, -1, -1):
                #2020Oct08: new reward scheme
                # if pedIndx == 0: print(G) 
                state = expSta[i,0]
                choice= expSta[i,1]
                t1 = expSta[i   , 2]
                t2 = expSta[i+1 , 2]
                R = self.stepReward * (t2 - t1)
                G *= self.discount**(t2-t1)
                G += R
                
                if ifConstStepSize:
                    self.stateMat[state, 11 + choice] += alpha*(G - self.stateMat[state, 11 + choice])
                    self.stateMat[state, 21 + choice] += 1
                else:
                    # The following line code comes from the following expression
                    # G_Av(N+1) = G_Av(N) + ( G - G_Av(N) )/ (N+1)
                    self.stateMat[state, 11 + choice] += (G - self.stateMat[state, 11 + choice])/(self.stateMat[state, 21 + choice] + 1)
                    # Add one more visit to the state
                    self.stateMat[state, 21 + choice] += 1
                

                
                # # print("type(node), node")
                # # print(type(node), node)
                # R = self.nodesdb[node, 4]
                # t1 = expSta[i   , 2]
                # t2 = expSta[i+1 , 2]
                # G *= self.discount**(t2-t1)
                # G += R
                
                # ###### delete this block
                # # if node == 149:
                # #     print("node 149")
                # #     print(R, t1, t2, G, self.stateMat[state, 11 + choice])
                # #     if t1 > t2:
                # #         print(pedIndx)
                # #         print(expSta)
                # #####
                # state = expSta[i,0]
                # choice = expSta[i,1]
                # node = int(self.stateMat[state,0])
                
                # if expSta[i,0] in expSta[:i,0]:
                #     continue
                # if ifConstStepSize:
                #     self.stateMat[state, 11 + choice] += alpha*(G - self.stateMat[state, 11 + choice])
                #     self.stateMat[state, 21 + choice] += 1
                # else:
                #     # The following line code comes from the following expression
                #     # G_Av(N+1) = G_Av(N) + ( G - G_Av(N) )/ (N+1)
                #     self.stateMat[state, 11 + choice] += (G - self.stateMat[state, 11 + choice])/(self.stateMat[state, 21 + choice] + 1)
                #     # Add one more visit to the state
                #     self.stateMat[state, 21 + choice] += 1
        return
    
    def updateValueFunctionDB(self):
        """
        The matrix "stateMat" is updated for all pedestrians

        """
        # Filter out pedestrian that never started evacuation
        indx_arr =np.where(self.pedDB[:,9] < self.time)[0]
        for indx in indx_arr:
            # 2020Aug30: Here we replace the option "updateValuefunctionByAgent" 
            # by the new function "updateValuefunctionByAgentV2"
            # self.updateValuefunctionByAgent(indx)
            self.updateValuefunctionByAgentV2(indx)
        return 
    
    def computeAction_Value_Policy(self):
        """
        2020Aug28: Function to export the Action-value function (QFun),
        Value-function (VFun), and policy
        """
        # Get action values
        QFun = np.zeros(( self.stateMat.shape[0] , 10))
        QFun = self.stateMat[:,11:21]
        # Get policy
        nodeState = self.stateMat[ : , 0 ].astype(np.int)
        numActions = self.transLinkdb[ nodeState , 1]
        policy = np.zeros( self.stateMat.shape[0] )
        for i in range( self.stateMat.shape[0] ):
            if not numActions[i]:
                continue 
            policy[i] = np.argmax( QFun[i, : numActions[i] ] )
        # Get value function
        VFun = np.sum( self.stateMat[ : , 11:21 ] * self.stateMat[ : , 21:31 ] , axis = 1) / np.sum(self.stateMat[ : , 21:31 ] , axis=1) 
        return  QFun, VFun, policy 
 
    ########## Visualization functions ##########
    def setFigureCanvas(self):
        self.fig, self.ax = plt.subplots(figsize=(12,5))
        for i in range(self.linksdb.shape[0]):
            self.ax.plot([self.nodesdb[int(self.linksdb[i,1]),1], self.nodesdb[int(self.linksdb[i,2]),1]],[self.nodesdb[int(self.linksdb[i,1]),2], self.nodesdb[int(self.linksdb[i,2]),2]], c='k', lw=1)
        #indxZeroActions = np.where(self.transLinkdb[:,1] != 0)[0]
        indxEv = self.nodesdb[:,3] == 1.0
        #self.p1, = self.ax.plot(self.nodesdb[indxZeroActions][:,1], self.nodesdb[indxZeroActions][:,2], 'bo', ms=0.5)
        self.p1, = self.ax.plot(self.nodesdb[indxEv,1], self.nodesdb[indxEv,2], 'rD', ms=4, mfc="none", mec="r")
        indx = np.where(self.pedDB[:,9] <= self.time)[0]
        # 2020Oct07: trying to place color velocity
        # self.p2, = self.ax.plot(self.pedDB[indx,0], self.pedDB[indx,1], 'ro', ms=1)
        speed= ( (self.pedDB[indx,4])**2 + (self.pedDB[indx,5])**2)**0.5
        self.p2 = self.ax.scatter(self.pedDB[indx,0], self.pedDB[indx,1], 
                                  c= speed, s=10, vmin=0.1, vmax=1.3, 
                                  cmap="jet_r", edgecolors='none')
        # indxN= self.nodesdb[:,3] == 1
        
        self.fig.colorbar(self.p2)
        self.ax.axis("equal")
        self.ax.set_axis_off()
        self.lblCoord = np.array([max(self.nodesdb[:,1]), max(self.nodesdb[:,2])])
        self.labelTime = self.fig.text( 0, 0, " ")
        return
    
    def getSnapshotV2(self):
        self.p2.remove()
        indx = np.where(self.pedDB[:,9] <= self.time)[0]
        # 2020Oct07: trying to place color velocity
        # self.p2, = self.ax.plot(self.pedDB[indx,0], self.pedDB[indx,1], 'ro', ms=1)
        speed= ( (self.pedDB[indx,4])**2 + (self.pedDB[indx,5])**2)**0.5
        self.p2 = self.ax.scatter(self.pedDB[indx,0], self.pedDB[indx,1], 
                                  c= speed, s=10, vmin=0.1, vmax=1.3, 
                                  cmap="jet_r", edgecolors='none') 
        # self.fig.colorbar(self.p2)
        self.labelTime.remove()
        self.labelTime = self.fig.text( 0, 0, "t = %.2f min; evacuated: %d of %d" % (self.time/60., np.sum(self.pedDB[:,10] == 1), self.pedDB.shape[0]))
        self.fig.savefig(os.path.join("figures", "Figure_%04d.png" % self.snapshotNumber), 
                         bbox_inches="tight", dpi=150)
        
        self.snapshotNumber += 1
        return
    
    def makeVideo(self, nameVideo = "Simul.avi"):
        listImagesUS = glob.glob( os.path.join("figures", "*png"))
        numSS_ar= np.zeros( len(listImagesUS) , dtype= np.int)
        for i, li in enumerate(listImagesUS):
            numSS_ar[i]= int( li[-8:-4] ) 
            # print(code)
        indSort= np.argsort(numSS_ar)
        listImages= []
        for i in indSort:
            listImages.append( listImagesUS[i] )
            
        img = cv2.imread(listImages[0])
        height_im, width_im, layers_im = img.shape
        video = cv2.VideoWriter(nameVideo, cv2.VideoWriter_fourcc('M','J','P','G'),15,(width_im, height_im))  # Only works with openCV3
        for im in listImages:
            print(im)
            img = cv2.imread(im)
            video.write(img)
        cv2.destroyAllWindows()
        video.release()
        return
    
    def destroyCanvas(self):
        self.p1 = None
        self.p2 = None
        self.ax = None
        self.fig = None
        return
    
    def deleteFigures(self):
        figures = glob.glob( os.path.join("figures","*") ) 
        
        for f in figures:
            os.remove(f)
    
    def plotNetwork(self):
        colorNode = ["b","r"] 
        plt.figure(num="Network",figsize=(5,4))
        for i in range(self.linksdb.shape[0]):
            plt.plot([self.nodesdb[int(self.linksdb[i,1]),1], self.nodesdb[int(self.linksdb[i,2]),1]],[self.nodesdb[int(self.linksdb[i,1]),2], self.nodesdb[int(self.linksdb[i,2]),2]], c='k', lw=1)
            plt.text(0.5*(self.nodesdb[int(self.linksdb[i,1]),1] + self.nodesdb[int(self.linksdb[i,2]),1]) , 0.5*(self.nodesdb[int(self.linksdb[i,1]),2] + self.nodesdb[int(self.linksdb[i,2]),2]), 
                    "%d" % self.linksdb[i,0], fontsize=8, color='g')
        # for i in range(self.nodesdb.shape[0]):
        #     plt.text(self.nodesdb[i,1], self.nodesdb[i,2], self.nodesdb[i,0], fontsize = 7, color=colorNode[ int(self.nodesdb[i,3]) ])
        indxZeroActions = np.where(self.transLinkdb[:,1] != 0)[0]
        plt.scatter(self.nodesdb[indxZeroActions][:,1], self.nodesdb[indxZeroActions][:,2], s=20, edgecolor='k', linewidths=0.0)
        plt.scatter(self.pedDB[:self.numPedestrian,0], self.pedDB[:self.numPedestrian,1], s=40, c = "r", linewidth=0.0)
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.show()