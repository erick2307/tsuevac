# Tsunami Evacuation using Reinforcement Learning

root directory  
--------------
main.py => file to set a case of SARSA simulation  
sarsa.py => SARSA class and functions (needs to be cleaned)  

evac_plots.py => to plot evacuation curves from each epoch  
make_video.py => to create a video (AVI) file of a particular epoch (statematrix or policy)  

mc.py => Monte Carlo method (needs to be revised)  

data directory
--------------
agentsdb.csv => input data of population (currently only 'Node' field is used)  
    > 'Node': Starting node (int)  

linksdb.csv => edges or links of a road network  
    > 'codeLink': the ID of the link/edge (int)  
    > 'node1': starting node of the edge (int)  
    > 'node2': ending node of the edge (int)  
    > 'length': length of the edge in meters (int)  
    > 'width': width of the road in meters (int)  

nodesdb.csv => nodes information (intersections of roads)  
    > number: node ID  
    > coord_x,coord_y: node coordinates  
    > evacuation: flag (0=common node; 1=evacuation point)  
    > reward: abs of the penalty 'reward' given at each node (-1 to account for time pressure in evacuation)  
    
actionsdb.csv;transitionsdb.csv => created with 'pre/SetActionsandTransitions.py'  

figures directory
-----------------
A folder to store snapshots for video (not present in this repository)

other directory
---------------
An informal Python Notebook for various calculations (weights, etc)

pre directory
-------------
    > CensusAndBuildingDatabase is a folder with original census data  
    > Household_database is a folder with original census data (integrated but not in used at the moment)  
    > Population_database is a folder with original census data  
    > defPathsFromNodes.py => a function to calculate the next node for a pre-determined shortest path run.  
    > DetectionShelters.py => to detect evacuation points from the network  
    > lib_importOSM => to transform OSM data into suitable format  

