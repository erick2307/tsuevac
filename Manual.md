# Manual to use the RL-TsuEvac model

## Overview of the model

The model consists on training nodes or intersections from a road network (`nodesdb.csv` and `linksdb.csv`) extracted from [OSM](https://www.openstreetmap.org/) to be able to optimally guide the evacuation of a group of agents or population (`agentsdb.csv`).

A database of actions (`actionsdb.csv`) is constructed  out of the possible directions to be taken by an evacuee at an intersection or node in the network. Thus, the matrix is as follows:

$A=[N,M_N,l_i...l_m]$

where, $A$ is the action matrix that contains a set $N$ of $n$ nodes linked to $m$ other nodes through a set of links $l_i ... l_m$. This matrix maps the transition from one node through a particular link. $M_N$ is the set of $m$ nodes set of each node $n in the entire set of nodes $N$.

Similarly, a `transitionsdb.csv` matrix is made to map the possible transitions from one node to another. This is of the form:

$T=[N,M_N,n_i ... n_m]$

where, $T$ is the transition matrix between nodes, and $n_i ... n_m$ are the nodes available from a particular node $n$ in the set of nodes $N$.

## `main_{#method}.py`

The code imports from `mc.py` the `MonteCarlo` class and from `sarsa.py` the `SARSA` class. In addition, `qlearn.py` is the `Q-Learning` version.

The files required as input are:   
* `agentsdb.csv`  
* `nodesdb.csv`  
* `linksdb.csv`  
* `actionsdb.csv`  
* `transitionsdb.csv`  

***These files should be located in a `data` folder within a dedicated folder for the study case***

The parameters needed are:  
* `meanRayleigh` .- This is the mean value of a Rayleigh distribution for the evacuation departure time decision.  
* `folderstateNames` .- A name of a folder to store states explored during the learning process.
