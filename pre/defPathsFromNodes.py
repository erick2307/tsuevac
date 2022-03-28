#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function to create short paths and nodedb
"""

import networkx as nx
import pandas as pd
import numpy as np
import time

t0 = time.time()

# create a graph
G = nx.Graph()

# read data
Nodesdf = pd.read_csv('nodesdb_utm_1.csv',names=["id","x","y","type"],skiprows=1)
Edgesdf = pd.read_csv('linksdb.csv',names=["id","n1","n2","length"],skiprows=1)
Sheltersdf = Nodesdf[Nodesdf["type"] == 1]

# add nodes to graph
for i,row in Nodesdf.iterrows():
    G.add_node(row[0],pos=(row[1],row[2]),ntype=row[3])

#add edges to graph
for i,row in Edgesdf.iterrows():
    G.add_edge(row[1],row[2],length=row[3])
    
numberOfNodes = G.number_of_nodes()
numberOfEdges = G.number_of_edges()
numberOfShelters = Sheltersdf.shape[0]

#calculate shortest paths lengths to each shelter and add to DF
for i,shrow in Sheltersdf.iterrows():
    lsh = []
    for j,nrow in Nodesdf.iterrows():
        try:
            lsh.append(nx.shortest_path_length(G,source=nrow[0],target=shrow[0],weight='length'))
        except:# nx.NetworkXNoPath:
            lsh.append(1.0e+10)
    Nodesdf[str(i)]=lsh
    lsh = []
    
#calculate from each node its closest shelter
ShelterColumnsSeries = []
for i, nrow in Nodesdf.iterrows():
    ShelterColumnsSeries.append(nrow[4:].idxmin())
Nodesdf['shelter']= ShelterColumnsSeries
Nodesdf.head()

#calculate shortest path from each node to corresponding shelter
ln = []
for i, nrow in Nodesdf.iterrows():
    try:
        if int(nrow[0]) != int(nrow.shelter):
            ln.append(nx.shortest_path(G,int(nrow[0]),int(nrow.shelter))[1])
        else:
            ln.append(int(nrow[0]))
    except:# nx.NetworkXNoPath:
        ln.append(int(-9999))
Nodesdf['nextnode']=ln

data = Nodesdf[['id','nextnode']]
data.to_csv('nextnode_1shelter.csv',index=False)

t1 = time.time()
total = t1-t0
print(total)