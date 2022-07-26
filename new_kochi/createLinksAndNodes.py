# -*- coding: utf-8 -*-
from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt

def readEdges(iShpPath):
    # driver = ogr.GetDriverByName("GPKG")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(iShpPath, 0)
    layer = dataSource.GetLayer()
    featureCount = layer.GetFeatureCount()
    
    mNodes= np.zeros((2*featureCount,3))   # [nodeCode, x, y]
    countMN= 0
    countN= 0   #count nodes
    countE= 0   #count edges
    listEd= []
    listNo= []
    MaxDistSquared= 16 # means the maximum distance is 4
    
    for feature in layer:
        fromID= feature.GetField("from")
        toID= feature.GetField("to")
        geom = feature.GetGeometryRef()
        #print("from-to:", fromID, toID)
        #print(geom)
        
        for i in range(0, geom.GetPointCount()):
            pt = geom.GetPoint(i)
            #print( "%i). POINT (%d %d)" %(i, pt[0], pt[1]) )
            if i == 0:
                dMS= (mNodes[:,1] - pt[0])**2 + (mNodes[:,2] - pt[1])**2
                if np.min(dMS) <= MaxDistSquared:
                    indx= np.argmin(dMS)
                    nodeSCode= mNodes[indx,0]     #node source code
                else:
                    nodeSCode= countN
                    listNo.append([nodeSCode, pt[0], pt[1]])
                    countN += 1
                    mNodes[countMN,:] = listNo[-1]
                    countMN += 1
                    
            elif i == (geom.GetPointCount()-1):
                dMS= (mNodes[:,1] - pt[0])**2 + (mNodes[:,2] - pt[1])**2
                # do more!
                if np.min(dMS) <= MaxDistSquared:
                    indx= np.argmin(dMS)
                    nodeTCode= mNodes[indx,0]
                else:
                    nodeTCode= countN
                    listNo.append([nodeTCode, pt[0], pt[1]])
                    countN += 1
                    mNodes[countMN,:] = listNo[-1]
                    countMN += 1
                listEd.append([countE, nodeSCode, nodeTCode, 0, 3])
                countE += 1
            else:
                nodeTCode= countN  # node target code
                listNo.append([nodeTCode, pt[0], pt[1]])
                countN += 1
                listEd.append([countE, nodeSCode, nodeTCode, 0, 3])
                countE += 1
                nodeSCode = nodeTCode
                    
        # break
    layer.ResetReading()
    listNo= np.array(listNo)
    listEd= np.array(listEd)
    for i in range(listEd.shape[0]):
        node0= int(listEd[i, 1])
        # print(node0)
        # print("listNo[node0]",listNo[node0])
        x0 = listNo[node0 , 1:]
        node1= int(listEd[i, 2])
        x1 = listNo[node1 , 1:]
        listEd[i,3]= np.linalg.norm(x1-x0)
        
    return np.array(listNo), np.array(listEd)

def plotNetwork():
    noDB = np.loadtxt("./tmp/nodesdb0.csv", delimiter=",")
    edDB = np.loadtxt("./tmp/linksdb0.csv", delimiter=",")
    print(noDB.shape, edDB.shape)
    plt.figure(num=1, figsize=(12,4))
    for i in range(edDB.shape[0]):
        indS, indT= int(edDB[i,1]), int(edDB[i,2])
        # print("here",indS, indT)
        plt.plot([ noDB[indS,1], noDB[indT,1] ],[ noDB[indS,2],noDB[indT,2] ], c="r", lw=1)
    plt.scatter(noDB[:,1], noDB[:,2], s= 5)
    plt.axis('equal')
    plt.show()
    return
 

def main():
    # plotNetwork()
    
    iShpPath= "./tmp/edges.shp"
    noDB, edDB= readEdges(iShpPath)
    np.savetxt("./tmp/nodesdb0.csv", noDB, fmt="%.2f", delimiter=",")
    np.savetxt("./tmp/linksdb0.csv", edDB, fmt="%d", delimiter=",")
    return

if __name__ == "__main__":
    main()
    
