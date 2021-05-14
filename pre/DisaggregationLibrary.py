# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:36:44 2018

@author: Moya
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

class PopulationDisaggregation:
    def __init__(self, censoFile):
        self.dataCenso = np.loadtxt(censoFile, usecols = range(1,44), skiprows=1, delimiter=',')
        print("dataCenso")
        print(self.dataCenso.shape, len(self.dataCenso), len(self.dataCenso.shape))
        if len(self.dataCenso.shape) == 1:
            self.dataCenso = self.dataCenso.reshape((1,43))
        print(self.dataCenso)
        self.mesh4thOrderCode = np.loadtxt(censoFile, usecols = (0,), skiprows=1, delimiter=',', dtype = np.str, ndmin=1)
        self.population = int(np.sum(self.dataCenso[:,:36]))
        print("population")
        print(self.population)
        self.attributes = 2     # age by gender AND household type
        self.categories = np.array([36, 7])
        self.distributions = [np.round(np.sum(self.dataCenso[:,:36], axis=0)), np.round(np.sum(self.dataCenso[:,36:], axis=0)*(1+np.arange(7)))]
        self.distributions[0][np.where(self.distributions[0] == 0.)] += 0.001
        self.distributions[1][np.where(self.distributions[1] == 0.)] += 0.001
        print("distributions")
        print(self.distributions)
        self.f_scale = np.ones(self.population)
        self.db = np.zeros((self.population, self.attributes + 1))
        self.db[:,0] = np.arange(self.population)
        self.db[:,1] = np.random.randint(0, self.categories[0], self.population)
        self.top_child_cat = 5   # from 0 to 5: the label for people younger than 19 years old
        n_childs = np.sum(self.db[:,1] < self.top_child_cat + 1)
        self.db[:,2] = np.random.randint(0, self.categories[1], self.population)
        self.db[:,2][self.db[:,1] < (self.top_child_cat + 1)] = np.random.randint(1, self.categories[1], n_childs)
#        print("db")
#        print(self.db)
        
#        def __init__(self, totalPop, numAtt, numCat, distributions, top_child_cat = 5):
#        self.population = totalPop
#        self.attributes = numAtt
#        self.categories = numCat
#        self.distributions = distributions
#        self.f_scale = np.ones(self.population)
#        self.db = np.zeros((self.population, self.attributes + 1))
#        self.db[:,0] = np.arange(self.population)
#        self.db[:,1] = np.random.randint(0, self.categories[0], self.population)
#        self.top_child_cat = top_child_cat
#        n_childs = np.sum(self.db[:,1] < self.top_child_cat + 1)
#        self.db[:,2] = np.random.randint(0, self.categories[1], self.population)
#        self.db[:,2][self.db[:,1] < (self.top_child_cat + 1)] = np.random.randint(1, self.categories[1], n_childs)
    
    def getBidimensionalDistribution(self):
        MD = np.zeros((self.categories[0], self.categories[1]))
        for i in range(self.categories[0]):
            ind_i = self.db[:,1] == i
            for j in range(self.categories[1]):
                ind_j = self.db[:,2] == j
                ind_and = np.logical_and(ind_i, ind_j)
                MD[i,j] = np.sum(self.f_scale[ind_and])
        return MD
    
    def getError(self):
        MD = self.getBidimensionalDistribution()
        error = np.sqrt(np.sum((self.distributions[0] - np.sum(MD, axis=1))**2) + np.sum((self.distributions[1] - np.sum(MD, axis=0))**2))
        error /= np.sum(self.categories)
        return error
    
    def IPF(self, errorTol = 0.01, max_iter = 500, ifPlot = False):
        print("db")
        print(self.db)
        i = 0
        md_tmp = self.getBidimensionalDistribution()
        print("md_tmp")
        print(md_tmp)
        print(np.sum(md_tmp, axis=1))
        print(np.sum(md_tmp, axis=0))
        if ifPlot:
            plt.figure(num=1, figsize=(8,3.5))
            plt.subplots_adjust(wspace=0.3)
            plt.subplot(1,2,2)
            plt.xlabel("Aggregated from Census")
            plt.ylabel("Aggregated from Simulation")
            plt.title("Household attribute types")
            plt.scatter(self.distributions[1], np.sum(md_tmp, axis=0), s=20, c='g', label="Initial estimation")
            plt.subplot(1,2,1)
            plt.xlabel("Aggregated from Census")
            plt.ylabel("Aggregated from Simulation")
            plt.title("Age-gender attribute types")
            plt.scatter(self.distributions[0], np.sum(md_tmp, axis=1), s=20, c='g', label="Initial estimation")
        
        while True:
            for j in range(self.attributes):
                for k in range(self.categories[j]):
                    indxTmp = (self.db[:,j+1] == k)
                    r = np.sum(self.f_scale[indxTmp])
                    r = self.distributions[j][k] / r
                    self.f_scale[indxTmp] *= r
                    md_tmp = self.getBidimensionalDistribution()
#                    plt.scatter(self.distributions[1], np.sum(md_tmp, axis=0), s=20, c='r')
                if self.getError() < errorTol:
                    print("f_scale")
                    print(np.round(self.f_scale, decimals = 2))
                    return
                i += 1
                if i > max_iter:
                    print("IPF: max iteration reach")
                    if ifPlot:
                        plt.subplot(1,2,2)
                        plt.scatter(self.distributions[1], np.sum(md_tmp, axis=0), s=50, c='b', label="Final estimation")
                        plt.legend()
                        plt.subplot(1,2,1)
                        plt.scatter(self.distributions[0], np.sum(md_tmp, axis=1), s=50, c='b', label="Final estimation")
                        plt.legend()
#                        figname = r"E:\DIM2SEA\MyJournals\Disagregation_Report\Figures\adjustedmarginals.eps"
#                        plt.savefig(figname, bbox_inches = 'tight', format='eps')
                        plt.show()
                    print("f_scale")
                    print(np.round(self.f_scale, decimals = 2))
                    return
#                print("run one iteration")
#                print(self.f_scale)
        
#            print("factor scale: ", self.getFactorScale()[:4])
    
    def getFactorScale(self):
        return self.f_scale

    def createSyntheticPopulation(self, outfilePop, outfileHH, bldDBFile):
        print("\n***************************************************")
        print("*********** CREATE_SYNTHETIC_POPULATION ***********")
        print("***************************************************\n")
        MD = np.round(self.getBidimensionalDistribution(), decimals=1)#.astype(np.int)
        MD = np.ceil(MD).astype(np.int)
        print("MD")
        print(MD)
        print("MD-no round")
        print(np.round(self.getBidimensionalDistribution(), decimals=2))
        adjusted_total_people = int(np.sum(MD))
        print("adjusted_total_people", adjusted_total_people)
        print(np.sum(MD, axis=0))
        print(np.sum(MD, axis=1))
        ASP = np.zeros((adjusted_total_people, 5))    # ASP: adjusted synthetic people; 5 attributes: [ID, Age, Gender, H_attribute, HH_code]
        ASP[:,0] = np.arange(adjusted_total_people)
        
        ind_id = 0   # index of people id
        
        for c in range(self.categories[1]):
            for r in range(self.categories[0]):
                if not MD[r,c]:
                    continue
                tmpDb = np.zeros((MD[r,c],2))   # temporal Database:   [Age, Gender]
                # Assigning gender
                if r%2:
                    tmpDb[:,1] = np.ones(MD[r,c])    # female person are labeled as 1
                else:
                    tmpDb[:,1] = np.zeros(MD[r,c])    # male person are labeled as 0
                # Assigning year
                intTmp = int(r/2)
                tmpDb[:,0] = np.random.randint(intTmp*5, (intTmp + 1)*5, MD[r,c])
                # Updated ASP database
                ASP[ind_id:ind_id + MD[r,c], 1:3] = tmpDb
                ASP[ind_id:ind_id + MD[r,c], 3] = (c + 1) * np.ones(MD[r,c]) # updating household code
                ind_id += MD[r,c]
        
        print("ASP: adjusted synthetic people")
        print(ASP.astype(np.int))

        adjustNumHH = np.ceil(np.sum(MD, axis=0)/(np.arange(len(self.distributions[1]))+1)).astype(np.int)
        print("adjustNumHH")
        print(adjustNumHH)
        totalHH = np.sum(adjustNumHH, dtype = np.int) # total adjusted households
        
        # Set household database 
        HH_DB = np.zeros((totalHH, 3))
        HH_DB[:,0] = np.arange(totalHH)  # household ID
        HH_DB[:,1] = np.array(adjustNumHH[0]*[1] + adjustNumHH[1]*[2] + adjustNumHH[2]*[3] + 
                     adjustNumHH[3]*[4] + adjustNumHH[4]*[5] + adjustNumHH[5]*[6] + adjustNumHH[6]*[7])   # household HH_code
                     
        # Assigning household ID to each synthetic person
        totalSubSample = np.sum(MD[:,0], dtype = np.int)
        hhCodeTmp = np.random.choice(np.arange(totalSubSample), size = totalSubSample, replace = False)
        ASP[0:totalSubSample, -1] = hhCodeTmp
        accum_total = totalSubSample
        accum_HH = adjustNumHH[0]
        
        for c in range(1, self.categories[1]-1):
            
            totalSubSample = int(np.sum(MD[:,c]))
            hhMatrixTmp = np.random.choice(np.arange(accum_total, accum_total + adjustNumHH[c]*(c + 1)), size=(adjustNumHH[c], c+1), replace=False)
            
            for h in range(adjustNumHH[c]):
                ASP[hhMatrixTmp[h],-1] = accum_HH + h  # assigning hh code for each family (each row of hhMatrixTmp represent a family)
            accum_total += totalSubSample
            accum_HH += adjustNumHH[c]
        totalSubSample = int(np.sum(MD[:,-1]))
#        ASP[accum_total:,-1] = np.random.random_integers(accum_HH, totalHH-1, totalSubSample)
        print("accum_HH, totalHH-1")
        print(accum_HH, totalHH-1)
        print("accum_HH, totalHH-1, totalSubSample")
        print(accum_HH, totalHH-1, totalSubSample)
        if accum_HH >= (totalHH-1):
            ASP[accum_total:,-1] = np.random.randint(totalHH-1, totalHH, totalSubSample)
        else:
            ASP[accum_total:,-1] = np.random.randint(accum_HH, totalHH-1, totalSubSample)
        np.savetxt(outfilePop,  ASP, delimiter=',', header="ID,Age,Gender,HHtype,HH_ID", fmt=['%d', '%d', '%d', '%d', '%d'])

        
        ###### Coordinate alocation ######
        print("self.mesh4thOrderCode")
        print(self.mesh4thOrderCode)
        numMesh4Order = len(self.mesh4thOrderCode)
        adjHHMesh4Order = np.zeros((numMesh4Order, 7))   # number of households by type for each sub mesh
        
        #  adjusting the number of households
        for i in range(7):
            factorScale = np.nan_to_num( float(adjustNumHH[i])/np.sum(self.dataCenso[:, 36 + i]) )
            adjHHMesh4Order[:,i] = np.round(self.dataCenso[:, 36 + i] * factorScale)
        remainHH = adjustNumHH - np.sum(adjHHMesh4Order, axis=0)
        print("adjustNumHH", adjustNumHH)
        print("np.sum(adjHHMesh4Order, axis=0)", np.sum(adjHHMesh4Order, axis=0))
        for i in range(7):    # adjusting the remain
            print("i, remainHH[i]", i, remainHH[i])
            if not remainHH[i]:
                continue
            elif remainHH[i] > 0:
                adjHHMesh4Order[np.random.choice(np.arange(numMesh4Order), size= int(remainHH[i]), replace = False), i] += 1
            else:
#                print(np.where(adjHHMesh4Order[:,i] != 0)[0])
                adjHHMesh4Order[np.random.choice(np.where(adjHHMesh4Order[:,i] != 0)[0], size= -1 * int(remainHH[i]), replace = False), i] -= 1
        HH_SubMeshLabel = []
        for i in range(7):
            for j in range(len(adjHHMesh4Order)):
                HH_SubMeshLabel += int(adjHHMesh4Order[j,i])*[self.mesh4thOrderCode[j]]
        HH_SubMeshLabel = np.array(HH_SubMeshLabel, dtype = np.str)
        
        print("HH_SubMeshLabel")
        print(HH_SubMeshLabel)
        
        # Assigning building code
        bldData = np.loadtxt(bldDBFile, delimiter = ",", skiprows = 1)
        if len(bldData.shape) == 1:
            bldData = bldData.reshape((1,len(bldData)))
            
        hh_per_submesh = np.sum(adjHHMesh4Order, axis = 1, dtype = np.int)
        
        print("hh_per_submesh")
        print(hh_per_submesh)
        
        for i in range(numMesh4Order):
#            bldDataSubmesh = bldData[ bldData[:,-1] == int(self.mesh4thOrderCode[i]) ]
            bldDataSubmesh = bldData
            
            print("checking here")
            print(bldDataSubmesh)
            print(bldDataSubmesh[:,3])
            print(bldDataSubmesh[:,4])
            total_vol = np.dot(bldDataSubmesh[:,3], bldDataSubmesh[:,4])
            total_hh = hh_per_submesh[i]
            total_bld = len(bldDataSubmesh)
            
            if (total_hh < total_bld):
#                print("numbers of households are lower than number of buildings in submesh %s" % self.mesh4thOrderCode[i])
                bld2hh_array = np.random.choice(bldDataSubmesh[:,0], size = total_hh, replace = False).astype(np.int)
                HH_DB[ HH_SubMeshLabel == self.mesh4thOrderCode[i], 2 ] = bld2hh_array
            else:
                hh_per_bld = np.ones(total_bld)
                hh_per_bld += np.floor(bldDataSubmesh[:,3] * bldDataSubmesh[:,4] * float(total_hh - total_bld) / float(total_vol)).astype(np.int)
                remain_bld = int(total_hh - np.sum(hh_per_bld))
                hh_per_bld[np.random.choice(np.arange(total_bld).astype(np.int), remain_bld, replace = False)] += 1    # adjusting the remain buildings
                bld2hh_array = []
                for j in range(total_bld):
                    bld2hh_array += int(hh_per_bld[j]) * [bldDataSubmesh[j, 0]]
                bld2hh_array = np.array(bld2hh_array, dtype = np.int)
                bld2hh_array = np.random.choice(bld2hh_array, size = total_hh, replace = False)
                HH_DB[ HH_SubMeshLabel == self.mesh4thOrderCode[i], 2 ] = bld2hh_array
            
        
        outfile1 = open(outfileHH, 'w')
        outfile1.write('# ID,HH_type,SubMeshCode, building_code\n')
        for i in range(len(HH_DB)):
            outfile1.write("%d,%d,%s,%d\n" % (HH_DB[i,0],HH_DB[i,1], HH_SubMeshLabel[i], HH_DB[i,2]))
        outfile1.close()
        return

def disaggregationSendai():
    listCensusFiles = glob.glob("E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Census_database\*.csv")
    for cf in listCensusFiles:
        print(os.path.split(cf)[1])
        bldf = os.path.split(cf)[1].replace("Censo", "BldDb")
        bldf = os.path.join("E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Building_database", bldf)
        outfilePop = os.path.join("E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Population_database", os.path.split(cf)[1].replace("Censo", "PopDb"))
        outfileHH = os.path.join("E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Household_database", os.path.split(cf)[1].replace("Censo", "HHDb"))
#        print(bldf)
#        print(outfilePop)
#        print(outfileHH)
#        print(" ")
        
        meshDist = PopulationDisaggregation(censoFile)
        meshDist.IPF()
        meshDist.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)

def disaggregationKochiPrefecture():
    pathCensus = r"C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\CensusAndBuildingDatabase\Censo_Code*.csv"
    cenPathList = glob.glob(pathCensus)
    
    pathHHFolder = r"C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\Household_database"
    pathPopFolder = r"C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\Population_database"
    
    for fc in cenPathList:
        print(fc)
        pathFolder = os.path.split(fc)[0]
        codeArea = os.path.split(fc)[1].split("_")[1]
        
        if codeArea == "Code207.csv":
            continue
        elif codeArea == "Code299.csv":
            continue
        elif codeArea == "Code383.csv":
            continue
#        codeArea = 
        fileNameBld = "BldDb_" + codeArea
        fb = os.path.join( pathFolder , fileNameBld )
        fHh = os.path.join(pathHHFolder , "HH_" + codeArea)
        fPop = os.path.join(pathPopFolder , "Pop_" + codeArea  )
        print(fb)
        print(fHh)
        print(fPop)
        print("\n")
        
        if os.path.isfile(fHh) or os.path.isfile(fPop):
            continue
        
        disg = PopulationDisaggregation(fc)
        disg.IPF(ifPlot=False) 
        disg.createSyntheticPopulation(fPop, fHh, fb) 
        disg = None 

    return

 
if __name__ == "__main__":
    disaggregationKochiPrefecture()
    
#    censoFile = r"C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\CensusAndBuildingDatabase\Censo_Code212.csv"
#    disg = PopulationDisaggregation(censoFile)
#    disg.IPF(ifPlot=False) 
#    outfilePop = "C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\Population_database\popTest2.csv"
#    outfileHH = "C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\Household_database\HHtest2.csv"
#    bldFileDB = "C:\Users\Moya\ReGID Dropbox\Luis Moya\Python_codes\Reinforcement_learning\KochiPrefecture\CensusAndBuildingDatabase\BldDb_Code212.csv"
#    disg.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB) 
    
#    disaggregationSendai()
    
    ### Running specific mesh codes for reinforcement learning case study ###
    
#    censoFile = r"Output\\Census_database\\574027683_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027683_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027683_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027683_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027682_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027682_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027682_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027682_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027583_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027583_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027583_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027583_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027681_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027681_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027681_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027681_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027684_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027684_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027684_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027684_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027584_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027584_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027584_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027584_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027672_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027672_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027672_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027672_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"Output\\Census_database\\574027574_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
#    testDis.IPF()
#    outfilePop = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Population_database\\574027574_PopDb.csv"
#    outfileHH = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Household_database\\574027574_HHDb.csv"
#    bldFileDB = "E:\\DIM2SEA\\PythonCodes\\DisaggregationAndAllocation\\Output\\Building_database\\574027574_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
    
#    censoFile = r"SyntheticDataOutput\\574037204_Censo.csv"
#    testDis = PopulationDisaggregation(censoFile)
##    print(testDis.distributions[1])
##    print(testDis.distributions[0])
##    print(np.sum(testDis.distributions[1]))
##    print(np.sum(testDis.distributions[0]))
##    print(testDis.population())
#    testDis.IPF()
#    MD = testDis.getBidimensionalDistribution()
##    print(np.sum(MD, axis=0))
##    print(np.sum(MD, axis=1))
#    outfilePop = "SyntheticDataOutput\\Population20180131.csv"
#    outfileHH = "SyntheticDataOutput\\HH20180131.csv"
#    bldFileDB = "SyntheticDataOutput\\574037204_BldDb.csv"
#    testDis.createSyntheticPopulation(outfilePop, outfileHH, bldFileDB)
