# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:38:45 2018

@author: Moya
"""
# from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import os
# import glob
import numpy as np


def dataBaseBy4thOrderMesh(bldShp, meshShp, meshCode, outBldDb_path,
                           outCensusDb_path):
    driverMesh = ogr.GetDriverByName("ESRI Shapefile")
    dataSourceMesh = driverMesh.Open(meshShp, 0)
    layerMesh = dataSourceMesh.GetLayer()
    layerMesh.SetAttributeFilter("C2 = '%s'" % meshCode)
    driverBld = ogr.GetDriverByName("ESRI Shapefile")
    # creating output file with censo data for disaggregation
    census_file = os.path.join(outCensusDb_path, "%s_Censo.csv" % meshCode)
    outfileCenso = open(census_file, 'w')
    outfileCenso.write("MeshCode,")
    outfileCenso.write("M0-4,F0-4,M5-9,F5-9,M10-14,F10-14,M15-19,F15-19,M20-24,F20-24,")
    outfileCenso.write("M25-29,F25-29,M30-34,F30-34,M35-39,F35-39,M40-44,F40-44,M45-49,F45-49,")
    outfileCenso.write("M50-54,F50-54,M55-59,F55-59,M60-64,F60-64,M65-69,F65-69,M70-74,F70-74,")
    outfileCenso.write("M75-79,F75-79,M80-84,F80-84,M_GTOE85,F_GTOE85,")
    outfileCenso.write("HH1Person,HH2People,HH3People,HH4People,HH5People,HH6People,HHMoreThan7\n")
    fieldCodes = ["C1","C15","C16","C18","C19","C21","C22","C24","C25","C27","C28",
                  "C30","C31","C33","C34","C36","C37","C39","C40","C42","C43",
                  "C45","C46","C48","C49","C51","C52","C54","C55","C57","C58",
                  "C60","C61","C63","C64","C66","C67",
                  "C117","C118","C119","C120","C121","C122","C123"]
    numFields = len(fieldCodes)
    # creating output file with building database
    bld_file = os.path.join(outBldDb_path, "%s_BldDb.csv" % meshCode)
    outfileBld = open(bld_file, 'w')
    outfileBld.write("BldCode,CenterX,CenterY,Height,Area,Mesh100MCode\n")
    targetSpaRef = osr.SpatialReference()
    # JGD2011 / UTM zone 53N - Kochi
    targetSpaRef.ImportFromEPSG(6690)
    # UTM zone 54N - Arahama
    # targetSpaRef.ImportFromEPSG(32654)
    for featureMesh in layerMesh:

        # Getting bld information
        dataSourceBld = driverBld.Open(bldShp, 0)
        layerBld = dataSourceBld.GetLayer()
        layerBld.SetSpatialFilter(featureMesh.geometry())
        # getting the spatial reference of the shapefile
        sourceSpaRef = layerBld.GetSpatialRef()
        # transformation function
        src2trg = osr.CoordinateTransformation(sourceSpaRef, targetSpaRef)
        countBld = 0
        for featureBld in layerBld:
            centerBld = featureBld.geometry().Centroid()
            geometry = featureBld.GetGeometryRef()
            geometry.Transform(src2trg)
            if (not centerBld.Within(featureMesh.geometry())) or (featureBld.GetField("LAYERCODE") != 233):
                continue
            # print featureBld.GetField("OBJECTID")
            countBld += 1
            outfileBld.write("%d,%f,%f,%d,%f,%s\n" % (featureBld.GetField("OBJECTID"), centerBld.GetX(), centerBld.GetY(), featureBld.GetField("HEIGHT"), geometry.GetArea(), featureMesh.GetField("C1")))
        if not countBld:
            continue
        # print("Mesh code: ", featureMesh.GetField("C1"))
        # print("count bld: ", countBld)
        # Getting census information
        outfileCenso.write("%s," % featureMesh.GetField(fieldCodes[0]))
        for i in range(1, numFields-1):
            if featureMesh.GetField(fieldCodes[i]) is None:
                outfileCenso.write(",")
            else:
                outfileCenso.write("%.2f," % featureMesh.GetField(fieldCodes[i]))
        if featureMesh.GetField(fieldCodes[-1]) is None:
            outfileCenso.write("\n")
        else:
            outfileCenso.write("%.2f\n" % featureMesh.GetField(fieldCodes[-1]))
        dataSourceBld = None

    dataSourceMesh = None
    outfileCenso.close()
    outfileBld.close()
    return


def createCensusAndPopDatabase(bldShp, meshShp, outBldDb_path,
                               outCensusDb_path, fileOriginalDataSource):
    dataSubmeshCode = np.loadtxt(fileOriginalDataSource.decode("utf-8"),
                                 skiprows=1, usecols=(3,), delimiter='"',
                                 dtype=np.str)
    submeshCodeArray = np.unique(dataSubmeshCode)
#    print(submeshCodeArray)
#    print(submeshCodeArray[0], len(submeshCodeArray), type(submeshCodeArray))
    for sm in submeshCodeArray:
        print(sm)
        dataBaseBy4thOrderMesh(bldShp, meshShp, sm, outBldDb_path,
                               outCensusDb_path)


if __name__ == "__main__":
    fileOriginalDataSource = r"E:\\DIM2SEA\\SendaiGIS\\Censo\\04宮城県\\国勢調査2010メッシュ100M人口04_Sendai.csv"
    bldShp = r"E:\DIM2SEA\SendaiGIS\Censo\SendaiUseBld.shp"
    meshShp = r"E:\DIM2SEA\SendaiGIS\Censo\CensusSendaiPopulation_Mesh100M.shp"
    outBldDb_path = r"E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Building_database"
    outCensusDb_path = r"E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Census_database"
    createCensusAndPopDatabase(bldShp, meshShp, outBldDb_path, outCensusDb_path, fileOriginalDataSource)
#    bldShp = r"E:\DIM2SEA\SendaiGIS\Censo\SendaiUseBld.shp"
#    meshShp = r"E:\DIM2SEA\SendaiGIS\Censo\CensusSendaiPopulation_Mesh100M.shp"
#    meshCode = "574037204"
#    outBldDb_path = r"E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Building_database"
#    outCensusDb_path = r"E:\DIM2SEA\PythonCodes\DisaggregationAndAllocation\Output\Census_database"
#    dataBaseBy4thOrderMesh(bldShp, meshShp, meshCode, outBldDb_path, outCensusDb_path)
