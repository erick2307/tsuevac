#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from evacrl import paths
from evacrl.sarsa import SARSA
from evacrl.qlearn import QLearning
import time
import pickle


def createVideo(filename, foldername, method='ql',area="kochi",
                simtime=30, meandeparture=15, options=None):
    """`options`: the ModelOptions the state matrix was trained with (default: the current defaults)."""
    # setup
    t0 = time.time()
    fn = paths.case_path(area, foldername, filename)
    videoNamefile = f"{method}_{area}_{filename[:-4]}.avi"
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = meandeparture*60
    simulTime = simtime*60

    # load files
    agentsProfileName = paths.case_path(area, "data", "agentsdb.csv")
    nodesdbFile = paths.case_path(area, "data", "nodesdb.csv")
    linksdbFile = paths.case_path(area, "data", "linksdb.csv")
    transLinkdbFile = paths.case_path(area, "data", "actionsdb.csv")
    transNodedbFile = paths.case_path(area, "data", "transitionsdb.csv")

    # check folders
    resultsfolder = paths.case_path(area, "results")
    figuresfolder = str(paths.FIGURES_DIR)
    if not os.path.exists(resultsfolder):
        os.mkdir(resultsfolder)
    if not os.path.exists(figuresfolder):
        os.mkdir(figuresfolder)

    # initiate class
    if method == 'sarsa':
        case = SARSA(agentsProfileName=agentsProfileName,
                    nodesdbFile=nodesdbFile,
                    linksdbFile=linksdbFile,
                    transLinkdbFile=transLinkdbFile,
                    transNodedbFile=transNodedbFile,
                    meanRayleigh=meanRayleighTest,
                    discount=0.9,
                    folderStateNames=foldername,
                    options=options)

    if method == 'ql':
        case = QLearning(agentsProfileName=agentsProfileName,
            nodesdbFile=nodesdbFile,
            linksdbFile=linksdbFile,
            transLinkdbFile=transLinkdbFile,
            transNodedbFile=transNodedbFile,
            meanRayleigh=meanRayleighTest,
            discount=0.9,
            folderStateNames=foldername,
            options=options)

    # input policy
    case.loadStateMatrixFromFile(namefile=fn)

    # output population initial condition
    outnamefile = paths.case_path(area, "results", "agents_startcondition.csv")
    case.exportAgentDBatTimet(outnamefile)

    # setup canvas
    case.setFigureCanvas()

    # start simulation
    for t in range(int(min(case.pedDB[:, 9])), simulTime):
        case.initEvacuationAtTime()
        case.stepForward()
        optimalChoice = bool(np.random.choice(2,
                             p=[randomChoiceRate, optimalChoiceRate]))
        case.checkTarget(ifOptChoice=optimalChoice)
        if not t % 10:
            print(t)
            case.getSnapshotV2()
            case.computePedHistDenVelAtLinks()
            case.updateVelocityAllPedestrians()

    # output population condition

    outnamefile = paths.case_path(area, "results", "agents_finalcondition.csv")
    case.exportAgentDBatTimet(outnamefile)

    # output population path and time (this is a list of arrays)
    # print(case.expeStat)
    fname = paths.case_path(area, "results", "agents_experience.pkl")
    f = open(fname, "wb")
    pickle.dump(case.expeStat, f)
    f.close()
    # np.savetxt(fname, case.expeStat, delimiter=',')

    case.makeVideo(nameVideo=videoNamefile)
    case.destroyCanvas()
    # case.deleteFigures()
    case = None
    print("\n***** Video created (%.2f seconds) *****" % (time.time() - t0))
    return


def main():
    filename = "sim_000000153.csv"  # name of state matrix to load
    area = "kochi"
    foldername = "state_ql_mod_30_15_1000"  # folder where state matrices are saved
    timeSimulation = 60  # total time of simulation in minutes
    meandeparture = 5  # this is the actual evacuation behavior in minutes
    # (not necessary the trained behavior)
    method = 'ql'
    createVideo(filename=filename, foldername=foldername, method=method, area=area,
                simtime=timeSimulation, meandeparture=meandeparture)
    return


if __name__ == "__main__":
    main()
