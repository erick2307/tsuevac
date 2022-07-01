#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import time

import numpy as np

from qlearn import QLearning

# from sarsa import SARSA


def createVideo(
    filename, foldername, method="ql", area="kochi", simtime=30, meandeparture=15
):
    # setup
    t0 = time.time()
    fn = os.path.join(area, foldername, filename)
    videoNamefile = f"{method}_{area}_{filename[:-4]}.avi"
    optimalChoiceRate = 0.99
    randomChoiceRate = 1.0 - optimalChoiceRate
    meanRayleighTest = meandeparture * 60
    simulTime = simtime * 60

    # load files
    agentsProfileName = os.path.join(area, "data", "agentsdb.csv")
    nodesdbFile = os.path.join(area, "data", "nodesdb.csv")
    linksdbFile = os.path.join(area, "data", "linksdb.csv")
    transLinkdbFile = os.path.join(area, "data", "actionsdb.csv")
    transNodedbFile = os.path.join(area, "data", "transitionsdb.csv")

    # check folders
    resultsfolder = os.path.join(area, "results")
    figuresfolder = os.path.join("figures")
    if not os.path.exists(resultsfolder):
        os.mkdir(resultsfolder)
    if not os.path.exists(figuresfolder):
        os.mkdir(figuresfolder)

    # initiate class
    if method == "sarsa":
        case = SARSA(
            agentsProfileName=agentsProfileName,
            nodesdbFile=nodesdbFile,
            linksdbFile=linksdbFile,
            transLinkdbFile=transLinkdbFile,
            transNodedbFile=transNodedbFile,
            meanRayleigh=meanRayleighTest,
            discount=0.9,
            folderStateNames=foldername,
        )

    if method == "ql":
        case = QLearning(
            agentsProfileName=agentsProfileName,
            nodesdbFile=nodesdbFile,
            linksdbFile=linksdbFile,
            transLinkdbFile=transLinkdbFile,
            transNodedbFile=transNodedbFile,
            meanRayleigh=meanRayleighTest,
            discount=0.9,
            folderStateNames=foldername,
        )

    # input policy
    case.loadStateMatrixFromFile(namefile=fn)

    # output population initial condition
    outnamefile = os.path.join(area, "results", "agents_startcondition.csv")
    case.exportAgentDBatTimet(outnamefile)

    # setup canvas
    case.setFigureCanvas()

    # start simulation
    for t in range(int(min(case.pedDB[:, 9])), simulTime):
        case.initEvacuationAtTime()
        case.stepForward()
        optimalChoice = bool(
            np.random.choice(2, p=[randomChoiceRate, optimalChoiceRate])
        )
        case.checkTarget(ifOptChoice=optimalChoice)
        if not t % 10:
            print(t)
            case.getSnapshotV2()
            case.computePedHistDenVelAtLinks()
            case.updateVelocityAllPedestrians()

    # output population condition

    outnamefile = os.path.join(area, "results", "agents_finalcondition.csv")
    case.exportAgentDBatTimet(outnamefile)

    # output population path and time (this is a list of arrays)
    # print(case.expeStat)
    fname = os.path.join(area, "results", "agents_experience.pkl")
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
    filename = "sim_000000010.csv"  # name of state matrix to load
    area = "arahama"
    foldername = "state_db_ql_mod_60_14"  # folder where state matrices are saved
    timeSimulation = 60  # total time of simulation in minutes
    meandeparture = 14  # this is the actual evacuation behavior in minutes
    # (not necessary the trained behavior)
    method = "ql"
    createVideo(
        filename=filename,
        foldername=foldername,
        method=method,
        area=area,
        simtime=timeSimulation,
        meandeparture=meandeparture,
    )
    return


if __name__ == "__main__":
    main()
