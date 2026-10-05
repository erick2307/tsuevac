#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing evacrl

import numpy as np
import matplotlib.pyplot as plt
import os
from evacrl import cli, paths
import functools
from evacrl.qlearn import QLearning
from evacrl.mc import MonteCarlo
plt.ioff()

def run_shortpath(area="kochi", simtime=30, meandeparture=15,
                  numSim0=0, numBlocks=5, simPerBlock=1000, name='r', options=None):
    nextnodepath = paths.case_path(area, "data", "nextnode.csv")
    if not os.path.exists(nextnodepath):
        raise SystemExit(f"{nextnodepath} not found. The shortest-path baseline needs it; it is not part of this repository "
                         "(pre/defPathsFromNodes.py is the legacy script that produced such a file).")
    agentsProfileName = paths.case_path(area, "data", "agentsdb.csv")
    nodesdbFile = paths.case_path(area, "data", "nodesdb.csv")
    linksdbFile = paths.case_path(area, "data", "linksdb.csv")
    transLinkdbFile = paths.case_path(area, "data", "actionsdb.csv")
    transNodedbFile = paths.case_path(area, "data", "transitionsdb.csv")
    folderStateNames = paths.case_path(area, f"state_{name}")
    if not os.path.exists(folderStateNames):
        os.mkdir(folderStateNames)
    meanRayleighTest = meandeparture * 60
    simulTime = simtime * 60
    survivorsPerSim = []

    if numSim0 == 0:
        randomChoiceRate = 0.  # 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        case = QLearning(agentsProfileName=agentsProfileName,
                         nodesdbFile=nodesdbFile,
                         linksdbFile=linksdbFile,
                         transLinkdbFile=transLinkdbFile,
                         transNodedbFile=transNodedbFile,
                         meanRayleigh=meanRayleighTest,
                         folderStateNames=folderStateNames,
                         options=options)

        totalagents = np.sum(case.pedDB.shape[0])
        case.loadShortestPathDB(namefile=nextnodepath)
        # case.setFigureCanvas()
        # survivedAgents = np.zeros((simulTime, 3))

        for t in range(int(min(case.pedDB[:, 9])), simulTime):
            case.initEvacuationAtTime()
            case.stepForward()
            # optimalChoice = bool(np.random.choice(2, 1,
            #                      p=[randomChoiceRate, optimalChoiceRate]))
            # case.checkTarget(ifOptChoice=optimalChoice)
            case.checkTargetShortestPath()
            if not t % 10:
                case.computePedHistDenVelAtLinks()
                case.updateVelocityAllPedestrians()
            # if not t % 60:
            #     print(t)
            #     case.getSnapshotV2()

            # survivedAgents[t,0] = np.sum( np.isin(case.pedDB[:,8], case.evacuationNodes[0]) )
            # survivedAgents[t,1] = np.sum( np.isin(case.pedDB[:,8], case.evacuationNodes[1]) )
            # survivedAgents[t,2] = np.sum( np.isin(case.pedDB[:,8], case.evacuationNodes[2]) )

        # outfile = os.path.join(folderStateNames, "sim_%09d.csv" % numSim0)
        # case.exportStateMatrix(outnamefile=outfile)
        # print(f"""\n\n ***** Simu {numSim0:d}
        #     (t= {(time.time()-t0)/60.0:.2f})*****""")
        # print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print(f"""survived: {np.sum(case.pedDB[:,10] == 1)}
            / total: {totalagents}""")

        survivorsPerSim.append([numSim0, np.sum(case.pedDB[:, 10] == 1)])
        fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}.csv"
        outSurvivors = os.path.join(folderStateNames, fname)
        np.savetxt(outSurvivors, np.array(survivorsPerSim),
                   delimiter=",", fmt="%d")
        evacs_list = [evacs[1] for evacs in survivorsPerSim]
        print(f"""Max value:{max(evacs_list)},
            Index:{evacs_list.index(max(evacs_list))}""")
        if survivorsPerSim[-1] == case.pedDB.shape[0]:
            return

        case = None

    numSim = numSim0 + 1
    for b in range(numBlocks):
        for s in range(simPerBlock):
            eoe = int(0.8 * simPerBlock)  # end of exploration
            if s < eoe:
                randomChoiceRate = -1 / (eoe) ** 2 * s ** 2 + 1
            else:
                randomChoiceRate = 0.0
            #  randomChoiceRate = (simPerBlock - s - 1.0)/
            # (simPerBlock - s + 1.0) #1.0/(0.015*s + 1.0)
            # added to check if this is Q-Learning 2021.08.03
            randomChoiceRate = 0.0
            optimalChoiceRate = 1.0 - randomChoiceRate
            case = QLearning(agentsProfileName=agentsProfileName,
                             nodesdbFile=nodesdbFile,
                             linksdbFile=linksdbFile,
                             transLinkdbFile=transLinkdbFile,
                             transNodedbFile=transNodedbFile,
                             meanRayleigh=meanRayleighTest,
                             folderStateNames=folderStateNames,
                             options=options)

            # Modified Oct 4, 2021
            # Check best state and load that one
            # namefile = os.path.join(folderStateNames, "sim_%09d.csv" % index)
            # case.loadStateMatrixFromFile(namefile=namefile)
            totalagents = np.sum(case.pedDB.shape[0])

            for t in range(int(min(case.pedDB[:, 9])), simulTime):
                case.initEvacuationAtTime()
                case.stepForward()
                np.random.choice(2, 1, p=[randomChoiceRate, optimalChoiceRate])  # the draw is kept: it advances the seeded stream
                # case.checkTarget(ifOptChoice=optimalChoice)
                case.checkTargetShortestPath()
                if not t % 10:
                    case.computePedHistDenVelAtLinks()
                    case.updateVelocityAllPedestrians()

            # outfile = os.path.join(folderStateNames, "sim_%09d.csv" % numSim)
            # case.exportStateMatrix(outnamefile=outfile)
            # print("\n\n ***** Simu %d (t= %.2f)*****" % (numSim,
            #       (time.time() - t0) / 60.))
            # print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print(f"""survived: {np.sum(case.pedDB[:,10] == 1)}
                  / total: {totalagents}""")

            # evaluate survivors in simulation
            survivorsPerSim.append([numSim, np.sum(case.pedDB[:, 10] == 1)])
            fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}.csv"
            outSurvivors = os.path.join(folderStateNames, fname)
            np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter=",",
                       fmt="%d")
            evacs_list = [evacs[1] for evacs in survivorsPerSim]
            print(f"""Max value:{max(evacs_list)},
                   Index:{evacs_list.index(max(evacs_list))}""")
            if survivorsPerSim[-1] == case.pedDB.shape[0]:
                return
            case = None
            numSim += 1
    return

    # case.makeVideo(nameVideo="Simulation20191211_shortestPath_OriginalCensus.avi")
    # case.destroyCanvas()
    # case.deleteFigures()
    # case = None
    # np.savetxt("agentsAtEvacuationNodesVsTime_shortesPath_OriginalCensus.csv", survivedAgents, delimiter=",")
    # return

def ArahamaMTRL_20191220_Video():
    folderStateNames = "Arahama_20191220"
    stateSimFile = "sim_0498.csv"
    namefile = os.path.join(folderStateNames, stateSimFile)
    fileNameAgentsAtEvacNodevsTime = "agentsAtEvacuationNodesVsTime.csv"
    videoNamefile = "ArahamaTest2020Oct06.avi"
    meanRayleighTest = 20 * 60
    simulTime = 67 * 60
    agentsProfileName = "IPF_AgentsCoordV2.csv"
    optimalChoiceRate = 0.9
    randomChoiceRate = 1.0 - optimalChoiceRate

    survivedAgents = np.zeros((simulTime, 3))

    arahama = MonteCarlo(agentsProfileName=agentsProfileName, meanRayleigh=meanRayleighTest)
    arahama.loadStateMatrixFromFile(namefile=namefile)
    arahama.setFigureCanvas()

    for t in range(int(min(arahama.pedDB[:, 9])), simulTime):
        arahama.initEvacuationAtTime()
        arahama.stepForward()
        optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate, optimalChoiceRate]))
        arahama.checkTarget(ifOptChoice=optimalChoice)
        if not t % 5:
            print(t)
            arahama.getSnapshotV2()
            arahama.computePedHistDenVelAtLinks()
            arahama.updateVelocityAllPedestrians()
        survivedAgents[t, 0] = np.sum(np.isin(arahama.pedDB[:, 8], arahama.evacuationNodes[0]))
        survivedAgents[t, 1] = np.sum(np.isin(arahama.pedDB[:, 8], arahama.evacuationNodes[1]))
        survivedAgents[t, 2] = np.sum(np.isin(arahama.pedDB[:, 8], arahama.evacuationNodes[2]))

    arahama.makeVideo(nameVideo=videoNamefile)
    arahama.destroyCanvas()
    arahama.deleteFigures()
    arahama = None
    np.savetxt(fileNameAgentsAtEvacNodevsTime, survivedAgents, delimiter=",")
    return


def SurvivedAgentsPerEvacuationNode():
    fileNameAgentsAtEvacNodevsTime = "agentsAtEvacuationNodesVsTime.csv"
    rlDb = np.loadtxt(fileNameAgentsAtEvacNodevsTime, delimiter=',')

    plt.figure(num="comparison")
    plt.subplot(1, 4, 1)
    plt.plot(rlDb[:, 0], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 1")
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(rlDb[:, 1], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 2")
    plt.grid()

    plt.subplot(1, 4, 3)
    plt.plot(rlDb[:, 2], color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at node")
    plt.title("Evacuation node 3")
    plt.grid()

    plt.subplot(1, 4, 4)
    plt.plot(np.sum(rlDb, axis=1), color="b", label="Montecarlo")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of evacuees at nodes")
    plt.title("All evacuation nodes")
    plt.grid()

    plt.show()
    return

def arahama_shortpath():
    run_shortpath(area="arahama", simtime=30, meandeparture=15,
                  numSim0=0, numBlocks=1, simPerBlock=10, name='a')

if __name__ == "__main__":
    # python scripts/main_ShortPath.py [kochi | arahama | new_kochi]   (default: arahama)
    _run = functools.partial(run_shortpath, simtime=30, meandeparture=15, numSim0=0, numBlocks=1, simPerBlock=10, name='a')
    cli.run_case({"arahama": arahama_shortpath, "kochi": functools.partial(_run, area="kochi"),
                  "new_kochi": functools.partial(_run, area="new_kochi")}, default="arahama")
    # SurvivedAgentsPerEvacuationNode()
    # ArahamaMTRL_20191220_Video()
    # ArahamaMTRL_20191220_SeqSim()
