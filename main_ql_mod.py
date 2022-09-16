#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import numpy as np

from qlearn import QLearning


def run_ql_mod(
    area="kochi",
    simtime=30,
    meandeparture=15,
    numSim0=0,
    numBlocks=5,
    simPerBlock=1000,
    name="r",
):
    t0 = time.time()
    agentsProfileName = os.path.join(area, "data", "agentsdb.csv")
    nodesdbFile = os.path.join(area, "data", "nodesdb.csv")
    linksdbFile = os.path.join(area, "data", "linksdb.csv")
    transLinkdbFile = os.path.join(area, "data", "actionsdb.csv")
    transNodedbFile = os.path.join(area, "data", "transitionsdb.csv")
    folderStateNames = os.path.join(area, f"state_{name}")
    if not os.path.exists(folderStateNames):
        os.mkdir(folderStateNames)
    meanRayleighTest = meandeparture * 60
    simulTime = simtime * 60
    survivorsPerSim = []

    if numSim0 == 0:
        randomChoiceRate = 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        case = QLearning(
            agentsProfileName=agentsProfileName,
            nodesdbFile=nodesdbFile,
            linksdbFile=linksdbFile,
            transLinkdbFile=transLinkdbFile,
            transNodedbFile=transNodedbFile,
            meanRayleigh=meanRayleighTest,
            discount=0.9,
            folderStateNames=folderStateNames,
        )

        totalagents = np.sum(case.pedDB.shape[0])

        for t in range(int(min(case.pedDB[:, 9])), simulTime):
            case.initEvacuationAtTime()
            case.stepForward()
            optimalChoice = bool(
                np.random.choice(2, 1, p=[randomChoiceRate, optimalChoiceRate])
            )
            case.checkTarget(ifOptChoice=optimalChoice)
            if not t % 10:
                case.computePedHistDenVelAtLinks()
                case.updateVelocityAllPedestrians()

        outfile = os.path.join(folderStateNames, "sim_%09d.csv" % numSim0)
        case.exportStateMatrix(outnamefile=outfile)
        print(
            f"""\n\n ***** Simu {numSim0:d}
              (t= {(time.time()-t0)/60.0:.2f})*****"""
        )
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print(
            f"""survived: {np.sum(case.pedDB[:,10] == 1)}
              / total: {totalagents}"""
        )

        survivorsPerSim.append([numSim0, np.sum(case.pedDB[:, 10] == 1)])
        fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}.csv"
        outSurvivors = os.path.join(folderStateNames, fname)
        np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter=",", fmt="%d")
        evacs_list = [evacs[1] for evacs in survivorsPerSim]
        print(
            f"""Max value:{max(evacs_list)},
              Index:{evacs_list.index(max(evacs_list))}"""
        )
        if survivorsPerSim[-1] == case.pedDB.shape[0]:
            return

        case = None

    numSim = numSim0 + 1
    for b in range(numBlocks):
        for s in range(simPerBlock):
            eoe = int(0.8 * simPerBlock)  # end of exploration
            if s < eoe:
                randomChoiceRate = -1 / (eoe) ** 2 * s**2 + 1
            else:
                randomChoiceRate = 0.0
            #  randomChoiceRate = (simPerBlock - s - 1.0)/
            # (simPerBlock - s + 1.0) #1.0/(0.015*s + 1.0)
            # added to check if this is Q-Learning 2021.08.03
            # randomChoiceRate = 0.0
            optimalChoiceRate = 1.0 - randomChoiceRate
            case = QLearning(
                agentsProfileName=agentsProfileName,
                nodesdbFile=nodesdbFile,
                linksdbFile=linksdbFile,
                transLinkdbFile=transLinkdbFile,
                transNodedbFile=transNodedbFile,
                meanRayleigh=meanRayleighTest,
                folderStateNames=folderStateNames,
            )

            # Modified Oct 4, 2021
            # Check best state and load that one
            index = evacs_list.index(max(evacs_list))
            namefile = os.path.join(folderStateNames, "sim_%09d.csv" % index)
            case.loadStateMatrixFromFile(namefile=namefile)
            totalagents = np.sum(case.pedDB.shape[0])

            for t in range(int(min(case.pedDB[:, 9])), simulTime):
                case.initEvacuationAtTime()
                case.stepForward()
                optimalChoice = bool(
                    np.random.choice(2, 1, p=[randomChoiceRate, optimalChoiceRate])
                )
                case.checkTarget(ifOptChoice=optimalChoice)
                if not t % 10:
                    case.computePedHistDenVelAtLinks()
                    case.updateVelocityAllPedestrians()

            outfile = os.path.join(folderStateNames, "sim_%09d.csv" % numSim)
            case.exportStateMatrix(outnamefile=outfile)
            print(
                "\n\n ***** Simu %d (t= %.2f)*****"
                % (numSim, (time.time() - t0) / 60.0)
            )
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print(
                f"""survived: {np.sum(case.pedDB[:,10] == 1)}
                  / total: {totalagents}"""
            )

            # evaluate survivors in simulation
            survivorsPerSim.append([numSim, np.sum(case.pedDB[:, 10] == 1)])
            fname = f"survivorsPerSim_{numBlocks}x{simPerBlock}.csv"
            outSurvivors = os.path.join(folderStateNames, fname)
            np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter=",", fmt="%d")
            evacs_list = [evacs[1] for evacs in survivorsPerSim]
            print(
                f"""Max value:{max(evacs_list)},
                   Index:{evacs_list.index(max(evacs_list))}"""
            )
            if survivorsPerSim[-1] == case.pedDB.shape[0]:
                return
            case = None
            numSim += 1
    return


def kochi_ql_mod():
    simtime = 30  # min
    meandeparture = 15  # min

    numSim0 = 0
    numBlocks = 1
    simPerBlock = 10

    name = f"ql_mod_{simtime}_{meandeparture}_{simPerBlock}"
    area = "kochi"

    run_ql_mod(
        area=area,
        simtime=simtime,
        meandeparture=meandeparture,
        numSim0=numSim0,
        numBlocks=numBlocks,
        simPerBlock=simPerBlock,
        name=name,
    )
    return


def arahama_ql_mod():
    simtime = 30  # min
    meandeparture = 7  # min

    numSim0 = 0
    numBlocks = 1
    simPerBlock = 10

    name = f"ql_mod_{simtime}_{meandeparture}_{simPerBlock}"
    area = "arahama"

    run_ql_mod(
        area=area,
        simtime=simtime,
        meandeparture=meandeparture,
        numSim0=numSim0,
        numBlocks=numBlocks,
        simPerBlock=simPerBlock,
        name=name,
    )
    return


def new_kochi_ql_mod():
    simtime = 30  # min
    meandeparture = 15  # min

    numSim0 = 0
    numBlocks = 1
    simPerBlock = 1000

    name = f"ql_{simtime}_{meandeparture}_{simPerBlock}"
    area = "new_kochi"

    run_ql_mod(
        area=area,
        simtime=simtime,
        meandeparture=meandeparture,
        numSim0=numSim0,
        numBlocks=numBlocks,
        simPerBlock=simPerBlock,
        name=name,
    )
    return


if __name__ == "__main__":
    t = time.time()
    arahama_ql_mod()
    # kochi_ql_mod()
    # new_kochi_ql_mod()
    print(f"Time:{time.time()-t} s.")
