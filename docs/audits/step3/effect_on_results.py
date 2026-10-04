# -*- coding: utf-8 -*-
"""A5. What the clean-up and the population method do to a result: shortest-path evacuation of area 2 (kochi2), 120 min.

    python effect_on_results.py [SEEDS]        (default 30; about 6 min on 4 cores; needs the 'casebuild' extra)

variant 1  the 2024 study's tables and its population file (622 agents, 14 of them starting at a shelter)
variant 2  rebuilt with the clusters clean-up, the same recipe for the population (622 agents, uniform, none at a shelter)
variant 3  rebuilt, the census total by area-weighting the boundary cells (1,704 agents), placed uniformly
variant 4  rebuilt, the same 1,704 agents placed by the census (each cell's people on the nodes inside it)
The seed of a run fixes the departure times; the population of a variant is the same in all its runs.
"""
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from common import DATA, RESULTS

from evacrl.casebuild import cli
from evacrl.options import ModelOptions
from evacrl.sarsa import SARSA

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
T = 120 * 60
WORK = tempfile.mkdtemp(prefix="step3_effect_")
GRAPH = f"{RESULTS}/kochi2/Graph"
COMMON = ["--areas", f"{DATA}/kochi-shi_tsunamievac_areas_crs4326.geojson", "--index", "2", "--shelters",
          f"{DATA}/kochi_tsunami_evacbldg_crs4326.geojson", f"{DATA}/kochi_tsunami_shelters_crs4326.geojson",
          "--census", f"{DATA}/kochi-shi_census_crs4326.geojson"]


def build(name, *extra):
    case = os.path.join(WORK, name)
    assert cli.main(["from-snapshot", GRAPH, case] + COMMON + list(extra)) == 0
    d = os.path.join(case, "data")
    return dict(agents=f"{d}/agentsdb.csv", nodes=f"{d}/nodesdb.csv", links=f"{d}/linksdb.csv", actions=f"{d}/actionsdb.csv",
                transitions=f"{d}/transitionsdb.csv", nextnode=f"{d}/nextnode.csv")


def variants():
    r = f"{RESULTS}/kochi2"
    return {
        "1  2024 tables": dict(agents=f"{r}/population_1.csv", nodes=f"{r}/nodes.csv", links=f"{r}/edges.csv", actions=f"{r}/actionsdb.csv",
                               transitions=f"{r}/transitionsdb.csv", nextnode=f"{r}/nextnode.csv"),
        "2  rebuilt, 622 agents": build("v2", "--census-method", "within", "--strategy", "uniform", "--seed", "0"),
        "3  rebuilt, 1704 uniform": build("v3", "--census-method", "weighted", "--strategy", "uniform", "--seed", "0"),
        "4  rebuilt, 1704 by census": build("v4", "--census-method", "weighted", "--strategy", "proportional"),
    }


def run(job):
    name, files, seed = job
    np.random.seed(seed)
    m = SARSA(agentsProfileName=files["agents"], nodesdbFile=files["nodes"], linksdbFile=files["links"],
              transLinkdbFile=files["actions"], transNodedbFile=files["transitions"], meanRayleigh=5 * 60, options=ModelOptions())
    m.loadShortestPathDB(files["nextnode"])
    t0 = int(min(m.pedDB[:, 9]))
    curve = []
    for t in range(t0, T):
        m.initEvacuationAtTime()
        m.stepForward()
        m.checkTargetShortestPath()
        if not t % 10:
            m.computePedHistDenVelAtLinks()
            m.updateVelocityAllPedestrians()
        curve.append(m.getNumberEvacuatedPed())
    curve = np.array(curve)
    last = t0 + int(np.argmax(curve >= curve[-1]))
    return name, dict(agents=m.numPedestrian, end=int(curve[-1]), last=last, at30=int(curve[1800 - t0 - 1]))


if __name__ == "__main__":
    spec = variants()
    jobs = [(n, f, s) for s in range(SEEDS) for n, f in spec.items()]
    with Pool(4) as pool:
        out = pool.map(run, jobs, chunksize=1)
    print(f"shortest path on kochi2, {SEEDS} departure-time seeds, mean departure 5 min, {T // 60} min simulated, default ModelOptions\n")
    print(f"{'variant':28s} {'agents':>6s} {'safe at the end':>16s} {'last evacuee, s: mean ± sd':>27s} {'median':>7s} {'safe at 30 min':>15s} {'= % of agents':>14s}")
    for name in spec:
        rows = [r for n, r in out if n == name]
        last = np.array([r["last"] for r in rows])
        at30 = np.array([r["at30"] for r in rows])
        n = rows[0]["agents"]
        print(f"{name:28s} {n:6d} {np.mean([r['end'] for r in rows]):16.1f} {last.mean():20.0f} ± {last.std(ddof=1):4.0f} {np.median(last):7.0f} "
              f"{at30.mean():15.1f} {100 * at30.mean() / n:13.1f}%")
