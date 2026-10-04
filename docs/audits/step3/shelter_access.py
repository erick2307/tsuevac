# -*- coding: utf-8 -*-
"""A7 (S2). Shelters attached to the network by an access link, against shelters snapped onto the nearest node.

    python shelter_access.py [SEEDS]        (default 20; about 10 min on 4 cores; needs the 'casebuild' extra and the 2024 data)

For each area the same inputs are built twice, `--shelters-as snap` (the 2024 study) and `--shelters-as attach`, with one agent at
every start node (so that both variants have the same people on the same street nodes, up to the few nodes that the clean-up
merges differently), and then:

  1. the structure: shelters, access links, nodes;
  2. the walking distance of every start node to the nearest shelter, along the network;
  3. shortest-path evacuation (default ModelOptions, 120 min, departure-time seeds paired between the two variants).
"""
import json
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from common import DATA, RESULTS

from evacrl.casebuild import cli, distance_to_shelter, read_tables
from evacrl.casebuild.network import Network
from evacrl.options import ModelOptions
from evacrl.sarsa import SARSA

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
AREAS = (0, 1, 2, 4)
T = 120 * 60
WORK = tempfile.mkdtemp(prefix="step3_s2_")


def build(area, mode):
    case = os.path.join(WORK, f"a{area}_{mode}")
    argv = ["from-snapshot", f"{RESULTS}/kochi{area}/Graph", case, "--areas", f"{DATA}/kochi-shi_tsunamievac_areas_crs4326.geojson",
            "--index", str(area), "--shelters", f"{DATA}/kochi_tsunami_evacbldg_crs4326.geojson",
            f"{DATA}/kochi_tsunami_shelters_crs4326.geojson", "--strategy", "per_node", "--shelters-as", mode]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(argv) == 0
    return case


def files(case):
    d = os.path.join(case, "data")
    return dict(agents=f"{d}/agentsdb.csv", nodes=f"{d}/nodesdb.csv", links=f"{d}/linksdb.csv", actions=f"{d}/actionsdb.csv",
                transitions=f"{d}/transitionsdb.csv", nextnode=f"{d}/nextnode.csv")


def run(job):
    name, f, seed = job
    np.random.seed(seed)
    m = SARSA(agentsProfileName=f["agents"], nodesdbFile=f["nodes"], linksdbFile=f["links"], transLinkdbFile=f["actions"],
              transNodedbFile=f["transitions"], meanRayleigh=5 * 60, options=ModelOptions())
    m.loadShortestPathDB(f["nextnode"])
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
    return name, dict(agents=m.numPedestrian, end=int(curve[-1]), last=t0 + int(np.argmax(curve >= curve[-1])), at30=int(curve[1800 - t0 - 1]))


if __name__ == "__main__":
    cases = {(a, mode): build(a, mode) for a in AREAS for mode in ("snap", "attach")}
    print("1  structure\n")
    print(f"{'area':5s} {'mode':7s} {'nodes':>5s} {'links':>5s} {'shelters':>8s} {'shelter points':>14s} {'access link m: median / max':>28s} {'> 100 m':>8s} {'start nodes':>11s}")
    for (a, mode), case in cases.items():
        p = json.load(open(os.path.join(case, "provenance.json")))
        s = p["shelters"]
        t = read_tables(os.path.join(case, "data"))
        med = (f"{s['access_length_median']} / {s['access_length_max']}" if mode == "attach" and "access_length_max" in s
               else f"{s['snap_distance_median']} / {s['snap_distance_max']} (snapped)" if "snap_distance_max" in s else "-")
        over = s.get("access_over_100_m", s.get("snapped_over_100_m", 0))
        print(f"{a:<5d} {mode:7s} {len(t['nodes']):5d} {len(t['links']):5d} {int(t['nodes'][:, 3].sum()):8d} {s['points_used']:14d} {med:>28s} {over:8d} {len(t['agents']):11d}")

    print("\n2  walking distance from a start node to the nearest shelter along the network (m)\n")
    print(f"{'area':5s} {'mode':7s} {'mean':>7s} {'median':>7s} {'p95':>7s} {'max':>7s} {'unreachable':>12s}")
    for (a, mode), case in cases.items():
        t = read_tables(os.path.join(case, "data"))
        net = Network(t["nodes"], t["links"])
        d = distance_to_shelter(net)
        d = d[net.nodes[:, 3] == 0]                      # every street node is a start node (one agent each)
        ok = np.isfinite(d)
        print(f"{a:<5d} {mode:7s} {d[ok].mean():7.0f} {np.median(d[ok]):7.0f} {np.percentile(d[ok], 95):7.0f} {d[ok].max():7.0f} {int((~ok).sum()):12d}")

    spec = {(a, mode): files(case) for (a, mode), case in cases.items()}
    jobs = [((a, mode), f, s) for s in range(SEEDS) for (a, mode), f in spec.items()]
    with Pool(4) as pool:
        out = pool.map(run, jobs, chunksize=1)
    print(f"\n3  shortest-path evacuation, {SEEDS} departure-time seeds paired between the variants, mean departure 5 min, {T // 60} min simulated\n")
    print(f"{'area':5s} {'mode':7s} {'agents':>6s} {'last evacuee, s':>16s} {'safe at 30 min':>15s} {'= %':>6s} {'safe at the end':>16s}")
    for a in AREAS:
        for mode in ("snap", "attach"):
            rows = [r for n, r in out if n == (a, mode)]
            last = np.array([r["last"] for r in rows]); at30 = np.array([r["at30"] for r in rows]); n = rows[0]["agents"]
            print(f"{a:<5d} {mode:7s} {n:6d} {last.mean():9.0f} ± {last.std(ddof=1):4.0f} {at30.mean():15.1f} {100 * at30.mean() / n:5.1f}% {np.mean([r['end'] for r in rows]):16.1f}")
