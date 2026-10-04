# Shortest-path run with the UNTOUCHED dev engine (git HEAD 35ecb8a), as scripts/main_ShortPath.py drives it.
import json, os, sys
os.environ.setdefault("MPLBACKEND", "Agg"); sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/head/src")
import numpy as np
import evacrl
assert evacrl.__file__.startswith(HERE), evacrl.__file__
from evacrl.sarsa import SARSA
d = f"{HERE}/kochi2_int"
out = []
for seed in range(int(sys.argv[1])):
    np.random.seed(seed)
    m = SARSA(agentsProfileName=f"{d}/population_1.csv", nodesdbFile=f"{d}/nodes.csv", linksdbFile=f"{d}/edges.csv",
              transLinkdbFile=f"{d}/actionsdb.csv", transNodedbFile=f"{d}/transitionsdb.csv", meanRayleigh=5 * 60)
    m.loadShortestPathDB(f"{d}/nextnode.csv")
    for t in range(int(min(m.pedDB[:, 9])), 120 * 60):
        m.initEvacuationAtTime(); m.stepForward(); m.checkTargetShortestPath()
        if not t % 10:
            m.computePedHistDenVelAtLinks(); m.updateVelocityAllPedestrians()
    out.append(dict(seed=seed, flag_final=int(np.sum(m.pedDB[:, 10] == 1)),
                    node_final=int(np.sum(np.isin(m.pedDB[:, 8], m.evacuationNodes))),
                    nodes_at_start=int(np.sum(np.isin(m.pedProfiles[:, 4], m.evacuationNodes)))))
    print(out[-1], flush=True)
json.dump(out, open(f"{HERE}/head_sp.json", "w"))
