# -*- coding: utf-8 -*-
"""Shared by the Step 4 audits. `kochi2` (the study's area 2, 622 agents) needs the 2024 Urushibara repository as a read-only
reference (git clone https://github.com/erick2307/2024_urushibara ~/2024_Urushibara, or set URUSHIBARA_DIR to its
EVACMODEL3_FocalPoints folder); any other name is a case folder of this repository (cases/<name>/data).
"""
import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
URU = os.environ.get("URUSHIBARA_DIR", os.path.expanduser("~/2024_Urushibara/EVACMODEL3_FocalPoints"))


def case_files(name):
    """Paths of the six tables of a case, as the keyword arguments the model wants."""
    if name == "kochi2":
        d = f"{URU}/results_for Usama/kochi2"
        return dict(agentsProfileName=f"{d}/population_1.csv", nodesdbFile=f"{d}/nodes.csv", linksdbFile=f"{d}/edges.csv",
                    transLinkdbFile=f"{d}/actionsdb.csv", transNodedbFile=f"{d}/transitionsdb.csv"), f"{d}/nextnode.csv"
    d = os.path.join(REPO, "cases", name, "data")
    return dict(agentsProfileName=f"{d}/agentsdb.csv", nodesdbFile=f"{d}/nodesdb.csv", linksdbFile=f"{d}/linksdb.csv",
                transLinkdbFile=f"{d}/actionsdb.csv", transNodedbFile=f"{d}/transitionsdb.csv"), f"{d}/nextnode.csv"
