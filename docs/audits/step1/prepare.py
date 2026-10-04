# -*- coding: utf-8 -*-
"""Prepare what head_sp.py needs: the engine as it was before Step 1, and an integer-formatted copy of kochi2.

    python prepare.py [git-ref-of-the-engine-before-step-1]      (default: 35ecb8a, the head of `dev` before Step 1)

head/src/           the `evacrl` package at that commit (the 2021 loaders read integers only, and skip the first line of
                    nextnode.csv, so the Urushibara tables are rewritten into kochi2_int/ for it)
kochi2_int/         kochi2 of EVACMODEL3_FocalPoints/results_for Usama with integer-formatted tables
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
URU = os.environ.get("URUSHIBARA_DIR", os.path.expanduser("~/2024_Urushibara/EVACMODEL3_FocalPoints"))
ref = sys.argv[1] if len(sys.argv) > 1 else "35ecb8a"

shutil.rmtree(f"{HERE}/head", ignore_errors=True)
os.makedirs(f"{HERE}/head")
archive = subprocess.run(["git", "-C", REPO, "archive", ref, "src"], check=True, capture_output=True).stdout
subprocess.run(["tar", "-x", "-C", f"{HERE}/head"], input=archive, check=True)

src = f"{URU}/results_for Usama/kochi2"
dst = f"{HERE}/kochi2_int"
os.makedirs(dst, exist_ok=True)


def ints(line):
    return ",".join(str(int(float(v))) for v in line.strip().split(","))


for name in ("edges.csv", "population_1.csv"):
    lines = [l.rstrip("\n") if l.startswith("#") else ints(l) for l in open(f"{src}/{name}")]
    open(f"{dst}/{name}", "w").write("\n".join(lines) + "\n")
for name in ("nodes.csv", "actionsdb.csv", "transitionsdb.csv"):
    shutil.copy(f"{src}/{name}", f"{dst}/{name}")
open(f"{dst}/nextnode.csv", "w").write("# node,next\n" + open(f"{src}/nextnode.csv").read())
print("prepared", f"{HERE}/head/src", dst)
