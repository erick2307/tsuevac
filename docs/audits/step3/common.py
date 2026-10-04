# -*- coding: utf-8 -*-
"""Shared by the Step 3 audits. Needs the 2024 Urushibara repository (read-only reference):

    git clone https://github.com/erick2307/2024_urushibara ~/2024_Urushibara     # or set URUSHIBARA_DIR=<its EVACMODEL3_FocalPoints>
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
RESULTS = f"{URU}/results_for Usama"
DATA = f"{URU}/kochi_data"
CASES = ("kochi0", "kochi1", "kochi2", "kochi4", "kochi42")   # the ones with actionsdb / nextnode / population
