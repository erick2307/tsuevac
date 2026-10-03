#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Golden regression test for the root Q-learning stack (main_ql_mod.run_ql_mod).

Runs three short simulations on the real Kochi road network (kochi/data) with a small
synthetic population (tests/fixtures/kochi_golden_agents.csv, NOT real Kochi data) and a
fixed random seed, then compares the survivors and the learned state matrices with the
recorded values in tests/fixtures/kochi_golden_expected.json.

Its purpose is to prove that restructuring the repository (moving files, changing how
paths are resolved) does not change the results.

Run:        python -m unittest discover tests        (or: pytest tests)
Regenerate: UPDATE_GOLDEN=1 python tests/test_golden_ql.py
            (only after an INTENTIONAL change of behaviour)
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no display needed

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXPECTED = FIXTURES / "kochi_golden_expected.json"

# The only places that know where the case data and the code live.
# Update these (and nothing else) when the repository layout changes.
CASE_DATA = REPO / "kochi" / "data"
CODE_DIRS = [REPO]

SEED = 20240101
RUN = dict(simtime=10, meandeparture=3, numSim0=0, numBlocks=1, simPerBlock=2, name="golden")
CASE_FILES = ("nodesdb", "linksdb", "actionsdb", "transitionsdb")


def run_golden_case(workdir):
    """Run the simulations in `workdir` and return a compact, comparable summary."""
    case = Path(workdir) / "case"
    (case / "data").mkdir(parents=True)
    for name in CASE_FILES:
        shutil.copy(CASE_DATA / f"{name}.csv", case / "data" / f"{name}.csv")
    shutil.copy(FIXTURES / "kochi_golden_agents.csv", case / "data" / "agentsdb.csv")

    for d in CODE_DIRS:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import main_ql_mod  # noqa: E402  (imported late so CODE_DIRS is honoured)

    np.random.seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        main_ql_mod.run_ql_mod(area=str(case), **RUN)

    state_dir = case / f"state_{RUN['name']}"
    survivors = np.loadtxt(
        state_dir / f"survivorsPerSim_{RUN['numBlocks']}x{RUN['simPerBlock']}.csv",
        delimiter=",", dtype=int,
    )
    summary = {"survivors": survivors.tolist(), "state": {}}
    for f in sorted(state_dir.glob("sim_*.csv")):
        m = np.loadtxt(f, delimiter=",")
        # root stack layout: [node, 10 density codes, 10 action values, 10 visit counts]
        summary["state"][f.name] = {
            "shape": list(m.shape),
            "q_sum": round(float(m[:, 11:21].sum()), 3),
            "count_sum": int(m[:, 21:31].sum()),
        }
    return summary


class GoldenQLearning(unittest.TestCase):
    def test_matches_recorded_results(self):
        expected = json.loads(EXPECTED.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            got = run_golden_case(tmp)

        self.assertEqual(got["survivors"], expected["survivors"], "survivors per simulation changed")
        self.assertEqual(sorted(got["state"]), sorted(expected["state"]), "set of state files changed")
        for name, exp in expected["state"].items():
            g = got["state"][name]
            self.assertEqual(g["shape"], exp["shape"], f"{name}: state-matrix shape changed")
            self.assertEqual(g["count_sum"], exp["count_sum"], f"{name}: visit counts changed")
            self.assertAlmostEqual(g["q_sum"], exp["q_sum"], places=2, msg=f"{name}: action values changed")


if __name__ == "__main__":
    if os.environ.get("UPDATE_GOLDEN") == "1":
        with tempfile.TemporaryDirectory() as tmp:
            result = run_golden_case(tmp)
        EXPECTED.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {EXPECTED}")
    else:
        unittest.main()
