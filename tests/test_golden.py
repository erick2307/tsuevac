#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Golden regression tests for the root stack: Q-learning (main_ql, main_ql_mod), SARSA and Monte Carlo.

Each entry of CASES runs one entry point with a fixed random seed on a short simulation and compares
the survivors and the learned state matrices with recorded results in
tests/fixtures/<entry>_golden_expected.json:

* kochi, kochi_ql, kochi_sarsa, kochi_mc   real 4,315-node road network + a small SYNTHETIC population
             (tests/fixtures/kochi_golden_agents.csv, NOT real Kochi data), 3 short simulations;
             run_ql_mod, run_ql, run_sarsa and run_mc respectively, with the default ModelOptions
* kochi_sarsa_legacy, kochi_mc_legacy   the same runs of SARSA and Monte Carlo with ModelOptions.legacy(); their
             recordings are those of the original code (see below)
* new_kochi  real 19,207-node road network + a deterministic 1-in-1000 sample of its real
             population (cases/new_kochi/data/agentsdb.csv), 2 short simulations; run_ql_mod

The recorded results were checked to be byte-identical to what the ORIGINAL code (before the
repository was reorganised, run with NumPy 1.23) produces with the same seeds and inputs.
Their purpose is to prove that restructuring the repository (moving files, changing how paths
are resolved) or merging the algorithm modules does not change the results.

Run:        python -m unittest discover tests        (or: pytest tests)
Strict:     GOLDEN_STRICT=1 python -m unittest discover tests
            also compares the sha256 of every output file (same NumPy/Python only)
Regenerate: UPDATE_GOLDEN=1 python tests/test_golden.py [entry ...]   (e.g. kochi_sarsa)
            (only after an INTENTIONAL change of behaviour)
"""
import contextlib
import hashlib
import importlib
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

# The only places that know where the case data and the code live.
# Update these (and nothing else) when the repository layout changes.
CODE_DIRS = [REPO / "src", REPO / "scripts"]


def case_data_dir(area):
    return REPO / "cases" / area / "data"


CASE_FILES = ("nodesdb", "linksdb", "actionsdb", "transitionsdb")

# entry point of each method: (module in scripts/, function)
METHODS = {
    "ql_mod": ("main_ql_mod", "run_ql_mod"),
    "ql": ("main_ql", "run_ql"),
    "sarsa": ("main_sarsa", "run_sarsa"),
    "mc": ("main_mc", "run_mc"),
}


def _kochi(method):
    return dict(
        method=method,
        network="kochi",
        seed=20240101,
        run=dict(simtime=10, meandeparture=3, numSim0=0, numBlocks=1, simPerBlock=2, name="golden"),
        agents=lambda: (FIXTURES / "kochi_golden_agents.csv").read_text(),
    )


CASES = {
    "kochi": _kochi("ql_mod"),  # the names of the first two entries are kept: they name their fixture files
    "new_kochi": dict(
        method="ql_mod",
        network="new_kochi",
        seed=1,
        run=dict(simtime=10, meandeparture=3, numSim0=0, numBlocks=1, simPerBlock=1, name="golden"),
        # header + every 1000th agent of the real population (deterministic)
        agents=lambda: _sample_lines((case_data_dir("new_kochi") / "agentsdb.csv").read_text(), 1000),
    ),
    "kochi_ql": _kochi("ql"),
    "kochi_sarsa": _kochi("sarsa"),
    "kochi_mc": _kochi("mc"),
    # the same runs with ModelOptions.legacy(): the recordings made before any option existed, kept to prove
    # that the 2021 behaviour is still reproducible after the defaults changed
    "kochi_sarsa_legacy": dict(_kochi("sarsa"), options="legacy"),
    "kochi_mc_legacy": dict(_kochi("mc"), options="legacy"),
}


def _sample_lines(text, step):
    lines = text.splitlines()
    return "\n".join([lines[0]] + lines[1::step]) + "\n"


def expected_path(name):
    return FIXTURES / f"{name}_golden_expected.json"


def run_golden_case(name, workdir):
    """Run entry `name` in `workdir` and return a compact, comparable summary."""
    cfg = CASES[name]
    case = Path(workdir) / "case"
    (case / "data").mkdir(parents=True)
    for f in CASE_FILES:
        shutil.copy(case_data_dir(cfg["network"]) / f"{f}.csv", case / "data" / f"{f}.csv")
    (case / "data" / "agentsdb.csv").write_text(cfg["agents"]())

    for d in CODE_DIRS:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from evacrl.options import ModelOptions  # noqa: E402
    options = getattr(ModelOptions, cfg["options"])() if cfg.get("options") else None  # None: the defaults
    extra = {} if options is None else {"options": options}
    module, function = METHODS[cfg["method"]]
    run_method = getattr(importlib.import_module(module), function)  # imported late so CODE_DIRS is honoured
    from evacrl import paths  # noqa: E402

    # Resolve the case by NAME through paths.CASES_DIR, exactly as real runs do (area="kochi").
    saved_cases_dir, paths.CASES_DIR = paths.CASES_DIR, Path(workdir)
    try:
        np.random.seed(cfg["seed"])
        with contextlib.redirect_stdout(io.StringIO()):
            run_method(area="case", **cfg["run"], **extra)
    finally:
        paths.CASES_DIR = saved_cases_dir

    run = cfg["run"]
    state_dir = case / f"state_{run['name']}"
    survivors = np.loadtxt(
        state_dir / f"survivorsPerSim_{run['numBlocks']}x{run['simPerBlock']}.csv",
        delimiter=",", dtype=int, ndmin=2,
    )
    summary = {"survivors": survivors.tolist(), "state": {}, "sha256": {}}
    for f in sorted(state_dir.iterdir()):
        summary["sha256"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        if f.name.startswith("sim_"):
            m = np.loadtxt(f, delimiter=",")
            # root stack layout (Q-learning, SARSA and Monte Carlo): [node, 10 density codes, 10 action values, 10 visit counts]
            summary["state"][f.name] = {
                "shape": list(m.shape),
                "q_sum": round(float(m[:, 11:21].sum()), 3),
                "count_sum": int(m[:, 21:31].sum()),
            }
    return summary


class Golden(unittest.TestCase):
    def check(self, name):
        expected = json.loads(expected_path(name).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            got = run_golden_case(name, tmp)

        self.assertEqual(got["survivors"], expected["survivors"], "survivors per simulation changed")
        self.assertEqual(sorted(got["state"]), sorted(expected["state"]), "set of state files changed")
        for fname, exp in expected["state"].items():
            g = got["state"][fname]
            self.assertEqual(g["shape"], exp["shape"], f"{fname}: state-matrix shape changed")
            self.assertEqual(g["count_sum"], exp["count_sum"], f"{fname}: visit counts changed")
            self.assertAlmostEqual(g["q_sum"], exp["q_sum"], places=2, msg=f"{fname}: action values changed")
        if os.environ.get("GOLDEN_STRICT") == "1":
            self.assertEqual(got["sha256"], expected["sha256"], "output files are not byte-identical")

    def test_kochi(self):
        self.check("kochi")

    def test_new_kochi(self):
        self.check("new_kochi")

    def test_kochi_ql(self):
        self.check("kochi_ql")

    def test_kochi_sarsa(self):
        self.check("kochi_sarsa")

    def test_kochi_mc(self):
        self.check("kochi_mc")

    def test_kochi_sarsa_legacy(self):
        self.check("kochi_sarsa_legacy")

    def test_kochi_mc_legacy(self):
        self.check("kochi_mc_legacy")


if __name__ == "__main__":
    if os.environ.get("UPDATE_GOLDEN") == "1":
        for name in sys.argv[1:] or CASES:
            with tempfile.TemporaryDirectory() as tmp:
                result = run_golden_case(name, tmp)
            expected_path(name).write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {expected_path(name)}")
    else:
        unittest.main()
