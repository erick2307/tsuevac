#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The 2021 entry points (scripts/main_*.py) and their case selection: the shortest-path script runs more than one simulation, any folder of
`cases/` can be named on the command line, and the defaults of the Monte Carlo script start a fresh run.

Run: python -m unittest discover tests
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evacrl import cli, paths  # noqa: E402
from test_experiment import write_case  # noqa: E402


class ShortestPathScript(unittest.TestCase):
    def test_every_simulation_of_a_run_loads_the_shortest_path_table(self):
        import main_ShortPath
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = write_case(Path(tmp) / "tiny")
            np.random.seed(0)
            # numSim0=0 runs one simulation, then numBlocks * simPerBlock more, each with a new model: it used to stop at the second
            main_ShortPath.run_shortpath(area=str(folder), simtime=10, meandeparture=1, numSim0=0, numBlocks=1, simPerBlock=2, name="t")
            rows = np.loadtxt(folder / "state_t" / "survivorsPerSim_1x2.csv", delimiter=",", ndmin=2)
        self.assertEqual(rows[:, 0].tolist(), [0, 1, 2])
        self.assertTrue((rows[:, 1] == 10).all(), rows)     # all ten agents reach the shelter in ten minutes

    def test_a_missing_table_is_explained_before_anything_runs(self):
        import main_ShortPath
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_case(Path(tmp) / "tiny", nextnode=False)
            with self.assertRaises(SystemExit) as caught:
                main_ShortPath.run_shortpath(area=str(folder), simtime=1, numBlocks=1, simPerBlock=1)
        self.assertIn("nextnode.csv not found", str(caught.exception))


class CaseSelection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        write_case(root / "cases" / "mine")
        (root / "cases" / "empty").mkdir()
        self.patch = mock.patch.object(paths, "CASES_DIR", root / "cases")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self._tmp.cleanup()

    def test_a_folder_of_cases_without_a_runner_goes_to_the_generic_one(self):
        seen = []
        cli.run_case({"kochi": lambda: seen.append("kochi")}, "kochi", argv=["mine"], generic=seen.append)
        self.assertEqual(seen, ["mine"])

    def test_a_named_runner_wins_over_the_generic_one(self):
        seen = []
        (paths.CASES_DIR / "kochi" / "data").mkdir(parents=True)
        cli.run_case({"kochi": lambda: seen.append("runner")}, "kochi", argv=["kochi"], generic=seen.append)
        self.assertEqual(seen, ["runner"])

    def test_an_unknown_name_is_still_an_error_and_says_what_is_accepted(self):
        for argv in (["nosuch"], ["empty"]):                 # not a folder at all; a folder without data/
            with self.assertRaises(SystemExit) as caught:
                cli.run_case({"kochi": lambda: None}, "kochi", argv=argv, generic=lambda name: None)
            self.assertIn("unknown case", str(caught.exception))
            self.assertIn("folder of cases/", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            cli.run_case({"kochi": lambda: None}, "kochi", argv=["mine"])    # no generic runner: as before
        self.assertIn("unknown case 'mine'", str(caught.exception))
        self.assertNotIn("folder of cases/", str(caught.exception))


class Defaults(unittest.TestCase):
    def test_the_monte_carlo_script_starts_a_fresh_run_on_any_case(self):
        import main_mc
        with mock.patch.object(main_mc, "run_mc") as run:
            main_mc.kochi_mc(area="kochi_area2")
        kwargs = run.call_args.kwargs
        self.assertEqual((kwargs["area"], kwargs["numSim0"]), ("kochi_area2", 0))     # it resumed simulation 1950 of a run that is not in the repository
        self.assertLessEqual(kwargs["numBlocks"] * kwargs["simPerBlock"], 1000)

    def test_the_learning_scripts_take_the_case_as_an_argument(self):
        import main_ql
        import main_ql_mod
        import main_sarsa
        for module, helper, runner in ((main_ql, "kochi_ql", "run_ql"), (main_ql_mod, "kochi_ql_mod", "run_ql_mod"), (main_sarsa, "kochi_sarsa", "run_sarsa")):
            with self.subTest(script=module.__name__), mock.patch.object(module, runner) as run:
                getattr(module, helper)()
                self.assertEqual(run.call_args.kwargs["area"], "kochi")
                getattr(module, helper)(area="kochi_area0")
                self.assertEqual(run.call_args.kwargs["area"], "kochi_area0")


if __name__ == "__main__":
    unittest.main()
