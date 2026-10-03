#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of the case selection of the entry-point scripts (evacrl.cli and scripts/main_*.py).

Run: python -m unittest discover tests
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[1]

# The only places that know where the code lives (update these when the layout changes).
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

sys.path.insert(0, str(SRC))
from evacrl import cli, paths  # noqa: E402


class RunCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        for name in ("alpha", "beta"):
            (root / name / "data").mkdir(parents=True)
        (root / "no_data").mkdir()
        self._saved, paths.CASES_DIR = paths.CASES_DIR, root
        self.calls = []
        self.runners = {n: (lambda n=n: self.calls.append(n)) for n in ("alpha", "beta", "gamma", "no_data")}

    def tearDown(self):
        paths.CASES_DIR = self._saved
        self._tmp.cleanup()

    def test_named_case_runs(self):
        cli.run_case(self.runners, default="alpha", argv=["beta"])
        self.assertEqual(self.calls, ["beta"])

    def test_default_runs_without_argument(self):
        cli.run_case(self.runners, default="alpha", argv=[])
        self.assertEqual(self.calls, ["alpha"])

    def test_unknown_case_is_reported(self):
        with self.assertRaises(SystemExit) as cm:
            cli.run_case(self.runners, default="alpha", argv=["nope"])
        self.assertIn("unknown case 'nope'", str(cm.exception))
        self.assertEqual(self.calls, [])

    def test_case_without_data_lists_what_is_available(self):
        for name in ("gamma", "no_data"):  # not in the folder / folder without data/
            with self.subTest(case=name), self.assertRaises(SystemExit) as cm:
                cli.run_case(self.runners, default="alpha", argv=[name])
            msg = str(cm.exception)
            self.assertIn("alpha, beta", msg)
            self.assertNotIn("no_data,", msg)
        self.assertEqual(self.calls, [])

    def test_missing_default_says_it_is_the_default(self):
        with self.assertRaises(SystemExit) as cm:
            cli.run_case(self.runners, default="gamma", argv=[])
        self.assertIn("default of", str(cm.exception))

    def test_available_cases_only_counts_folders_with_data(self):
        self.assertEqual(cli.available_cases(), ["alpha", "beta"])


class Scripts(unittest.TestCase):
    """The real scripts, run as a user would, from an unrelated working directory."""

    def run_script(self, script, *args):
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=cwd,
                                  capture_output=True, text=True, timeout=120)

    def test_unknown_case(self):
        r = self.run_script("main_ql_mod.py", "nosuchcase")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown case 'nosuchcase'", r.stderr)

    def test_case_not_in_this_repository_lists_available_cases(self):
        # arahama is the default of three scripts and is not part of the repository
        for script in ("main_ql.py", "main_ql_mod.py", "main_ShortPath.py"):
            with self.subTest(script=script):
                r = self.run_script(script)  # no argument: the default case
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("default of", r.stderr)
                self.assertIn("kochi", r.stderr)
                self.assertIn("new_kochi", r.stderr)

    def test_shortest_path_baseline_explains_the_missing_nextnode_file(self):
        r = self.run_script("main_ShortPath.py", "kochi")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nextnode.csv not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
