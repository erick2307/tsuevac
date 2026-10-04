#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of how evacrl.paths finds the repository (a checkout, an installed package, or an override).

Run: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # the only place that knows where the code lives

from evacrl import paths  # noqa: E402


def make_repo(root):
    (root / "cases").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return root


class FindRepoRoot(unittest.TestCase):
    def test_checkout_is_found_from_the_source_file(self):
        self.assertEqual(paths.REPO_ROOT, REPO)

    def test_environment_variable_overrides_everything(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"EVACRL_ROOT": tmp}):
            self.assertEqual(paths._find_repo_root(), Path(tmp).resolve())

    def test_installed_package_uses_the_repository_it_is_run_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            repo = make_repo(tmp / "repo")
            (repo / "notebooks").mkdir()
            site = tmp / "venv" / "lib" / "site-packages" / "evacrl"  # no pyproject.toml above it
            site.mkdir(parents=True)
            cwd = os.getcwd()
            os.chdir(repo / "notebooks")  # run from a subfolder of the repository
            try:
                with mock.patch.object(paths, "__file__", str(site / "paths.py")), \
                        mock.patch.dict(os.environ, clear=False):
                    os.environ.pop("EVACRL_ROOT", None)
                    self.assertEqual(paths._find_repo_root(), repo)
            finally:
                os.chdir(cwd)

    def test_nothing_found_keeps_the_previous_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            site = tmp / "a" / "b" / "evacrl"
            site.mkdir(parents=True)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.object(paths, "__file__", str(site / "paths.py")), \
                        mock.patch.dict(os.environ, clear=False):
                    os.environ.pop("EVACRL_ROOT", None)
                    self.assertEqual(paths._find_repo_root(), site.parents[1])  # parents[2] of .../a/b/evacrl/paths.py
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
