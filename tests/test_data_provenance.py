#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provenance tests: the derived input files of the cases can be regenerated from their sources.

* actionsdb.csv / transitionsdb.csv of both cases  <-  nodesdb.csv + linksdb.csv
  (cases/new_kochi/setActionsAndTransitions.py, the script of the root stack)
* cases/kochi/data/agentsdb.csv                    <-  datasets/census + nodesdb.csv
  (pre/SetPopDB.py)
* cases/new_kochi/data/linksdb.csv                 <-  cases/new_kochi/tmp/linksdb0.csv + nodesdb.csv
  (the length-0 -> 2 correction of fixLinksDBAndNodesDB in cases/new_kochi/preProcess.py, then the removal of the longest link
  at a node with more than 10 links, cases/new_kochi/fixExcessLinks.py; needs SciPy for the second part)

If one of these fails, a data file and the script that produces it have drifted apart.

Run: python -m unittest discover tests
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The only places that know where the data and the scripts live.
# Update these (and nothing else) when the repository layout changes.
CASES = REPO / "cases"
ACTIONS_SCRIPT = CASES / "new_kochi" / "setActionsAndTransitions.py"
SETPOP_SCRIPT = REPO / "pre" / "SetPopDB.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regenerate_actions_and_transitions(data_dir, workdir):
    """Run setMatrices() on copies of nodesdb/linksdb; return the two regenerated files' bytes."""
    (Path(workdir) / "data").mkdir(parents=True)
    for name in ("nodesdb", "linksdb"):
        shutil.copy(Path(data_dir) / f"{name}.csv", Path(workdir) / "data" / f"{name}.csv")
    module = _load(ACTIONS_SCRIPT, "setActionsAndTransitions_under_test")
    cwd = os.getcwd()
    os.chdir(workdir)  # the script reads and writes ./data, by design
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.setMatrices()
    finally:
        os.chdir(cwd)
    return {n: (Path(workdir) / "data" / f"{n}.csv").read_bytes() for n in ("actionsdb", "transitionsdb")}


def regenerate_kochi_agents(nodes_csv, workdir):
    """Run SetPopDB.setPopDB() on a case folder holding `nodes_csv`; return agentsdb.csv bytes."""
    case = Path(workdir) / "case"
    (case / "data").mkdir(parents=True)
    shutil.copy(nodes_csv, case / "data" / "nodesdb.csv")
    module = _load(SETPOP_SCRIPT, "SetPopDB_under_test")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a few census areas are empty files; the script skips them
        module.setPopDB(area=str(case))
    return (case / "data" / "agentsdb.csv").read_bytes()


def derive_linksdb(linksdb0_csv, nodes_csv=None):
    """The correction applied by fixLinksDBAndNodesDB (cases/new_kochi/preProcess.py), as bytes; with `nodes_csv`, followed
    by the removal of links at nodes that have more than 10 (fixExcessLinks.py)."""
    links = np.loadtxt(linksdb0_csv, delimiter=",")
    links[:, 3][np.where(links[:, 3] == 0)] = 2
    if nodes_csv is not None:
        from evacrl.casebuild import Network, prune_excess_links
        nodes = np.loadtxt(nodes_csv, delimiter=",", skiprows=1)
        links = prune_excess_links(Network(nodes, links))[0].links
    buf = io.BytesIO()
    np.savetxt(buf, links, delimiter=",", header="number,node1,node2,length,width", fmt="%d,%d,%d,%d,%d")
    return buf.getvalue()


class DataProvenance(unittest.TestCase):
    def test_actions_and_transitions_regenerate(self):
        for case in ("kochi", "new_kochi"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                got = regenerate_actions_and_transitions(CASES / case / "data", tmp)
                for name, data in got.items():
                    self.assertEqual(data, (CASES / case / "data" / f"{name}.csv").read_bytes(),
                                     f"{case}/{name}.csv is not what setActionsAndTransitions.py produces")

    def test_kochi_agents_regenerate_from_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = regenerate_kochi_agents(CASES / "kochi" / "data" / "nodesdb.csv", tmp)
        self.assertEqual(got, (CASES / "kochi" / "data" / "agentsdb.csv").read_bytes(),
                         "kochi/agentsdb.csv is not what SetPopDB.py produces")

    def test_new_kochi_linksdb_derives_from_tmp_linksdb0(self):
        try:
            import scipy  # noqa: F401  (evacrl.casebuild needs it)
        except ImportError:
            self.skipTest("needs SciPy")
        got = derive_linksdb(CASES / "new_kochi" / "tmp" / "linksdb0.csv", CASES / "new_kochi" / "data" / "nodesdb.csv")
        self.assertEqual(got, (CASES / "new_kochi" / "data" / "linksdb.csv").read_bytes(),
                         "new_kochi/linksdb.csv is not tmp/linksdb0.csv with zero lengths corrected and the extra link removed")
        # and without the second step it is what it was: one node with 11 links, which setActionsAndTransitions.py now refuses
        before = derive_linksdb(CASES / "new_kochi" / "tmp" / "linksdb0.csv")
        self.assertEqual(before.count(b"\n") - 1, 21722)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            shutil.copy(CASES / "new_kochi" / "data" / "nodesdb.csv", Path(tmp) / "data" / "nodesdb.csv")
            (Path(tmp) / "data" / "linksdb.csv").write_bytes(before)
            module = _load(ACTIONS_SCRIPT, "setActionsAndTransitions_refuses")
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaisesRegex(ValueError, "node 3106 has 11 links"):
                    module.setMatrices()
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
