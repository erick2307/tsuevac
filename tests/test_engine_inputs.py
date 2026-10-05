#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The engine's inputs and edges found by the final review: what a shelter's state does with the "link" -1, a discount outside (0, 1],
tables that would fail later with a cryptic error, and the engine without OpenCV.

Run: python -m unittest discover tests
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evacrl.experiment import Case, make_model  # noqa: E402
from evacrl.options import ModelOptions  # noqa: E402
from test_experiment import write_case  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def model(self, folder="tiny", **kw):
        return make_model(Case.load(self.dir / folder), "qlearning", kw.pop("options", None), 1.0, kw.pop("discount", None))


class ShelterState(Base):
    def test_the_state_of_a_shelter_does_not_depend_on_the_last_link_of_the_table(self):
        write_case(self.dir / "tiny")
        for level in ("link", "segment"):
            with self.subTest(densityLevel=level):
                model = self.model(options=ModelOptions(densityLevel=level))
                rows = model.stateMat.shape[0]
                model.populationAtLinks[-1, 1] = 150             # the last link is packed with people
                model.denLvlArrPerLink[-1, :] = 2
                row = model.getStateIndexAtNode(3)               # node 3 is the shelter: its only action is the link -1, to stay
                self.assertEqual(model.stateMat.shape[0], rows)  # the state it already had (every node starts with its empty state), not a new one
                self.assertEqual(row, 3)
                self.assertEqual(model.stateMat[row, 1:11].tolist(), [0] * 10)

    def test_a_real_link_still_has_its_density(self):
        write_case(self.dir / "tiny")
        model = self.model()
        model.populationAtLinks[0, 1] = 150                      # link 0 is 100 m long and 2 m wide: 0.75 people per m2
        self.assertEqual(model.computeDensityLevel(0), 1)
        self.assertEqual(model.computeDensityLevel(-1), 0)


class Discount(Base):
    def test_a_discount_outside_zero_one_is_refused_by_the_model_too(self):
        write_case(self.dir / "tiny")
        for bad in (0, -0.5, 1.5, float("nan")):
            with self.subTest(discount=bad), self.assertRaisesRegex(ValueError, "discount must be in"):
                self.model(discount=bad)
        self.model(discount=1.0)
        self.model(discount=0.9)
        self.assertEqual(self.model().discount, ModelOptions().discount)


class Tables(Base):
    def edit(self, name, change):
        path = self.dir / "tiny" / "data" / name
        table = np.loadtxt(path, delimiter=",", ndmin=2)
        change(table)
        np.savetxt(path, table, delimiter=",", fmt="%d")

    def test_no_agents(self):
        write_case(self.dir / "tiny", starts=[])
        with self.assertRaisesRegex(ValueError, "no rows|no agents"):
            self.model()

    def test_a_node_with_more_links_than_the_state_matrix_holds(self):
        write_case(self.dir / "tiny")
        self.edit("actionsdb.csv", lambda t: t.__setitem__((1, 1), 11))
        self.edit("transitionsdb.csv", lambda t: t.__setitem__((1, 1), 11))
        with self.assertRaisesRegex(ValueError, "node 1 has 11 links"):
            self.model()

    def test_an_agent_that_starts_on_a_node_without_links(self):
        write_case(self.dir / "tiny", starts=[0, 0, 2, 2])
        self.edit("transitionsdb.csv", lambda t: t.__setitem__((2, 1), 0))
        with self.assertRaisesRegex(ValueError, "agent 2 .*starts at node 2, which has no links"):
            self.model()

    def test_an_agent_that_starts_at_a_shelter_is_fine(self):
        write_case(self.dir / "tiny", starts=[0, 3])
        self.model()
        self.edit("transitionsdb.csv", lambda t: t.__setitem__((3, 1), 0))                # even if the table lists no action for the shelter
        self.model()


class WithoutOpenCV(unittest.TestCase):
    def test_the_engine_and_the_experiment_layer_do_not_import_opencv(self):
        code = "import sys; sys.modules['cv2'] = None; import evacrl.core, evacrl.experiment, evacrl.experiment.cli, evacrl.casebuild; print('ok')"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO / "src")), timeout=120)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ok"), r.stderr[-400:])


if __name__ == "__main__":
    unittest.main()
