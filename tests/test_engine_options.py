#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The behaviours that differ between the 2021 and the 2024 code, and the shortest-path baseline, on a
five-node synthetic network (see docs/engine-reconciliation.md for what each one does to real results).

            0 ---------- 1 ---------- 2            3 ------ 4
          start         junction   evacuation       (no path to an evacuation node)
            link 0 (100 m)  link 1 (100 m)            link 2 (50 m)

Run: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evacrl.options import ModelOptions  # noqa: E402
from evacrl.sarsa import SARSA  # noqa: E402

NODES = [(0, 0, 0, 0, 1), (1, 100, 0, 0, 1), (2, 200, 0, 1, 1), (3, 0, 500, 0, 1), (4, 50, 500, 0, 1)]
LINKS = [(0, 0, 1, 100, 3), (1, 1, 2, 100, 3), (2, 3, 4, 50, 3)]
ACTIONS = {0: [0], 1: [0, 1], 2: [-1], 3: [2], 4: [2]}
TRANSITIONS = {0: [1], 1: [0, 2], 2: [2], 3: [4], 4: [3]}
NEXTNODE = [(0, 1), (1, 2), (2, 2), (3, -9999), (4, -9999)]


def _row(node, items):
    return [node, len(items)] + items + [0] * (10 - len(items))


def write_case(folder, agents_at, style="2021", links=None):
    """style '2021': '#' headers, integers as integers. style '2024': plain header lines, integers as floats."""
    folder = Path(folder)
    float_ints = style == "2024"

    def num(v):
        return f"{float(v)}" if float_ints else f"{v}"

    def table(name, header, rows):
        lines = [header] if header else []
        lines += [",".join(num(v) for v in row) for row in rows]
        (folder / name).write_text("\n".join(lines) + "\n")

    hash_ = "#" if style == "2021" else ""
    table("nodes.csv", f"{hash_}number,coord_x,coord_y,evacuation,reward", NODES)
    table("links.csv", f"{hash_}number,node1,node2,length,width", links or LINKS)
    table("agents.csv", f"{hash_}age,gender,hhType,hhId,Node", [(0, 0, 0, i, n) for i, n in enumerate(agents_at)])
    table("actions.csv", None, [_row(n, ACTIONS[n]) for n in range(5)])
    table("transitions.csv", None, [_row(n, TRANSITIONS[n]) for n in range(5)])
    nextnode = [f"{hash_}node,next"] if style == "2024" else []
    (folder / "nextnode.csv").write_text("\n".join(nextnode + [f"{a},{b}" for a, b in NEXTNODE]) + "\n")
    return folder


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        np.random.seed(0)

    def tearDown(self):
        self._tmp.cleanup()

    def model(self, agents_at=(0, 1, 3), options=None, style="2021", meanRayleigh=10, links=None):
        d = write_case(self.dir, agents_at, style, links)
        return SARSA(agentsProfileName=str(d / "agents.csv"), nodesdbFile=str(d / "nodes.csv"),
                     linksdbFile=str(d / "links.csv"), transLinkdbFile=str(d / "actions.csv"),
                     transNodedbFile=str(d / "transitions.csv"), meanRayleigh=meanRayleigh, options=options)

    def run_shortest_path(self, m, seconds=900):
        m.loadShortestPathDB(str(self.dir / "nextnode.csv"))
        for t in range(int(min(m.pedDB[:, 9])), seconds):
            m.initEvacuationAtTime()
            m.stepForward()
            m.checkTargetShortestPath()
            if not t % 10:
                m.computePedHistDenVelAtLinks()
                m.updateVelocityAllPedestrians()


class FileFormats(Base):
    def test_2024_format_gives_the_same_model_as_the_2021_format(self):
        a = self.model(style="2021")
        b = self.model(style="2024")
        for name in ("nodesdb", "linksdb", "transLinkdb", "transNodedb", "pedProfiles"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


class Options(Base):
    def test_defaults_are_the_2021_behaviour(self):
        m = self.model()
        self.assertEqual(m.options, ModelOptions.legacy())
        self.assertEqual(m.surviveReward, 100000)

    def test_survive_reward_comes_from_the_options(self):
        self.assertEqual(self.model(options=ModelOptions.kochi2024()).surviveReward, 10000000)

    def test_invalid_option_values_are_rejected(self):
        for bad in (dict(densityLevel="x"), dict(entrySpeed="x"), dict(segmentSizing="x"), dict(segmentIndex="x")):
            with self.assertRaises(ValueError):
                ModelOptions(**bad)


class LinkSegments(Base):
    LINKS = [(0, 0, 1, 105, 3), (1, 1, 2, 1, 3), (2, 3, 4, 50, 3)]  # 52.5 segments of 2 m, half a segment, 25

    def test_ceil_is_the_2021_sizing(self):
        m = self.model(links=self.LINKS)
        np.testing.assert_array_equal(m.popAtLink_HistParam[:, 1], [53, 1, 25])
        np.testing.assert_allclose(m.popAtLink_HistParam[:, 0], [105 / 53, 1.0, 2.0])

    def test_round_is_the_2024_sizing_with_at_least_one_segment(self):
        m = self.model(links=self.LINKS, options=ModelOptions(segmentSizing="round"))
        np.testing.assert_array_equal(m.popAtLink_HistParam[:, 1], [52, 1, 25])

    def test_kochi2024_uses_round(self):
        self.assertEqual(ModelOptions.kochi2024().segmentSizing, "round")

    def test_a_link_without_length_is_reported_by_number(self):
        with self.assertRaisesRegex(ValueError, r"links \[1\] have no length"):
            self.model(links=[(0, 0, 1, 100, 3), (1, 1, 2, 0, 3), (2, 3, 4, 50, 3)])


class DensityLevel(Base):
    def crowd_link0(self, m, n):
        m.pedDB[:, 6] = 0
        m.pedDB[:, 0:2] = (1.0, 0.0)  # all in the first 2 m segment of link 0
        m.populationAtLinks[0, 1] = n
        m.computePedHistDenVelAtLinks()

    def test_segment_level_sees_a_crowd_that_the_link_average_does_not(self):
        n = 25  # 25 people in 2 m x 3 m: 4.2 persons/m2 (level 2); over the whole 100 m x 2 m link: 0.125 (level 0)
        link = self.model(agents_at=[0] * n)
        seg = self.model(agents_at=[0] * n, options=ModelOptions(densityLevel="segment"))
        self.crowd_link0(link, n)
        self.crowd_link0(seg, n)
        self.assertEqual(link.computeDensityLevel(0), 0)
        self.assertEqual(seg.computeDensityLevel(0), 2)
        self.assertEqual(seg.denLvlArrPerLink[0, 0], 2)
        self.assertEqual(int(seg.denLvlArrPerLink[0, 1:].max()), 0)

    def test_empty_link_is_level_zero_in_both(self):
        for opt in (ModelOptions(), ModelOptions(densityLevel="segment")):
            m = self.model(options=opt)
            m.computePedHistDenVelAtLinks()
            self.assertEqual(m.computeDensityLevel(0), 0)


class FarEndOfLink(Base):
    """An agent standing at the end of a link whose straight length equals its stored length."""

    def agent_at_far_end(self, options):
        m = self.model(agents_at=[1], options=options)
        m.speArrPerLink[0, :50] = 1.0   # the 50 real segments of link 0 ...
        m.speArrPerLink[0, 50] = 0.0    # ... and the padding column one past them
        m.pedDB[0, 0:2] = (100.0, 0.0)  # at node 1, 100 m from the link's first node
        m.pedDB[0, 2:4] = (0.0, 0.0)    # walking to node 0
        m.pedDB[0, 6] = 0
        return m

    def test_raw_index_reads_the_padding_and_stops_the_agent(self):
        m = self.agent_at_far_end(ModelOptions())
        m.updateVelocityV2(0)
        self.assertLess(np.linalg.norm(m.pedDB[0, 4:6]), 0.02)

    def test_clamped_index_uses_the_last_segment(self):
        m = self.agent_at_far_end(ModelOptions(segmentIndex="clamped"))
        m.updateVelocityV2(0)
        speed = np.linalg.norm(m.pedDB[0, 4:6])
        self.assertAlmostEqual(speed, 1.0, delta=0.02)
        self.assertLess(m.pedDB[0, 4], 0)  # towards node 0


class EntrySpeed(Base):
    def agent_choosing_link0(self, options):
        m = self.model(agents_at=[1], options=options)
        m.speArrPerLink[0, :] = 0.0
        m.speArrPerLink[0, 0] = 0.5     # first segment (the end at node 0)
        m.speArrPerLink[0, 49] = 1.1    # last segment (the end at node 1, where the agent enters)
        m.pedDB[0, 0:2] = (100.0, 0.0)
        m.pedDB[0, 6] = 1
        m.pedDB[0, 7] = 1               # just arrived at node 1
        m.populationAtLinks[1, 1] = 1
        m.expeStat[0] = [np.array([0, 0, 0])]
        state = m.getStateIndexAtNode(1)
        m.stateMat[state, 11] = 1000.0  # action 0: link 0, towards node 0
        m.updateTarget(0, ifOptChoice=True)
        return m

    def test_first_segment_speed_is_the_2021_behaviour(self):
        m = self.agent_choosing_link0(ModelOptions())
        self.assertEqual(int(m.pedDB[0, 6]), 0)
        self.assertAlmostEqual(np.linalg.norm(m.pedDB[0, 4:6]), 0.5, places=6)

    def test_position_speed_is_that_of_the_segment_where_the_agent_enters(self):
        m = self.agent_choosing_link0(ModelOptions(entrySpeed="position", segmentIndex="clamped"))
        self.assertAlmostEqual(np.linalg.norm(m.pedDB[0, 4:6]), 1.1, delta=0.02)


class CheckTarget(Base):
    def test_an_agent_that_has_not_started_is_ignored(self):
        m = self.model(agents_at=[0])
        m.pedDB[0, 0:4] = 0.0  # on top of its (placeholder) target, like an agent at coordinate (0, 0)
        self.assertIsNone(m.expeStat[0])
        m.checkTarget()        # used to fail: None has no append
        self.assertEqual(m.pedDB[0, 10], 0)


class ShortestPathBaseline(Base):
    def test_agents_that_reach_an_evacuation_node_are_counted(self):
        for entry in ("first_segment", "position"):
            with self.subTest(entrySpeed=entry):
                np.random.seed(0)
                m = self.model(agents_at=[0, 1, 0, 1], options=ModelOptions(entrySpeed=entry, segmentIndex="clamped"))
                self.run_shortest_path(m)
                self.assertEqual(m.getNumberEvacuatedPed(), 4)
                self.assertTrue(np.all(np.isin(m.pedDB[:, 8], m.evacuationNodes)))
                self.assertTrue(np.all(m.populationAtLinks[:, 1] == 0))

    def test_an_agent_without_a_path_stops_instead_of_walking_on(self):
        m = self.model(agents_at=[3, 0])
        self.run_shortest_path(m, seconds=600)
        stuck = m.pedDB[0]
        self.assertEqual(stuck[10], 0)
        self.assertEqual(int(stuck[6]), -1)
        np.testing.assert_array_equal(stuck[4:6], [0, 0])
        position = stuck[0:2].copy()
        self.run_shortest_path(m, seconds=700)  # keeps the same place, no error
        np.testing.assert_array_equal(m.pedDB[0, 0:2], position)
        self.assertEqual(m.getNumberEvacuatedPed(), 1)  # the agent of node 0 got out

    def test_next_node_file_with_or_without_header(self):
        m = self.model()
        m.loadShortestPathDB(str(self.dir / "nextnode.csv"))
        plain = m.shortestPathDB.copy()
        (self.dir / "withheader.csv").write_text("# node,next\n" + (self.dir / "nextnode.csv").read_text())
        m.loadShortestPathDB(str(self.dir / "withheader.csv"))
        np.testing.assert_array_equal(m.shortestPathDB, plain)
        self.assertEqual(plain.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
