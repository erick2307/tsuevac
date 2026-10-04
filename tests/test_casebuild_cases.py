#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The cases built by evacrl.casebuild (cases/kochi_area*): they are valid, they can be rebuilt offline from their own
raw/ folder with the settings in provenance.json, and the model runs on them.

If the rebuild fails, the tables and the code that makes them have drifted apart: rebuild the case
(`python -m evacrl.casebuild from-raw cases/<name>/raw cases/<name> ...`, see cases/README.md) or fix the code.

Run: python -m unittest discover tests
"""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evacrl.casebuild import (POPULATION_FILE, PopulationSpec, build_tables, read_node_population, read_raw, read_tables,  # noqa: E402
                              validate_case)

CASES = sorted(p for p in (REPO / "cases").glob("*/provenance.json"))


class ShippedCases(unittest.TestCase):
    def test_there_are_cases_to_check(self):
        self.assertGreaterEqual(len(CASES), 4)

    def test_every_case_is_valid_and_has_nothing_to_warn_about(self):
        for provenance in CASES:
            with self.subTest(case=provenance.parent.name):
                report = validate_case(provenance.parent / "data")
                self.assertTrue(report.ok, str(report))
                self.assertEqual(report.warnings, [], str(report))
                self.assertEqual(report.stats["components"], 1)

    def test_the_raw_files_are_the_ones_the_case_was_made_from(self):
        for provenance in CASES:
            info = json.loads(provenance.read_text(encoding="utf-8"))
            for name, digest in info["raw_sha256"].items():
                with self.subTest(case=provenance.parent.name, file=name):
                    self.assertEqual(hashlib.sha256((provenance.parent / "raw" / name).read_bytes()).hexdigest(), digest)

    def test_every_case_is_rebuilt_from_its_raw_folder_to_the_same_tables(self):
        for provenance in CASES:
            case = provenance.parent
            with self.subTest(case=case.name):
                info = json.loads(provenance.read_text(encoding="utf-8"))
                s = info["settings"]
                pop = s["population"]
                people = read_node_population(case / POPULATION_FILE) if pop["strategy"] == "proportional" else None
                spec = PopulationSpec(strategy=pop["strategy"], total=pop["total"], per_node=pop["per_node"],
                                      weights=None if people is None else (lambda network: people),   # the census is not needed offline
                                      exclude_shelters=pop["exclude_shelters"], seed=pop["seed"])
                built = build_tables(read_raw(case / "raw"), merge=s["merge"], threshold=s["threshold"], parallel=s["parallel"],
                                     routing=s["routing"], population=spec, excess=s["excess"])
                shipped = read_tables(case / "data")
                np.testing.assert_allclose(built["network"].nodes, shipped["nodes"], atol=1e-9 + 5e-7)  # the file keeps 6 decimals
                np.testing.assert_array_equal(built["network"].links, shipped["links"])
                np.testing.assert_array_equal(built["actions"], shipped["actions"])
                np.testing.assert_array_equal(built["transitions"], shipped["transitions"])
                np.testing.assert_array_equal(built["nextnode"], shipped["nextnode"])
                if info["versions"]["numpy"] == np.__version__:
                    np.testing.assert_array_equal(built["agents"], shipped["agents"])   # the same draw with the same NumPy
                else:
                    self.assertEqual(built["agents"].shape, shipped["agents"].shape)

    def test_the_people_file_is_the_one_recorded_and_holds_the_census_total(self):
        found = 0
        for provenance in CASES:
            info = json.loads(provenance.read_text(encoding="utf-8"))
            if info["settings"]["population"]["strategy"] != "proportional":
                continue
            found += 1
            with self.subTest(case=provenance.parent.name):
                file = provenance.parent / POPULATION_FILE
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), info["population_weights"]["sha256"])
                people = read_node_population(file, num_nodes=len(read_tables(provenance.parent / "data")["nodes"]))
                self.assertAlmostEqual(float(people.sum()), info["census"]["area_total"], delta=0.01)   # nobody lost to a shelter
                self.assertAlmostEqual(float(people.sum()), info["population_weights"]["people"], delta=0.001)
                self.assertEqual(info["census"]["method"], "weighted")
        self.assertGreaterEqual(found, 4)

    def test_the_agents_are_as_many_as_the_census_says_and_none_starts_at_a_shelter(self):
        for provenance in CASES:
            info = json.loads(provenance.read_text(encoding="utf-8"))
            t = read_tables(provenance.parent / "data")
            with self.subTest(case=provenance.parent.name):
                self.assertEqual(len(t["agents"]), info["census"]["agents"])
                self.assertEqual(float(np.mean(t["nodes"][t["agents"][:, 4], 3] == 1)), 0.0)

    def test_attached_shelters_are_leaves_with_the_access_link_of_the_recorded_length(self):
        checked = 0
        for provenance in CASES:
            info = json.loads(provenance.read_text(encoding="utf-8"))
            if info.get("shelters", {}).get("mode") != "attach":
                continue
            t = read_tables(provenance.parent / "data")
            nodes, links = t["nodes"], t["links"]
            shelters = np.where(nodes[:, 3] == 1)[0]
            lengths = []
            with self.subTest(case=provenance.parent.name):
                self.assertEqual(len(shelters), info["shelters"]["shelter_nodes"])
                for shelter in shelters:
                    own = links[(links[:, 1] == shelter) | (links[:, 2] == shelter)]
                    self.assertEqual(len(own), 1, f"shelter node {shelter} is not at the end of exactly one link")
                    lengths.append(own[0, 3])
                    # the link is as long as the straight-line distance between its ends (whole metres, floored), up to the
                    # shift of a street node by the clean-up (which moves a merged node to the mean of its members)
                    a, b = (nodes[int(own[0, 1]), 1:3], nodes[int(own[0, 2]), 1:3])
                    self.assertAlmostEqual(own[0, 3], np.hypot(*(a - b)), delta=6)
                if len(lengths) and "access_length_max" in info["shelters"]:
                    self.assertLessEqual(max(lengths), info["shelters"]["access_length_max"] + 6)
                checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_the_model_runs_on_a_case_with_its_shortest_path_table(self):
        from evacrl.sarsa import SARSA
        case = REPO / "cases" / "kochi_area2" / "data"
        np.random.seed(0)
        m = SARSA(agentsProfileName=str(case / "agentsdb.csv"), nodesdbFile=str(case / "nodesdb.csv"),
                  linksdbFile=str(case / "linksdb.csv"), transLinkdbFile=str(case / "actionsdb.csv"),
                  transNodedbFile=str(case / "transitionsdb.csv"), meanRayleigh=5 * 60)
        m.loadShortestPathDB(str(case / "nextnode.csv"))
        for t in range(int(min(m.pedDB[:, 9])), 40 * 60):
            m.initEvacuationAtTime()
            m.stepForward()
            m.checkTargetShortestPath()
            if not t % 10:
                m.computePedHistDenVelAtLinks()
                m.updateVelocityAllPedestrians()
        self.assertEqual(m.numPedestrian, 1704)
        self.assertGreater(m.getNumberEvacuatedPed(), 1400)   # about 84% are safe after 30 min; 40 min is enough for most

    def test_the_model_learns_on_a_case(self):
        from evacrl.qlearn import QLearning
        case = REPO / "cases" / "kochi_area2" / "data"
        np.random.seed(1)
        m = QLearning(agentsProfileName=str(case / "agentsdb.csv"), nodesdbFile=str(case / "nodesdb.csv"),
                      linksdbFile=str(case / "linksdb.csv"), transLinkdbFile=str(case / "actionsdb.csv"),
                      transNodedbFile=str(case / "transitionsdb.csv"), meanRayleigh=5 * 60)
        for t in range(int(min(m.pedDB[:, 9])), 10 * 60):
            m.initEvacuationAtTime()
            m.stepForward()
            m.checkTarget(ifOptChoice=bool(np.random.choice(2, p=[0.9, 0.1])))
            if not t % 10:
                m.computePedHistDenVelAtLinks()
                m.updateVelocityAllPedestrians()
        self.assertGreater(m.stateMat[:, 21:31].sum(), m.stateMat.shape[0] * 3)   # visits were counted


if __name__ == "__main__":
    unittest.main()
