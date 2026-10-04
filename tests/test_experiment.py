#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evacrl.experiment on a tiny hand-made case (only NumPy and Matplotlib are needed). The network (metres):

        4 (100,100) ---150--- 5 (250,100)
         |                      |
         100                    100
         |                      |
   0 --100-- 1 (100,0) --100-- 2 (200,0) --100-- 3 (300,0)  shelter     (the shortest way from 0 is 0-1-2-3, 300 m;
 (0,0)                                                                     by 4 and 5 it is 450 m; 5 is also linked to 3)

Ten agents start at nodes 0, 0, 0, 0, 0, 1, 1, 2, 2, 4.  Run: python -m unittest discover tests
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evacrl.experiment import (Case, Convergence, RunResult, build_manifest, calibrate, convergence_trace, derive_seeds, epsilon_at,  # noqa: E402
                               evaluate_policy, evaluate_state, make_model, repeat_shortest_path, run_episode, shortest_path_run,
                               write_manifest)
from evacrl.experiment import training as training_module  # noqa: E402
from evacrl.experiment import cli, output  # noqa: E402
from evacrl.experiment.plots import plot_comparison, plot_learning, plot_policy, policy_arrows  # noqa: E402
from evacrl.experiment.policy import compare_with_shortest_path, greedy_next_nodes, walks  # noqa: E402
from evacrl.options import ModelOptions  # noqa: E402

NODES = [(0, 0, 0, 0), (1, 100, 0, 0), (2, 200, 0, 0), (3, 300, 0, 1), (4, 100, 100, 0), (5, 250, 100, 0)]
LINKS = [(0, 1, 100), (1, 2, 100), (2, 3, 100), (1, 4, 100), (4, 5, 150), (5, 3, 100)]
STARTS = [0, 0, 0, 0, 0, 1, 1, 2, 2, 4]
SIM = 900   # seconds: the walk of 300 m takes about 250 s, departures within the first few minutes


def write_case(folder, layout="cases", popfile=1, nextnode=True, starts=None):
    """The tables of the tiny case; layout "cases": folder/data/*db.csv, "2024": nodes.csv, edges.csv, population_<n>.csv side by side."""
    folder = Path(folder)
    data = folder / "data" if layout == "cases" else folder
    data.mkdir(parents=True, exist_ok=True)
    nodes = np.array([[n, x, y, e, 1000 if e else 1] for n, x, y, e in NODES], dtype=float)
    links = np.array([[k, a, b, length, 3] for k, (a, b, length) in enumerate(LINKS)], dtype=int)
    actions = np.zeros((len(NODES), 12), dtype=int)
    transitions = np.zeros((len(NODES), 12), dtype=int)
    for n, _, _, shelter in NODES:
        actions[n, 0] = transitions[n, 0] = n
        if shelter:
            actions[n, 1] = transitions[n, 1] = 1
            actions[n, 2], transitions[n, 2] = -1, n
            continue
        mine = [(k, b) for k, a, b, _, _ in links if a == n] + [(k, a) for k, a, b, _, _ in links if b == n]
        actions[n, 1] = transitions[n, 1] = len(mine)
        for i, (k, other) in enumerate(mine):
            actions[n, 2 + i], transitions[n, 2 + i] = k, other
    starts = STARTS if starts is None else starts
    agents = np.zeros((len(starts), 5), dtype=int)
    agents[:, 4] = starts
    names = dict(agents="agentsdb.csv", nodes="nodesdb.csv", links="linksdb.csv") if layout == "cases" else \
        dict(agents=f"population_{popfile}.csv", nodes="nodes.csv", links="edges.csv")
    np.savetxt(data / names["nodes"], nodes, delimiter=",", fmt="%d,%.6f,%.6f,%d,%d", header="number,coord_x,coord_y,evacuation,reward")
    np.savetxt(data / names["links"], links, delimiter=",", fmt="%d", header="number,node1,node2,length,width")
    np.savetxt(data / names["agents"], agents, delimiter=",", fmt="%d", header="age,gender,hhType,hhId,Node")
    np.savetxt(data / "actionsdb.csv", actions, delimiter=",", fmt="%d")
    np.savetxt(data / "transitionsdb.csv", transitions, delimiter=",", fmt="%d")
    if nextnode:
        np.savetxt(data / "nextnode.csv", [[0, 1], [1, 2], [2, 3], [3, 3], [4, 5], [5, 3]], delimiter=",", fmt="%d", header="node,next node")
    return folder


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.case = Case.load(write_case(self.dir / "tiny"))

    def tearDown(self):
        self._tmp.cleanup()


class Seeds(unittest.TestCase):
    def test_the_first_seeds_of_a_longer_list_are_those_of_a_shorter_one(self):
        self.assertEqual(derive_seeds(7, 12)[:5], derive_seeds(7, 5))

    def test_the_same_base_seed_gives_the_same_seeds_and_another_gives_others(self):
        self.assertEqual(derive_seeds(3, 6), derive_seeds(3, 6))
        self.assertTrue(set(derive_seeds(3, 6)).isdisjoint(derive_seeds(4, 6)))

    def test_the_streams_of_an_experiment_do_not_share_seeds(self):
        names = ("shortest_path", "training", "evaluation", "policy")
        lists = [set(derive_seeds(0, 50, s)) for s in names]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertTrue(lists[i].isdisjoint(lists[j]), (names[i], names[j]))

    def test_the_seeds_can_seed_numpy(self):
        for s in derive_seeds(1, 20):
            self.assertIsInstance(s, int)
            self.assertTrue(0 <= s < 2 ** 32)
            np.random.seed(s)


class Cases(Base):
    def test_the_layout_of_cases_and_its_files(self):
        self.assertEqual(self.case.name, "tiny")
        self.assertEqual([k for k, _ in self.case.files], ["agentsProfileName", "nodesdbFile", "linksdbFile", "transLinkdbFile", "transNodedbFile"])
        self.assertTrue(self.case.kwargs["nodesdbFile"].endswith(os.path.join("data", "nodesdb.csv")))
        self.assertTrue(self.case.nextnode.endswith("nextnode.csv"))

    def test_the_layout_of_the_2024_study_and_its_population_file(self):
        write_case(self.dir / "study", layout="2024", popfile=3)
        case = Case.load(self.dir / "study", popfile=3)
        self.assertEqual([os.path.basename(p) for _, p in case.files], ["population_3.csv", "nodes.csv", "edges.csv", "actionsdb.csv", "transitionsdb.csv"])
        with self.assertRaisesRegex(FileNotFoundError, "population_1.csv"):
            Case.load(self.dir / "study")

    def test_no_next_node_table_is_allowed_and_a_missing_case_is_reported(self):
        self.assertIsNone(Case.load(write_case(self.dir / "plain", nextnode=False)).nextnode)
        with self.assertRaisesRegex(FileNotFoundError, "no case here"):
            Case.load(self.dir / "nothing")

    def test_checksums_name_every_input_file(self):
        sums = self.case.checksums()
        self.assertEqual(sorted(sums), ["actionsdb.csv", "agentsdb.csv", "linksdb.csv", "nextnode.csv", "nodesdb.csv", "transitionsdb.csv"])
        self.assertTrue(all(len(v) == 64 for v in sums.values()))
        path = Path(self.case.kwargs["linksdbFile"])
        path.write_text(path.read_text() + "\n")
        self.assertNotEqual(Case.load(self.dir / "tiny").checksums()["linksdb.csv"], sums["linksdb.csv"])

    def test_a_case_is_hashable_and_picklable(self):
        import pickle
        self.assertEqual(pickle.loads(pickle.dumps(self.case)), self.case)
        self.assertEqual(len({self.case, Case.load(self.dir / "tiny")}), 1)


class Results(unittest.TestCase):
    def test_safe_at_and_the_last_evacuee_by_hand(self):
        r = RunResult(seed=1, agents=5, t0=5, curve=np.array([0, 0, 3, 3, 5, 5, 5]))   # after second 5, 6, ..., 11
        self.assertEqual([r.safe_at(s) for s in (4, 5, 6, 7, 8, 9, 10, 100)], [0, 0, 0, 0, 3, 3, 5, 5])   # after s seconds: the steps up to second s - 1
        self.assertEqual(r.last_evacuee, 9.0)                                          # 5 is first recorded at second 9 (the study's time column)
        self.assertEqual(r.evacuated, 5)
        self.assertEqual(r.safe_at(r.last_evacuee + 1), r.evacuated)                   # ... which is 10 s after the start of the clock
        empty = RunResult(1, 5, 300, np.zeros(0, dtype=np.int64))                       # the simulation ended before the first departure
        self.assertEqual((empty.safe_at(1800), empty.evacuated), (0, 0))
        self.assertTrue(np.isnan(empty.last_evacuee))
        none = RunResult(1, 5, 0, np.zeros(10, dtype=int))
        self.assertTrue(np.isnan(none.last_evacuee))
        self.assertEqual(none.evacuated, 0)

    def test_curves_are_aligned_in_time_and_carried_forward(self):
        a = RunResult(1, 2, 2, np.array([1, 2]))
        b = RunResult(2, 3, 2, np.array([0, 3]))
        time, safe = output.curves_matrix([a, b], 6)
        self.assertEqual(time.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(safe.tolist(), [[0, 0, 1, 2, 2, 2], [0, 0, 0, 3, 3, 3]])

    def test_runs_and_curves_are_written_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = [RunResult(11, 4, 1, np.array([0, 1, 4, 4])), RunResult(12, 4, 2, np.array([0, 0, 0, 0]))]
            output.write_runs(tmp, results, horizon=3)
            rows = Path(tmp, "runs.csv").read_text().splitlines()
            self.assertEqual(rows[0], "run,seed,agents,safe_at_3s,evacuated,last_evacuee_s")
            self.assertEqual(rows[1:], ["0,11,4,1,4,3", "1,12,4,0,0,"])
            time, safe, agents = output.read_curves(tmp)
            self.assertEqual(safe.tolist(), [[0, 0, 1, 4, 4, 4], [0, 0, 0, 0, 0, 0]])
            self.assertEqual(agents.tolist(), [4, 4])
            self.assertEqual(Path(tmp, "curves_summary.csv").read_text().splitlines(), ["time_s,mean,p5,p95", "0,0.00,0.0,0.0"])   # one row per 10 s

    def test_the_curve_summary_is_the_mean_and_the_percentiles_every_ten_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            curves = [np.arange(25) * k for k in (1, 2, 3, 4, 5)]
            output.write_runs(tmp, [RunResult(k, 100, 0, c) for k, c in enumerate(curves)], horizon=20)
            rows = Path(tmp, "curves_summary.csv").read_text().splitlines()
            self.assertEqual(rows[0], "time_s,mean,p5,p95")
            # at second 10 the five runs record 10, 20, 30, 40, 50: mean 30, 5th percentile 10 + 0.2 * 10, 95th 40 + 0.8 * 10; at 20 twice that
            self.assertEqual(rows[1:], ["0,0.00,0.0,0.0", "10,30.00,12.0,48.0", "20,60.00,24.0,96.0"])


class Convergence_(unittest.TestCase):
    def test_the_statistics_by_hand(self):
        t = convergence_trace([1, 2, 3, 4], batch=2, min_runs=2, tol=0.5)
        self.assertEqual([c.n for c in t], [2, 4])
        self.assertAlmostEqual(t[0].mean, 1.5)
        self.assertAlmostEqual(t[0].sd, 0.5 ** 0.5)
        self.assertAlmostEqual(t[0].cv, 0.5 ** 0.5 / 1.5)
        self.assertAlmostEqual(t[1].mean, 2.5)
        self.assertAlmostEqual(t[1].sd, (5 / 3) ** 0.5)
        self.assertAlmostEqual(t[1].sem_rel, t[1].cv / 2)
        self.assertAlmostEqual(t[1].change, (2.5 - 1.5) / 2.5)
        self.assertTrue(np.isnan(t[0].change))

    def test_a_series_that_has_settled_converges_after_the_minimum_number_of_runs(self):
        rng = np.random.default_rng(0)
        values = 100 + rng.normal(0, 1, 200)
        t = convergence_trace(values, batch=10, min_runs=30, tol=0.01)
        self.assertFalse(any(c.converged for c in t if c.n < 30))
        self.assertTrue(t[-1].converged)

    def test_the_tolerance_is_strict_and_the_minimum_number_of_runs_is_enough(self):
        self.assertTrue(convergence_trace([5.0] * 30, batch=10, min_runs=30, tol=0.01)[-1].converged)         # exactly min_runs
        self.assertFalse(convergence_trace([5.0] * 29, batch=10, min_runs=30, tol=0.01)[-1].converged)
        self.assertFalse(convergence_trace([5.0] * 40, batch=10, min_runs=30, tol=0.0)[-1].converged)         # sem 0 is not below 0

    def test_a_noisy_series_does_not(self):
        values = np.tile([0.0, 100.0], 15)                                              # cv about 1; sem about 0.18
        self.assertFalse(convergence_trace(values, batch=10, min_runs=30, tol=0.01)[-1].converged)

    def test_a_zero_mean_never_converges_and_the_last_partial_batch_is_reported(self):
        t = convergence_trace([0.0] * 35, batch=10, min_runs=30)
        self.assertFalse(t[-1].converged)
        self.assertEqual([c.n for c in t], [10, 20, 30, 35])

    def test_a_metric_that_is_not_defined_yet_does_not_converge_and_does_not_warn(self):
        t = convergence_trace([np.nan] * 6, batch=3, min_runs=1, tol=10.0)
        self.assertEqual([c.n for c in t], [3, 6])
        self.assertFalse(any(c.converged for c in t))
        self.assertTrue(all(np.isnan(c.mean) for c in t))

    def test_nan_values_are_ignored(self):
        t = convergence_trace([1, np.nan, 3, 3], batch=4, min_runs=1)
        self.assertAlmostEqual(t[-1].mean, 7 / 3)


class ShortestPath(Base):
    def test_a_run_is_reproducible_and_everybody_arrives(self):
        a = shortest_path_run(self.case, 5, sim_time=SIM)
        b = shortest_path_run(self.case, 5, sim_time=SIM)
        np.testing.assert_array_equal(a.curve, b.curve)
        self.assertEqual((a.agents, a.evacuated), (10, 10))
        self.assertTrue((np.diff(a.curve) >= 0).all())
        self.assertGreater(a.last_evacuee, 300 / 1.19)                                   # nobody is faster than the free speed
        self.assertFalse(np.array_equal(a.curve, shortest_path_run(self.case, 6, sim_time=SIM).curve))

    def test_the_options_reach_the_model(self):
        model = make_model(self.case, "sarsa", ModelOptions.kochi2024())
        self.assertEqual(model.options, ModelOptions.kochi2024())
        self.assertEqual(make_model(self.case, "sarsa").options, ModelOptions())
        curves = [shortest_path_run(self.case, 2, sim_time=SIM, options=o).curve for o in (ModelOptions.kochi2024(), ModelOptions.legacy(), ModelOptions())]
        self.assertFalse(np.array_equal(curves[0], curves[1]) and np.array_equal(curves[1], curves[2]))     # the options are not ignored
        self.assertEqual([int(c[-1]) for c in curves], [10, 10, 10])

    def test_a_case_without_a_next_node_table_is_refused(self):
        case = Case.load(write_case(self.dir / "plain", nextnode=False))
        with self.assertRaisesRegex(ValueError, "nextnode.csv"):
            shortest_path_run(case, 0)

    def test_repeated_runs_have_derived_seeds_and_the_same_curves_for_any_number_of_workers(self):
        one = repeat_shortest_path(self.case, 6, seed=3, workers=1, sim_time=SIM, batch=3)
        two = repeat_shortest_path(self.case, 6, seed=3, workers=2, sim_time=SIM, batch=3)
        self.assertEqual(one.seeds, derive_seeds(3, 6, "shortest_path"))
        self.assertEqual(one.seeds, two.seeds)
        for a, b in zip(one.runs, two.runs):
            np.testing.assert_array_equal(a.curve, b.curve)
        self.assertEqual([c.n for c in one.trace["last_evacuee"]], [3, 6])

    def test_until_converged_stops_at_a_batch_and_does_not_depend_on_the_workers(self):
        kw = dict(until_converged=True, seed=4, sim_time=SIM, batch=5, min_runs=10, max_runs=40, tol=0.2)
        one = repeat_shortest_path(self.case, workers=1, **kw)
        two = repeat_shortest_path(self.case, workers=2, **kw)
        self.assertTrue(one.converged)
        self.assertEqual(len(one.runs) % 5, 0)
        self.assertGreaterEqual(len(one.runs), 10)
        self.assertEqual(one.seeds, two.seeds)
        self.assertEqual(one.seeds, derive_seeds(4, len(one.runs), "shortest_path"))

    def test_without_convergence_it_stops_at_the_maximum(self):
        res = repeat_shortest_path(self.case, until_converged=True, seed=1, sim_time=SIM, batch=4, min_runs=8, max_runs=8, tol=1e-9)
        self.assertFalse(res.converged)
        self.assertEqual(len(res.runs), 8)

    def test_a_fixed_number_of_runs_is_run_in_full_even_if_the_rule_is_met_earlier(self):
        res = repeat_shortest_path(self.case, 40, seed=1, sim_time=SIM, batch=10, min_runs=10, tol=10.0)
        self.assertEqual(len(res.runs), 40)
        self.assertTrue(res.converged)

    def test_the_departure_times_are_rayleigh_with_the_mean_in_minutes_that_was_asked_for(self):
        for minutes in (5.0, 1.0):
            np.random.seed(0)
            expected = np.round(np.random.rayleigh(scale=minutes * 60 * (2 / np.pi) ** 0.5, size=len(STARTS))).astype(int)
            np.random.seed(0)
            np.testing.assert_array_equal(make_model(self.case, "sarsa", mean_departure=minutes).pedDB[:, 9], expected)

    def test_it_needs_a_number_of_runs_or_the_rule(self):
        with self.assertRaises(ValueError):
            repeat_shortest_path(self.case)

    def test_runs_that_ended_with_agents_still_walking_are_counted(self):
        full = repeat_shortest_path(self.case, 3, seed=0, sim_time=SIM, batch=3, mean_departure=1.0)
        short = repeat_shortest_path(self.case, 3, seed=0, sim_time=250, batch=3, mean_departure=1.0)                # the walk of 300 m takes 250 s from the first departure
        self.assertEqual((full.incomplete, short.incomplete), (0, 3))

    def test_the_metrics(self):
        res = repeat_shortest_path(self.case, 3, seed=0, sim_time=SIM, batch=3)
        np.testing.assert_array_equal(res.metric("last_evacuee"), [r.last_evacuee for r in res.runs])
        np.testing.assert_array_equal(res.metric("safe", 600), [r.safe_at(600) for r in res.runs])
        with self.assertRaises(ValueError):
            res.metric("x")


class Training(Base):
    def kw(self, **extra):
        return dict(method="qlearning", sims=6, eval_every=3, eval_runs=2, sim_time=SIM, mean_departure=1.0, seed=2, **extra)

    def test_the_random_choice_rates(self):
        self.assertEqual([epsilon_at("calibration", s, 100) for s in (0, 50, 100)], [1.0, 1 / 1.5, 0.5])
        self.assertEqual([epsilon_at("quadratic", s, 10) for s in (0, 4, 8, 9)], [1.0, 0.75, 0.0, 0.0])
        self.assertEqual(epsilon_at("constant", 3, 10, constant=0.25), 0.25)
        with self.assertRaises(ValueError):
            epsilon_at("x", 0, 1)

    def test_checkpoints_every_few_simulations_and_the_best_by_greedy_evaluation(self):
        res = calibrate(self.case, **self.kw())
        self.assertEqual([c.sim for c in res.history], [3, 6])
        self.assertEqual(res.history[0].epsilon, epsilon_at("calibration", 2, 6))
        means = [c.eval_mean for c in res.history]
        self.assertEqual(res.best_eval, max(means))
        self.assertEqual(res.best_sim, res.history[int(np.argmax(means))].sim)               # the first of equals
        self.assertEqual([c.best for c in res.history].count(True), 1 + (means[1] > means[0]))
        self.assertEqual(res.agents, 10)
        self.assertTrue(res.best_state.shape[1] == 31 and res.final_state.shape[1] == 31)
        self.assertEqual(res.seeds["training"], derive_seeds(2, 6, "training"))
        self.assertEqual(res.seeds["evaluation"], derive_seeds(2, 2, "evaluation"))

    def test_the_best_checkpoint_is_chosen_by_the_greedy_evaluation_alone(self):
        class Fake:
            def __init__(self, n):
                self.evacuated = n
        scores = iter([3, 3, 8, 8, 5, 5])                                                    # evaluations of the three checkpoints, two seeds each
        original = training_module.map_jobs
        training_module.map_jobs = lambda function, jobs, workers=1: [Fake(next(scores)) for _ in jobs]
        try:
            res = calibrate(self.case, **dict(self.kw(), sims=6, eval_every=2))
        finally:
            training_module.map_jobs = original
        self.assertEqual([c.eval_mean for c in res.history], [3.0, 8.0, 5.0])
        self.assertEqual((res.best_sim, res.best_eval), (4, 8.0))
        self.assertEqual([c.best for c in res.history], [True, True, False])

    def test_the_final_checkpoint_is_always_evaluated(self):
        res = calibrate(self.case, **dict(self.kw(), sims=5))
        self.assertEqual([c.sim for c in res.history], [3, 5])

    def test_reproducible_and_the_same_for_any_number_of_workers(self):
        a = calibrate(self.case, **self.kw())
        b = calibrate(self.case, **self.kw(workers=2))
        self.assertEqual([(c.sim, c.eval_mean, c.train_safe) for c in a.history], [(c.sim, c.eval_mean, c.train_safe) for c in b.history])
        np.testing.assert_array_equal(a.final_state, b.final_state)
        np.testing.assert_array_equal(a.best_state, b.best_state)

    def test_evaluating_does_not_change_the_training(self):
        a = calibrate(self.case, **dict(self.kw(), eval_every=2))
        b = calibrate(self.case, **dict(self.kw(), eval_every=6))
        np.testing.assert_array_equal(a.final_state, b.final_state)

    def fake_evaluations(self, scores, **extra):
        class Fake:
            def __init__(self, n):
                self.evacuated = n
        scores = iter(scores)
        original = training_module.map_jobs
        training_module.map_jobs = lambda function, jobs, workers=1: [Fake(next(scores)) for _ in jobs]
        try:
            return calibrate(self.case, **dict(self.kw(), sims=6, eval_every=2, eval_runs=1, **extra))
        finally:
            training_module.map_jobs = original

    def by_hand(self, go_back_before):
        """The training of 6 simulations with checkpoints after 2, 4 and 6, written out: episode `go_back_before` (0-based) starts from a copy
        of the state after episode 2 instead of from the one the last episode left."""
        state, best = None, None
        for s, seed in enumerate(derive_seeds(2, 6, "training")):
            np.random.seed(seed)
            model = make_model(self.case, "qlearning", mean_departure=1.0)
            if s == go_back_before:
                model.stateMat = np.array(best, copy=True)
            elif state is not None:
                model.stateMat = state
            run_episode(model, SIM, epsilon=epsilon_at("calibration", s, 6))
            state = model.stateMat
            if s == 1:
                best = np.array(state, copy=True)
        return state, best

    def test_restarting_from_the_best_goes_back_to_it_after_a_checkpoint_that_did_not_improve(self):
        # evaluations: 5 after 2 simulations (the best), 3 after 4 (worse: the next episode starts from the best), 4 after 6
        res = self.fake_evaluations([5, 3, 4], restart_from_best=True)
        self.assertEqual([c.best for c in res.history], [True, False, False])
        expected_final, expected_best = self.by_hand(go_back_before=4)
        np.testing.assert_array_equal(res.final_state, expected_final)
        np.testing.assert_array_equal(res.best_state, expected_best)                         # the stored best was not trained on in place
        continuing, _ = self.by_hand(go_back_before=-1)
        self.assertFalse(np.array_equal(res.final_state, continuing))
        np.testing.assert_array_equal(self.fake_evaluations([5, 3, 4]).final_state, continuing)   # without the option: the latest state throughout

    def test_restarting_from_the_best_does_nothing_while_the_checkpoints_improve(self):
        res = self.fake_evaluations([3, 4, 5], restart_from_best=True)
        continuing, _ = self.by_hand(go_back_before=-1)
        np.testing.assert_array_equal(res.final_state, continuing)

    def test_the_progress_function_is_called_for_every_checkpoint(self):
        seen = []
        calibrate(self.case, **self.kw(progress=seen.append))
        self.assertEqual([c.sim for c in seen], [3, 6])

    def test_the_methods_all_learn(self):
        for method in ("sarsa", "qlearning", "mc"):
            res = calibrate(self.case, **dict(self.kw(), method=method, sims=2, eval_every=2))
            self.assertEqual(len(res.history), 1)
            self.assertGreater(res.final_state[:, 21:31].sum(), 0, method)                     # visits were counted
            self.assertNotEqual(float(np.abs(res.final_state[:, 11:21]).sum()), 0.0, method)  # and values were moved

    def test_the_training_loop_is_the_one_a_script_would_write(self):
        """The same training by hand: one episode per seed, each from the state the last one left, the random-choice rate of the schedule."""
        res = calibrate(self.case, **dict(self.kw(), sims=4, eval_every=1))
        state, exploring = None, []
        for s, seed in enumerate(derive_seeds(2, 4, "training")):
            np.random.seed(seed)
            model = make_model(self.case, "qlearning", mean_departure=1.0)
            if state is not None:
                model.stateMat = state
            exploring.append(int(run_episode(model, SIM, epsilon=epsilon_at("calibration", s, 4))[-1]))
            state = model.stateMat
        self.assertEqual([c.train_safe for c in res.history], [float(x) for x in exploring])
        self.assertEqual([c.epsilon for c in res.history], [epsilon_at("calibration", s, 4) for s in range(4)])
        np.testing.assert_array_equal(res.final_state, state)

    def test_the_train_safe_of_a_checkpoint_is_the_mean_since_the_previous_one(self):
        every_one = calibrate(self.case, **dict(self.kw(), sims=4, eval_every=1))
        every_two = calibrate(self.case, **dict(self.kw(), sims=4, eval_every=2))
        singles = [c.train_safe for c in every_one.history]
        self.assertEqual([c.train_safe for c in every_two.history], [np.mean(singles[:2]), np.mean(singles[2:])])

    def test_the_evaluation_uses_the_same_seeds_at_every_checkpoint(self):
        res = calibrate(self.case, **self.kw())
        runs = [evaluate_state(self.case, "qlearning", res.best_state, es, sim_time=SIM, mean_departure=1.0) for es in res.seeds["evaluation"]]
        self.assertEqual(res.best_eval, float(np.mean([r.evacuated for r in runs])))

    def test_the_arguments_are_checked(self):
        for bad in (dict(schedule="x"), dict(sims=0), dict(eval_every=0), dict(eval_runs=0)):
            with self.assertRaises(ValueError):
                calibrate(self.case, **dict(self.kw(), **bad))

    def test_evaluating_a_policy_leaves_its_state_alone(self):
        res = calibrate(self.case, **self.kw())
        before = res.best_state.copy()
        runs = evaluate_policy(self.case, "qlearning", res.best_state, 3, seed=1, sim_time=SIM, mean_departure=1.0)
        np.testing.assert_array_equal(res.best_state, before)
        self.assertEqual(len(runs), 3)
        self.assertEqual([r.seed for r in runs], derive_seeds(1, 3, "policy"))
        again = evaluate_state(self.case, "qlearning", res.best_state, runs[0].seed, sim_time=SIM, mean_departure=1.0)
        np.testing.assert_array_equal(again.curve, runs[0].curve)

    def spy(self, model, name):
        calls, original = [], getattr(model, name)

        def wrapper(*args, **kwargs):
            calls.append(kwargs.get("ifOptChoice", args[0] if args else None))
            return original(*args, **kwargs)
        setattr(model, name, wrapper)
        return calls

    def test_epsilon_is_the_probability_of_a_random_choice(self):
        for eps, low, high in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.4, 0.6)):
            np.random.seed(0)
            model = make_model(self.case, "qlearning", mean_departure=1.0)
            calls = self.spy(model, "checkTarget")
            run_episode(model, SIM, epsilon=eps)
            random_share = 1.0 - float(np.mean(calls))                                       # ifOptChoice False: a random choice
            self.assertTrue(low <= random_share <= high, (eps, random_share))

    def test_a_shortest_path_episode_never_asks_for_a_choice(self):
        np.random.seed(0)
        model = make_model(self.case, "sarsa", mean_departure=1.0)
        model.loadShortestPathDB(self.case.nextnode)
        calls = self.spy(model, "checkTarget")
        run_episode(model, 300, shortest_path=True)
        self.assertEqual(calls, [])

    def test_only_a_monte_carlo_model_learns_at_the_end(self):
        for method, expected in (("mc", 1), ("qlearning", 0), ("sarsa", 0)):
            np.random.seed(0)
            model = make_model(self.case, method, mean_departure=1.0)
            calls = self.spy(model, "updateValueFunctionDB")
            run_episode(model, 300, epsilon=1.0)
            self.assertEqual(len(calls), expected, method)

    def test_run_episode_with_random_choices_explores(self):
        np.random.seed(0)
        model = make_model(self.case, "qlearning", mean_departure=1.0)
        curve = run_episode(model, SIM, epsilon=1.0)
        self.assertEqual(len(curve), SIM - int(min(model.pedDB[:, 9])))
        self.assertGreater(model.stateMat[:, 21:31].sum(), 0)                              # something was learned


class Manifest(Base):
    def test_what_is_recorded(self):
        m = build_manifest("shortest_path", self.case, dict(runs=3, tol=np.float64(0.01)), options=ModelOptions.legacy(),
                           seeds=dict(base=1, runs=derive_seeds(1, 3)), results=dict(mean=np.float64(2.5)), started=0.0, argv=["python", "x"])
        self.assertEqual(m["kind"], "shortest_path")
        self.assertEqual(m["case"]["files"], self.case.checksums())
        self.assertEqual(m["options"]["segmentIndex"], "raw")
        self.assertEqual(m["parameters"], dict(runs=3, tol=0.01))
        self.assertEqual(m["seeds"]["runs"], derive_seeds(1, 3))
        self.assertEqual((m["results"], m["command"]), (dict(mean=2.5), "python x"))
        for key in ("evacrl", "python", "numpy", "platform", "git"):
            self.assertIn(key, m["software"])
        self.assertTrue(m["created_utc"].endswith("Z"))

    def test_it_is_written_as_json(self):
        m = build_manifest("calibration", self.case, dict(a=np.arange(3)), argv=["x"])
        path = write_manifest(self.dir / "out", m)
        self.assertEqual(json.loads(Path(path).read_text())["parameters"], dict(a=[0, 1, 2]))


class Robustness(Base):
    """What the experiments do with inputs at the edges."""

    def test_zero_runs_are_refused(self):
        for bad in (dict(runs=0), dict(runs=-1), dict(until_converged=True, max_runs=0), dict(runs=3, batch=0)):
            with self.assertRaises(ValueError):
                repeat_shortest_path(self.case, **dict(dict(sim_time=SIM), **bad))
        with self.assertRaises(ValueError):
            evaluate_policy(self.case, "qlearning", make_model(self.case, "qlearning").stateMat, 0)
        with self.assertRaisesRegex(ValueError, "no runs to write"):
            output.write_runs(self.dir / "x", [])

    def test_the_discount_is_the_options_unless_given(self):
        self.assertEqual(make_model(self.case, "qlearning").discount, 0.999)
        self.assertEqual(make_model(self.case, "qlearning", ModelOptions.legacy()).discount, 0.9)
        self.assertEqual(make_model(self.case, "qlearning", ModelOptions(discount=0.99)).discount, 0.99)
        self.assertEqual(make_model(self.case, "qlearning", ModelOptions(discount=0.99), discount=0.5).discount, 0.5)
        args = cli.build_parser().parse_args(["calibrate", "c", "--out", "o", "--set", "discount=0.95"])
        self.assertEqual((args.discount, cli.options_from(args).discount), (None, 0.95))
        args = cli.build_parser().parse_args(["calibrate", "c", "--out", "o", "--preset", "legacy"])
        self.assertEqual(cli.options_from(args).discount, 0.9)

    def test_a_negative_seed_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            derive_seeds(-1, 3)

    def test_a_case_folder_named_by_a_dot_has_the_name_of_the_folder(self):
        cwd = os.getcwd()
        os.chdir(self.dir / "tiny")
        try:
            self.assertEqual(Case.load(".").name, "tiny")
        finally:
            os.chdir(cwd)

    def test_a_min_runs_of_values_is_needed_not_a_min_runs_of_runs(self):
        t = convergence_trace([100.0, 100.5] + [np.nan] * 38, batch=10, min_runs=30, tol=0.01)
        self.assertFalse(any(c.converged for c in t))

    def test_a_state_of_another_case_is_refused_by_the_evaluation(self):
        good = make_model(self.case, "qlearning").stateMat
        for bad in (good[:2], good[:, :30], good[::-1], np.zeros((10, 31))):
            with self.assertRaisesRegex(ValueError, "not a state matrix of this case"):
                evaluate_state(self.case, "qlearning", bad, 0, sim_time=SIM)

    def test_the_evaluation_is_frozen_unless_asked_otherwise(self):
        np.random.seed(0)
        model = make_model(self.case, "qlearning", mean_departure=1.0)
        n = model.stateMat.shape[0]
        trained = calibrate(self.case, **dict(method="qlearning", sims=3, eval_every=3, eval_runs=1, sim_time=SIM, mean_departure=1.0, seed=1)).final_state
        for learn, changed in ((False, False), (True, True)):
            np.random.seed(0)
            model = make_model(self.case, "qlearning", mean_departure=1.0)
            model.stateMat = np.array(trained, copy=True)
            run_episode(model, SIM, epsilon=0.0, learn=learn)
            same = np.array_equal(model.stateMat[:len(trained), 11:31], trained[:, 11:31])
            self.assertEqual(not same, changed, learn)
        calls = []
        model = make_model(self.case, "mc", mean_departure=1.0)
        model.updateValueFunctionDB = lambda: calls.append(1)
        run_episode(model, 300, epsilon=1.0, learn=False)
        self.assertEqual(calls, [])

    def record(self, module, name):
        """Replace `module.name` by a wrapper that records the keyword arguments of its calls; returns (calls, restore)."""
        calls, original = [], getattr(module, name)

        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)
        setattr(module, name, wrapper)
        return calls, lambda: setattr(module, name, original)

    def test_the_evaluation_runs_frozen_unless_asked(self):
        from evacrl.experiment import runs as runs_module
        state = make_model(self.case, "qlearning").stateMat
        calls, restore = self.record(runs_module, "run_episode")
        try:
            evaluate_state(self.case, "qlearning", state, 0, sim_time=SIM)
            evaluate_state(self.case, "qlearning", state, 0, sim_time=SIM, learn=True)
            calibrate(self.case, **dict(method="qlearning", sims=2, eval_every=2, eval_runs=1, sim_time=SIM, mean_departure=1.0, seed=1))
            calibrate(self.case, **dict(method="qlearning", sims=2, eval_every=2, eval_runs=1, sim_time=SIM, mean_departure=1.0, seed=1, eval_learn=True))
        finally:
            restore()
        # the evaluations only (the training episodes call `run_episode` of the training module): two direct, one in each calibration
        self.assertEqual([kw["epsilon"] for _, kw in calls], [0.0] * 4)
        self.assertEqual([kw["learn"] for _, kw in calls], [False, True, False, True])

    def test_the_command_line_passes_the_learning_flags_on(self):
        state_file = self.dir / "s.csv"
        output.write_state(state_file, make_model(self.case, "qlearning").stateMat)
        calls, restore = self.record(cli, "evaluate_policy")
        calls2, restore2 = self.record(cli, "calibrate")
        try:
            for extra in ([], ["--keep-learning"]):
                self.run_cli("evaluate", self.dir / "tiny", "--state", state_file, "--runs", 1, "--out", self.dir / "e", "--time", SIM / 60, *extra)
            for extra in ([], ["--eval-keep-learning"]):
                self.run_cli("calibrate", self.dir / "tiny", "--out", self.dir / "c", "--sims", 2, "--eval-every", 2, "--eval-runs", 1, "--time", SIM / 60,
                             "--departure", 1, *extra)
        finally:
            restore()
            restore2()
        self.assertEqual([kw["learn"] for _, kw in calls], [False, True])
        self.assertEqual([kw["eval_learn"] for _, kw in calls2], [False, True])

    def test_a_frozen_and_an_adapting_evaluation_can_differ_and_each_is_reproducible(self):
        state = calibrate(self.case, **dict(method="qlearning", sims=4, eval_every=4, eval_runs=1, sim_time=SIM, mean_departure=1.0, seed=1)).final_state
        kw = dict(sim_time=SIM, mean_departure=1.0)
        frozen = evaluate_state(self.case, "qlearning", state, 5, **kw)
        again = evaluate_state(self.case, "qlearning", state, 5, **kw)
        adapting = evaluate_state(self.case, "qlearning", state, 5, learn=True, **kw)
        np.testing.assert_array_equal(frozen.curve, again.curve)
        self.assertEqual((frozen.evacuated, adapting.evacuated), (10, 10))

    def test_one_run_writes_valid_json(self):
        code, out = self.run_cli("sp", self.dir / "tiny", "--out", self.dir / "one", "--time", SIM / 60, "--horizon", SIM / 60, "--runs", 1)
        self.assertEqual(code, 0, out)
        text = (self.dir / "one" / "manifest.json").read_text()
        self.assertNotIn("NaN", text)
        results = json.loads(text, parse_constant=lambda c: self.fail(f"{c} in the manifest"))["results"]
        self.assertIsNone(results["safe_sd"])
        self.assertIsNone(results["cv_last_evacuee"])

    def test_the_horizon_cannot_be_after_the_end_and_zero_runs_end_with_a_message(self):
        with self.assertRaisesRegex(SystemExit, "later than the end of the simulation"):
            self.run_cli("sp", self.dir / "tiny", "--out", self.dir / "x", "--time", 5, "--horizon", 30, "--runs", 2)
        with self.assertRaisesRegex(SystemExit, "at least 1"):
            self.run_cli("sp", self.dir / "tiny", "--out", self.dir / "x", "--time", 5, "--horizon", 5, "--runs", 0)
        with self.assertRaisesRegex(SystemExit, "must not be negative"):
            self.run_cli("sp", self.dir / "tiny", "--out", self.dir / "x", "--time", 5, "--horizon", 5, "--runs", 2, "--seed", -1)

    def test_minutes_become_whole_seconds_by_rounding(self):
        self.assertEqual([cli._seconds(m) for m in (2.05, 4.1, 16.15, 30, 0.5)], [123, 246, 969, 1800, 30])

    def test_a_state_of_another_case_ends_the_evaluation_with_a_message(self):
        bad = self.dir / "bad.csv"
        np.savetxt(bad, np.zeros((3, 31)), delimiter=",")
        with self.assertRaisesRegex(SystemExit, "not a state matrix of this case"):
            self.run_cli("evaluate", self.dir / "tiny", "--state", bad, "--runs", 1, "--out", self.dir / "x", "--time", SIM / 60)

    def test_the_evaluation_manifest_identifies_the_policy_and_how_it_was_run(self):
        import hashlib
        state_file = self.dir / "s.csv"
        output.write_state(state_file, make_model(self.case, "qlearning").stateMat)
        for flag, name in ((None, "frozen"), ("--keep-learning", "adapting")):
            extra = [flag] if flag else []
            code, out = self.run_cli("evaluate", self.dir / "tiny", "--state", state_file, "--runs", 2, "--out", self.dir / name, "--time", SIM / 60,
                                     "--departure", 1, *extra)
            self.assertEqual(code, 0, out)
            self.assertIn("learning on" if flag else "frozen", out)
            manifest = json.loads((self.dir / name / "manifest.json").read_text())
            self.assertEqual(manifest["parameters"]["state_sha256"], hashlib.sha256(state_file.read_bytes()).hexdigest())
            self.assertEqual(manifest["parameters"]["keep_learning"], bool(flag))

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([str(a) for a in argv])
        return code, out.getvalue()


class Policy(Base):
    """Actions of the tiny case in the order of its tables: node 0: [1]; node 1: [2, 4, 0]; node 2: [3, 1]; node 4: [5, 1]; node 5: [3, 4]."""

    def state(self, values):
        state = make_model(self.case, "qlearning").stateMat.copy()
        state[:, 11:21] = 0.0
        for node, q in values.items():
            state[node, 11:11 + len(q)] = q
        return state

    def test_the_greedy_next_node_of_every_node(self):
        state = self.state({0: [1], 1: [0.1, 0.9, 0.2], 2: [5, 1], 4: [1, 2], 5: [3, 4]})
        # node 1: best is the second action (to 4); node 4: the second action (to 1); node 5: the second action (to 4); node 2: the first (to 3)
        self.assertEqual(greedy_next_nodes(self.case, state).tolist(), [1, 4, 3, 3, 1, 4])
        np.testing.assert_array_equal(greedy_next_nodes(self.case, self.state({1: [0.1, 0.1, 0.1]}))[:3], [1, 2, 3])   # ties: the first action

    def test_unused_action_slots_are_not_looked_at(self):
        state = self.state({0: [-5.0], 2: [-1.0, -2.0]})
        state[0, 12:21] = 99.0                                                              # slots of actions that do not exist
        self.assertEqual(greedy_next_nodes(self.case, state)[0], 1)

    def test_a_state_matrix_of_another_case_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a state matrix of this case"):
            greedy_next_nodes(self.case, self.state({})[::-1])

    def test_walks_by_hand(self):
        policy = np.array([1, 4, 3, 3, 5, 3])                                                # 0-1-4-5-3, 1-4-5-3, 2-3, 4-5-3, 5-3
        metres, hops = walks(self.case, policy)
        self.assertEqual(metres.tolist(), [450, 350, 100, 0, 250, 100])
        self.assertEqual(hops.tolist(), [4, 3, 1, 0, 2, 1])
        metres, hops = walks(self.case, policy, starts=[2, 0])
        self.assertEqual((metres.tolist(), hops.tolist()), ([100, 450], [1, 4]))

    def test_the_shortest_of_parallel_links_is_the_length_of_a_step(self):
        folder = self.dir / "parallel" / "data"
        folder.mkdir(parents=True)
        np.savetxt(folder / "nodesdb.csv", [[0, 0, 0, 0, 1], [1, 100, 0, 1, 1000]], delimiter=",", fmt="%d", header="number,x,y,evacuation,reward")
        np.savetxt(folder / "linksdb.csv", [[0, 0, 1, 500, 3], [1, 0, 1, 100, 3]], delimiter=",", fmt="%d", header="number,node1,node2,length,width")
        np.savetxt(folder / "actionsdb.csv", [[0, 2, 0, 1] + [0] * 8, [1, 1, -1] + [0] * 9], delimiter=",", fmt="%d")
        np.savetxt(folder / "transitionsdb.csv", [[0, 2, 1, 1] + [0] * 8, [1, 1, 1] + [0] * 9], delimiter=",", fmt="%d")
        np.savetxt(folder / "agentsdb.csv", [[0, 0, 0, 0, 0]], delimiter=",", fmt="%d", header="age,gender,hhType,hhId,Node")
        metres, hops = walks(Case.load(self.dir / "parallel"), np.array([1, 1]))
        self.assertEqual((metres.tolist(), hops.tolist()), ([100, 0], [1, 0]))

    def test_a_loop_and_a_dead_end_never_arrive(self):
        metres, _ = walks(self.case, np.array([1, 4, 3, 3, 1, 3]))                           # 1 -> 4 -> 1 ...
        self.assertTrue(np.isnan(metres[[0, 1, 4]]).all())
        self.assertEqual(metres[[2, 3, 5]].tolist(), [100, 0, 100])
        metres, _ = walks(self.case, np.array([-9999, 4, 3, 3, 5, 3]))
        self.assertTrue(np.isnan(metres[0]) and not np.isnan(metres[1]))

    def test_the_comparison_with_the_shortest_path_by_hand(self):
        state = self.state({0: [1], 1: [0.1, 0.9, 0.2], 2: [5, 1], 4: [1.0, 0.5], 5: [4, 3]})
        c = compare_with_shortest_path(self.case, state)
        # agents at nodes 0 x5, 1 x2, 2 x2, 4: policy walks 450, 350, 100, 250 m (mean 340); shortest path 300, 200, 100, 250 m (mean 235)
        self.assertEqual(c.agents, 10)
        self.assertAlmostEqual(c.metres, 340.0)
        self.assertAlmostEqual(c.metres_sp, 235.0)
        self.assertAlmostEqual(c.longer, 100 * (340 / 235 - 1))
        self.assertAlmostEqual(c.hops, 3.0)
        self.assertAlmostEqual(c.hops_sp, 2.3)
        self.assertAlmostEqual(c.agreement, 0.8)                                            # only the two agents at node 1 choose differently
        self.assertEqual(c.never_arrive, 0)

    def test_an_agent_who_starts_at_a_shelter_makes_no_choice_and_is_not_counted_as_agreeing(self):
        case = Case.load(write_case(self.dir / "mixed", starts=[1, 3]))
        c = compare_with_shortest_path(case, self.state({1: [0.1, 0.9, 0.2], 2: [5, 1]}))   # node 1 goes to 4, the shortest path goes to 2
        self.assertEqual((c.agents, c.agreement), (2, 0.0))

    def test_a_case_without_a_shortest_path_table_cannot_be_compared(self):
        case = Case.load(write_case(self.dir / "plain", nextnode=False))
        with self.assertRaisesRegex(ValueError, "nextnode.csv"):
            compare_with_shortest_path(case, self.state({}))

    def test_agents_that_never_arrive_are_counted_and_left_out_of_the_means(self):
        state = self.state({0: [1], 1: [0.1, 0.1, 0.9], 2: [5, 1], 4: [1.0, 0.5], 5: [4, 3]})   # 1 -> 0 -> 1: the agents at 0 and 1 loop
        c = compare_with_shortest_path(self.case, state)
        self.assertEqual(c.never_arrive, 7)
        self.assertAlmostEqual(c.metres, (2 * 100 + 250) / 3)

    def test_arrows_are_for_the_nodes_that_move_and_orange_where_the_shortest_path_differs(self):
        from evacrl.tables import load_table
        nodes = load_table(self.case.kwargs["nodesdbFile"])
        move, differs = policy_arrows(nodes, np.array([1, 4, 3, 3, 5, 3]), np.array([1, 2, 3, 3, 5, 3]))
        self.assertEqual((move.tolist(), differs.tolist()), ([0, 1, 2, 4, 5], [False, True, False, False, False]))   # not the shelter, node 3
        move, differs = policy_arrows(nodes, np.array([1, -9999, 3, 3, 5, 3]), None)
        self.assertEqual((move.tolist(), differs.tolist()), ([0, 2, 4, 5], [False] * 4))                              # node 1 has no way on

    def test_the_map_is_drawn(self):
        state = self.state({0: [1], 1: [0.1, 0.9, 0.2], 2: [5, 1], 4: [1.0, 0.5], 5: [4, 3]})
        from evacrl.tables import load_table
        with tempfile.TemporaryDirectory() as tmp:
            nxt = greedy_next_nodes(self.case, state)
            for sp in (load_table(self.case.nextnode, dtype=int)[:, 1], None):
                path = plot_policy(os.path.join(tmp, "m.png"), self.case, nxt, sp, title="t")
                self.assertEqual(Path(path).read_bytes()[:4], b"\x89PNG")

    def test_the_command_line(self):
        out = io.StringIO()
        state_file = self.dir / "state.csv"
        output.write_state(state_file, self.state({0: [1], 1: [0.1, 0.9, 0.2], 2: [5, 1], 4: [1.0, 0.5], 5: [4, 3]}))
        with contextlib.redirect_stdout(out):
            code = cli.main(["policy", str(self.dir / "tiny"), "--state", str(state_file), "--out", str(self.dir / "map.png")])
        self.assertEqual(code, 0)
        self.assertIn("10 agents", out.getvalue())
        self.assertIn("first choice is the shortest path's for 80.0 %", out.getvalue())
        self.assertIn("walk 340 m against 235 m (+44.7 %), 3.0 nodes against 2.3", out.getvalue())
        self.assertEqual((self.dir / "map.png").read_bytes()[:4], b"\x89PNG")


class Tables(unittest.TestCase):
    def test_the_convergence_and_calibration_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            output.write_convergence(tmp, {"safe": [Convergence(10, 5.0, 1.0, 0.2, 0.0632, float("nan"), False), Convergence(20, 5.1, 1.0, 0.196, 0.0438, 0.0196, True)]})
            rows = Path(tmp, "convergence.csv").read_text().splitlines()
            self.assertEqual(rows[0], "metric,n,mean,sd,cv,sem_rel,change,converged")
            self.assertEqual(rows[2], "safe,20,5.1,1,0.196,0.0438,0.0196,1")
            from evacrl.experiment import Checkpoint
            output.write_history(tmp, [Checkpoint(5, 0.9, 3.5, 4.0, 0.5, True), Checkpoint(10, 0.8, 4.5, 3.0, float("nan"), False)])
            h = output.read_history(tmp)
            self.assertEqual(h["sim"].tolist(), [5.0, 10.0])
            self.assertEqual(h["eval_mean"].tolist(), [4.0, 3.0])
            self.assertEqual(h["best"].tolist(), [1.0, 0.0])
            self.assertTrue(np.isnan(h["eval_sd"][1]))

    def test_a_state_matrix_is_written_as_the_model_does(self):
        base = make_model(Case.load(write_case(tempfile.mkdtemp())), "qlearning")
        base.stateMat = np.array(base.stateMat, dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            output.write_state(os.path.join(tmp, "a.csv"), base.stateMat)
            base.exportStateMatrix(os.path.join(tmp, "b.csv"))
            self.assertEqual(Path(tmp, "a.csv").read_bytes(), Path(tmp, "b.csv").read_bytes())


class Plots(unittest.TestCase):
    def test_the_figures_are_written(self):
        rng = np.random.default_rng(0)
        time = np.arange(1800)
        sp = (time, np.cumsum(rng.random((5, 1800)) < 0.01, axis=1))
        rl = (time, np.cumsum(rng.random((4, 1800)) < 0.008, axis=1))
        history = dict(sim=np.array([10.0, 20, 30]), train_safe=np.array([3.0, 4, 5]), eval_mean=np.array([4.0, 6, 5]),
                       eval_sd=np.array([1.0, 1, np.nan]), best=np.array([1.0, 1, 0]))
        with tempfile.TemporaryDirectory() as tmp:
            import matplotlib.image as mpimg
            colour = lambda hex_: np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
            shows = lambda image, hex_: bool((np.abs(image[:, :, :3] - colour(hex_)).max(axis=2) < 0.02).sum() > 200)    # pixels of that colour
            for path, expected in ((plot_comparison(os.path.join(tmp, "a.png"), sp, rl, agents=50, title="t"), ("#2a78d6", "#eb6834")),
                                   (plot_comparison(os.path.join(tmp, "b.png"), sp, rl, horizon=600), ("#2a78d6", "#eb6834")),
                                   (plot_learning(os.path.join(tmp, "c.png"), history, reference=7.0, title="t"), ("#2a78d6", "#eb6834", "#1baf7a")),
                                   (plot_learning(os.path.join(tmp, "d.png"), history), ("#eb6834", "#1baf7a"))):
                self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                image = mpimg.imread(path)
                self.assertGreater(image.shape[0], 400)
                for hex_ in expected:
                    self.assertTrue(shows(image, hex_), (path, hex_))                                                     # the series are drawn in their colours
            self.assertFalse(shows(mpimg.imread(os.path.join(tmp, "d.png")), "#2a78d6"))                                  # no reference line, no blue


class CommandLine(Base):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([str(a) for a in argv])
        return code, out.getvalue()

    def sp(self, name, *extra):
        return self.run_cli("sp", self.dir / "tiny", "--out", self.dir / name, "--time", SIM / 60, "--horizon", SIM / 60, "--seed", 3, *extra)

    def test_shortest_path_runs_with_their_results_and_manifest(self):
        code, out = self.sp("sp", "--runs", 4, "--batch", 2)
        self.assertEqual(code, 0, out)
        self.assertIn("4 shortest-path runs, 10 agents", out)
        folder = self.dir / "sp"
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["convergence.csv", "curves.npz", "curves_summary.csv", "manifest.json", "runs.csv"])
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual((manifest["kind"], manifest["results"]["runs"]), ("shortest_path", 4))
        self.assertEqual(manifest["results"]["incomplete_runs"], 0)
        self.assertEqual(manifest["seeds"]["runs"], derive_seeds(3, 4, "shortest_path"))
        self.assertEqual(manifest["case"]["files"], self.case.checksums())
        self.assertEqual(len((folder / "runs.csv").read_text().splitlines()), 5)

    def test_a_warning_when_agents_are_still_walking_at_the_end(self):
        code, out = self.run_cli("sp", self.dir / "tiny", "--out", self.dir / "cut", "--time", 250 / 60, "--horizon", 250 / 60, "--runs", 2, "--seed", 3)
        self.assertEqual(code, 0)
        self.assertIn("warning: 2 of 2 runs ended with agents still walking", out)
        self.assertEqual(json.loads((self.dir / "cut" / "manifest.json").read_text())["results"]["incomplete_runs"], 2)
        self.assertNotIn("warning", self.sp("fine", "--runs", 2)[1])

    def test_until_converged_from_the_command_line(self):
        code, out = self.sp("conv", "--until-converged", "--batch", 2, "--min-runs", 4, "--max-runs", 10, "--tol", 5)
        self.assertEqual(code, 0, out)
        self.assertIn("converged: True", out)
        manifest = json.loads((self.dir / "conv" / "manifest.json").read_text())
        self.assertTrue(manifest["results"]["converged"])
        self.assertEqual(manifest["results"]["runs"], 4)                                    # the first batch boundary at which the rule holds
        rows = (self.dir / "conv" / "convergence.csv").read_text().splitlines()
        self.assertEqual(rows[0], "metric,n,mean,sd,cv,sem_rel,change,converged")
        self.assertTrue(rows[-1].endswith(",1"))

    def test_the_same_seed_gives_the_same_files_for_any_number_of_workers(self):
        self.sp("a", "--runs", 4, "--workers", 1)
        self.sp("b", "--runs", 4, "--workers", 2)
        self.assertEqual((self.dir / "a" / "runs.csv").read_text(), (self.dir / "b" / "runs.csv").read_text())
        np.testing.assert_array_equal(output.read_curves(self.dir / "a")[1], output.read_curves(self.dir / "b")[1])

    def test_it_needs_runs_or_the_rule_and_the_options_are_checked(self):
        with self.assertRaises(SystemExit):
            self.sp("x")
        with self.assertRaises(SystemExit):
            self.sp("x", "--runs", 2, "--set", "nope=1")
        with self.assertRaises(SystemExit):
            self.sp("x", "--runs", 2, "--set", "discounting=")

    def test_options_are_set_from_the_command_line(self):
        parser = cli.build_parser()
        args = parser.parse_args(["sp", "c", "--out", "o", "--preset", "kochi2024", "--set", "discounting=second", "--set", "surviveReward=5e4"])
        o = cli.options_from(args)
        self.assertEqual((o.segmentIndex, o.discounting, o.surviveReward), ("raw", "second", 5e4))

    def test_calibrate_evaluate_and_compare(self):
        self.sp("sp", "--runs", 3)
        code, out = self.run_cli("calibrate", self.dir / "tiny", "--out", self.dir / "cal", "--sims", 4, "--eval-every", 2, "--eval-runs", 2,
                                 "--time", SIM / 60, "--departure", 1, "--seed", 5, "--sp", self.dir / "sp")
        self.assertEqual(code, 0, out)
        self.assertIn("best checkpoint", out)
        cal = self.dir / "cal"
        self.assertEqual(sorted(p.name for p in cal.iterdir()), ["best_state.csv", "calibration.csv", "final_state.csv", "learning.png", "manifest.json"])
        history = output.read_history(cal)
        self.assertEqual(history["sim"].tolist(), [2.0, 4.0])
        _, sp_safe, _ = output.read_curves(self.dir / "sp")
        reference = json.loads((cal / "manifest.json").read_text())["results"]["sp_reference"]
        self.assertEqual(reference, float(sp_safe[:, SIM - 1].mean()))                       # the shortest-path mean at the end of the episode
        self.assertGreater(reference, 0)
        code, out = self.run_cli("evaluate", self.dir / "tiny", "--state", cal / "best_state.csv", "--runs", 3, "--out", self.dir / "ev",
                                 "--time", SIM / 60, "--departure", 1, "--seed", 5)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.dir / "ev" / "manifest.json").read_text())["kind"], "evaluation")
        code, out = self.run_cli("compare", "--sp", self.dir / "sp", "--rl", self.dir / "ev", "--out", self.dir / "fig.png", "--horizon", SIM / 60)
        self.assertEqual(code, 0, out)
        self.assertEqual((self.dir / "fig.png").read_bytes()[:4], b"\x89PNG")

    def test_calibration_is_reproducible_through_the_command_line(self):
        for name, workers in (("c1", 1), ("c2", 2)):
            self.run_cli("calibrate", self.dir / "tiny", "--out", self.dir / name, "--sims", 4, "--eval-every", 2, "--eval-runs", 2,
                         "--time", SIM / 60, "--departure", 1, "--seed", 9, "--workers", workers)
        self.assertEqual((self.dir / "c1" / "calibration.csv").read_text(), (self.dir / "c2" / "calibration.csv").read_text())
        self.assertEqual((self.dir / "c1" / "best_state.csv").read_bytes(), (self.dir / "c2" / "best_state.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
