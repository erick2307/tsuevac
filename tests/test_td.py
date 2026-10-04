#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The temporal-difference update (SARSA, Q-learning) and the discounting of the three methods, with the
expected values worked out by hand on the five-node corridor of test_engine_options.py:

    node 0 (1 action) --- node 1 (2 actions: to 0, to 2) --- node 2 (evacuation node, 1 action)

With alpha = 0.05, discount = 0.9 (per decision unless a test says otherwise: `HAND`), stepReward = -1 per second, surviveReward = 1e5.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evacrl.mc import MonteCarlo  # noqa: E402
from evacrl.options import ModelOptions  # noqa: E402
from evacrl.qlearn import QLearning  # noqa: E402
from evacrl.sarsa import SARSA  # noqa: E402
from evacrl.td import TemporalDifference  # noqa: E402
from test_engine_options import ACTIONS, TRANSITIONS, write_case  # noqa: E402

# the corridor with a second action at the evacuation node 2: stay (-1 / node 2) and go back to node 1 (link 1 / node 1)
ACTIONS_SHELTER_WITH_EXIT = {**ACTIONS, 2: [-1, 1]}
TRANSITIONS_SHELTER_WITH_EXIT = {**TRANSITIONS, 2: [2, 1]}

ALPHA, GAMMA, SURVIVE = 0.05, 0.9, 100000
HAND = ModelOptions(discounting="decision", discount=GAMMA)   # the settings of the hand-worked values below (not the defaults)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        np.random.seed(0)

    def tearDown(self):
        self._tmp.cleanup()

    def model(self, cls, agents_at=(0,), options=None, meanRayleigh=10, **tables):
        d = write_case(self.dir, agents_at, **tables)
        return cls(agentsProfileName=str(d / "agents.csv"), nodesdbFile=str(d / "nodes.csv"),
                   linksdbFile=str(d / "links.csv"), transLinkdbFile=str(d / "actions.csv"),
                   transNodedbFile=str(d / "transitions.csv"), meanRayleigh=meanRayleigh, options=HAND if options is None else options)

    def experience(self, m, steps):
        """Give agent 0 the experience [(node, action, time), ...]; returns the state index of each step."""
        states = [m.getStateIndexAtNode(node) for node, _, _ in steps]
        m.expeStat[0] = [np.array([s, a, t]) for s, (_, a, t) in zip(states, steps)]
        return states


class Classes(Base):
    def test_q_learning_is_not_sarsa(self):
        self.assertFalse(issubclass(QLearning, SARSA))
        self.assertFalse(issubclass(SARSA, QLearning))
        self.assertTrue(issubclass(QLearning, TemporalDifference) and issubclass(SARSA, TemporalDifference))

    def test_the_base_class_has_no_value_to_bootstrap_from(self):
        m = self.model(TemporalDifference)
        with self.assertRaises(NotImplementedError):
            m.bootstrapValue(0, 0, False)


class UpdateRule(Base):
    def two_choices(self, cls, chosen):
        """At node 0 (action 0, Q = 2) then at node 1, 100 s later, where Q = (-50, -10): the agent chose `chosen`."""
        m = self.model(cls)
        s0, s1 = self.experience(m, [(0, 0, 10), (1, chosen, 110)])
        m.stateMat[s0, 11] = 2.0
        m.stateMat[s1, 11:13] = [-50.0, -10.0]
        m.tdControl(0)
        return m, s0

    def test_q_learning_bootstraps_from_the_best_action_whatever_was_chosen(self):
        # target = -1*100 + 0.9*max(-50, -10) = -109;  Q = 2 + 0.05*(-109 - 2) = -3.55
        for chosen in (0, 1):
            m, s0 = self.two_choices(QLearning, chosen)
            self.assertAlmostEqual(m.stateMat[s0, 11], -3.55, places=9, msg=f"chosen action {chosen}")

    def test_sarsa_bootstraps_from_the_action_that_was_chosen(self):
        # chose 0 (Q = -50): target = -100 + 0.9*(-50) = -145;  Q = 2 + 0.05*(-147) = -5.35
        m, s0 = self.two_choices(SARSA, 0)
        self.assertAlmostEqual(m.stateMat[s0, 11], -5.35, places=9)
        # chose 1 (the best, Q = -10): the same as Q-learning, -3.55
        m, s0 = self.two_choices(SARSA, 1)
        self.assertAlmostEqual(m.stateMat[s0, 11], -3.55, places=9)

    def test_the_visit_count_of_the_updated_action_goes_up_by_one(self):
        for cls in (SARSA, QLearning):
            m = self.model(cls)
            s0, s1 = self.experience(m, [(0, 0, 10), (1, 0, 110)])
            before = m.stateMat[s0, 21], m.stateMat[s1, 21]
            m.tdControl(0)
            self.assertEqual(m.stateMat[s0, 21], before[0] + 1)
            self.assertEqual(m.stateMat[s1, 21], before[1])  # the current choice is updated at the next one

    def test_unused_action_slots_are_not_candidates_for_the_maximum(self):
        # node 0 has one action; its other 9 slots hold 0, which must not beat Q = -30.
        # target = -1*30 + 0.9*(-30) = -57;  Q = -10 + 0.05*(-57 + 10) = -12.35   (unmasked: -11)
        m = self.model(QLearning)
        s1, s0 = self.experience(m, [(1, 1, 10), (0, 0, 40)])
        m.stateMat[s1, 12] = -10.0
        m.stateMat[s0, 11] = -30.0
        m.tdControl(0)
        self.assertAlmostEqual(m.stateMat[s1, 12], -12.35, places=9)

    def test_reaching_an_evacuation_node_ends_the_episode(self):
        # node 2 is the shelter, one action (stay), Q = 0.5 at first. The agent arrives 100 s after its choice at node 1.
        #   Q(shelter) = 0.5 + 0.05*(1e5 - 0.5) = 5000.475
        #   target = -100 + 0.9*5000.475 = 4400.4275;  Q(node 1, to 2) = -10 + 0.05*(4400.4275 + 10) = 210.521375
        for cls in (SARSA, QLearning):
            m = self.model(cls)
            s1, s2 = self.experience(m, [(1, 1, 10), (2, 0, 110)])
            m.stateMat[s1, 12] = -10.0
            m.pedDB[0, 10] = 1
            m.tdControl(0)
            self.assertAlmostEqual(m.stateMat[s2, 11], 5000.475, places=6, msg=cls.__name__)
            self.assertAlmostEqual(m.stateMat[s1, 12], 210.521375, places=6, msg=cls.__name__)
            self.assertEqual(m.stateMat[s2, 21], 2)  # the arrival counts as a visit of the shelter's action

    def test_arriving_is_terminal_even_if_the_evacuation_node_has_other_actions(self):
        # a shelter that also has a way out (to node 1), Q = 0.5 for both; the agent arrives by choosing to stay (action 0).
        # Terminal: Q(shelter, stay) -> 5000.475, then the target is -100 + 0.9*5000.475 = 4400.4275, the value of staying.
        # With the exit valued at 90000, a plain max over both actions would give -100 + 0.9*90000 = 80900 instead.
        tables = dict(actions={**ACTIONS_SHELTER_WITH_EXIT}, transitions={**TRANSITIONS_SHELTER_WITH_EXIT})
        m = self.model(QLearning, **tables)
        s1, s2 = self.experience(m, [(1, 1, 10), (2, 0, 110)])
        m.stateMat[s1, 12] = -10.0
        m.stateMat[s2, 12] = 90000.0       # the way out looks better than staying, which must not matter
        m.pedDB[0, 10] = 1
        m.tdControl(0)
        self.assertAlmostEqual(m.stateMat[s1, 12], 210.521375, places=6)

    def run_corridor(self, cls, epsilon, seed=3, seconds=500):
        np.random.seed(seed)
        m = self.model(cls, agents_at=[0] * 8 + [1] * 12)
        for t in range(int(min(m.pedDB[:, 9])), seconds):
            m.initEvacuationAtTime()
            m.stepForward()
            m.checkTarget(ifOptChoice=bool(np.random.choice(2, p=[epsilon, 1.0 - epsilon])))
            if not t % 10:
                m.computePedHistDenVelAtLinks()
                m.updateVelocityAllPedestrians()
        return m

    def test_with_greedy_choices_q_learning_is_sarsa_to_the_last_digit(self):
        q = self.run_corridor(QLearning, epsilon=0.0)
        s = self.run_corridor(SARSA, epsilon=0.0)
        self.assertGreater(q.stateMat[:, 21:31].sum(), q.stateMat.shape[0] * 3)  # it learned something
        np.testing.assert_array_equal(q.stateMat, s.stateMat)

    def test_with_exploration_they_differ(self):
        q = self.run_corridor(QLearning, epsilon=0.5)
        s = self.run_corridor(SARSA, epsilon=0.5)
        self.assertFalse(np.array_equal(q.stateMat, s.stateMat))


class Discounting(Base):
    def test_factor_by_method_and_option(self):
        expected = {"decision": GAMMA, "second": GAMMA ** 100}
        for cls, by_method in ((SARSA, "decision"), (QLearning, "decision"), (MonteCarlo, "second")):
            # "decision" (here: HAND) is once per decision for every method; "method" gives each its own, as in 2021
            self.assertEqual(self.model(cls).discountFactor(100), expected["decision"], cls.__name__)
            m = self.model(cls, options=ModelOptions(discounting="method", discount=GAMMA))
            self.assertEqual(m.discountFactor(100), expected[by_method], cls.__name__)
            self.assertEqual(self.model(cls, options=ModelOptions.legacy()).discountFactor(100), expected[by_method])
            for mode in ("decision", "second"):
                m = self.model(cls, options=ModelOptions(discounting=mode, discount=GAMMA))
                self.assertEqual(m.discountFactor(100), expected[mode], f"{cls.__name__} {mode}")

    def test_the_default_is_a_discount_of_0_999_per_second_for_every_method(self):
        for cls in (SARSA, QLearning, MonteCarlo):
            m = self.model(cls, options=ModelOptions())
            self.assertEqual((m.discount, m.options.discounting), (0.999, "second"))
            self.assertEqual(m.discountFactor(100), 0.999 ** 100, cls.__name__)
            self.assertEqual(m.discountFactor(0), 1.0)

    def test_a_discount_given_to_the_model_wins_over_the_options(self):
        d = write_case(self.dir, (0,))
        files = dict(agentsProfileName=str(d / "agents.csv"), nodesdbFile=str(d / "nodes.csv"), linksdbFile=str(d / "links.csv"),
                     transLinkdbFile=str(d / "actions.csv"), transNodedbFile=str(d / "transitions.csv"), meanRayleigh=10)
        m = QLearning(options=ModelOptions(), discount=0.5, **files)
        self.assertEqual(m.discountFactor(2), 0.25)

    def test_a_decision_is_discounted_once_whatever_the_time_it_took(self):
        for cls in (SARSA, QLearning):
            m = self.model(cls)
            s0, s1 = self.experience(m, [(0, 0, 0), (1, 0, 1000)])
            m.stateMat[s0, 11], m.stateMat[s1, 11] = 0.0, 100.0
            m.tdControl(0)
            # target = -1000 + 0.9*100 = -910;  Q = 0 + 0.05*(-910) = -45.5
            self.assertAlmostEqual(m.stateMat[s0, 11], -45.5, places=9, msg=cls.__name__)

    def test_a_second_is_discounted_when_asked(self):
        m = self.model(QLearning, options=ModelOptions(discounting="second", discount=GAMMA))
        s0, s1 = self.experience(m, [(0, 0, 0), (1, 0, 10)])
        m.stateMat[s0, 11], m.stateMat[s1, 11] = 0.0, 100.0
        m.tdControl(0)
        # target = -10 + 0.9**10 * 100;  Q = 0.05 * target
        self.assertAlmostEqual(m.stateMat[s0, 11], 0.05 * (-10 + GAMMA ** 10 * 100), places=9)

    def monte_carlo_values(self, mode):
        m = self.model(MonteCarlo, options=ModelOptions(discounting=mode, discount=GAMMA))
        s0, s1, s2 = self.experience(m, [(0, 0, 0), (1, 1, 100), (2, 0, 160)])
        m.updateValuefunctionByAgentV2(0)
        return m.stateMat[s0, 11], m.stateMat[s1, 12], m.stateMat[s2, 11]

    def test_monte_carlo_return_discounted_per_second(self):
        # shelter: G = 1e5;  Q = 0.5 + 0.05*(G - 0.5)
        # node 1 (60 s later): G = 1e5*0.9^60 - 60;  node 0 (100 s earlier): G = G*0.9^100 - 100
        for mode in ("second", "method"):
            g2 = SURVIVE
            g1 = g2 * GAMMA ** 60 - 60
            g0 = g1 * GAMMA ** 100 - 100
            q0, q1, q2 = self.monte_carlo_values(mode)
            self.assertAlmostEqual(q2, 0.5 + ALPHA * (g2 - 0.5), places=6, msg=mode)
            self.assertAlmostEqual(q1, 0.5 + ALPHA * (g1 - 0.5), places=6, msg=mode)
            self.assertAlmostEqual(q0, 0.5 + ALPHA * (g0 - 0.5), places=6, msg=mode)

    def test_monte_carlo_return_discounted_per_decision(self):
        g2 = SURVIVE
        g1 = g2 * GAMMA - 60          # 89940
        g0 = g1 * GAMMA - 100         # 80846
        q0, q1, q2 = self.monte_carlo_values("decision")
        self.assertAlmostEqual(g0, 80846.0, places=6)
        self.assertAlmostEqual(q1, 0.5 + ALPHA * (g1 - 0.5), places=6)
        self.assertAlmostEqual(q0, 0.5 + ALPHA * (g0 - 0.5), places=6)


if __name__ == "__main__":
    unittest.main()
