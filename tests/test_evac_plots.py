#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of evacrl.evac_plots.plotSurvivors, which works on a survivor-time table
(results/survivor-time_<alias>.csv: column 0 = time in s, column k+1 = survivors of simulation k).

They guard two bugs found by running the function:
* it used `startfile`, defined only in the script's __main__ block, so importing it gave a NameError;
* the reported maximum was read from the wrong column (the time column is column 0).

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
sys.path.insert(0, str(REPO / "src"))  # the only place that knows where the code lives

try:
    from evacrl import evac_plots  # noqa: E402  (needs scipy and pandas)
except ImportError as exc:
    raise unittest.SkipTest(f"evacrl.evac_plots needs the 'plots' extra ({exc}); pip install -e \".[plots]\"")


class DepartureCurve(unittest.TestCase):
    def test_the_reference_curve_is_the_distribution_the_engine_draws_departures_from(self):
        from evacrl.evac_plots import departureCDF
        self.assertAlmostEqual(float(departureCDF(15.0, 15.0)), 1 - np.exp(-np.pi / 4), places=12)     # F(mean) of a Rayleigh with scale mean*sqrt(2/pi)
        x = np.linspace(0, 200, 200001)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        self.assertAlmostEqual(float(trapezoid(1 - departureCDF(x, 15.0), x)), 15.0, places=2)         # the mean of the distribution is the mean given

    def test_the_engines_departure_times_follow_it(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evacrl.evac_plots import departureCDF
        from evacrl.experiment import Case, make_model
        from test_experiment import write_case
        with tempfile.TemporaryDirectory() as tmp:
            case = Case.load(write_case(Path(tmp) / "tiny", starts=[0] * 4000))
            np.random.seed(0)
            model = make_model(case, "sarsa", None, 1.0)                                               # mean departure 1 minute
        left_within_a_minute = float(np.mean(model.pedDB[:, 9] <= 60))
        self.assertAlmostEqual(left_within_a_minute, float(departureCDF(1.0, 1.0)), delta=0.02)        # 0.544; with scale = mean it would be 0.393


class PlotSurvivors(unittest.TestCase):
    POP = 100

    def run_in_case_folder(self, table):
        """plotSurvivors reads and writes results/ relative to the case folder it is run from."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.mkdir("results")
                np.savetxt(Path("results") / "survivor-time_t.csv", table, delimiter=",", fmt="%d")
                maxEvac = evac_plots.plotSurvivors(numfiles=table.shape[1] - 1, simtime=3, pop=self.POP,
                                                   meandeparture=2, allfiles=True, casealias="t")
                files = sorted(os.listdir("results"))
            finally:
                os.chdir(cwd)
        return maxEvac, files

    def table(self, finals):
        """3 time steps; simulation k ends with finals[k] survivors (and is lower before)."""
        t = np.array([[0], [60], [120]])
        sims = np.array([[0 for _ in finals], [f // 2 for f in finals], list(finals)])
        return np.hstack([t, sims])

    def test_importable_use_does_not_need_a_script_global(self):
        maxEvac, files = self.run_in_case_folder(self.table([20, 90, 40]))
        self.assertIn("evacuation_rate_t.png", files)
        self.assertIn("evacuation_per_episode_t.png", files)

    def test_reported_maximum_is_the_survivors_of_the_best_simulation(self):
        # simulation 1 is the best in every time step -> its final value (90), not the time (120) or sim 0's (20)
        maxEvac, _ = self.run_in_case_folder(self.table([20, 90, 40]))
        self.assertEqual(maxEvac, 90)

    def test_reported_maximum_never_exceeds_the_population(self):
        maxEvac, _ = self.run_in_case_folder(self.table([100, 10, 10]))
        self.assertLessEqual(maxEvac, self.POP)
        self.assertEqual(maxEvac, 100)


if __name__ == "__main__":
    unittest.main()
