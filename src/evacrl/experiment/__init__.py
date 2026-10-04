# -*- coding: utf-8 -*-
"""Experiments on a case: repeated shortest-path runs with a convergence rule, training with checkpoints evaluated greedily, runs of
a stored policy, comparison plots, run manifests, and a command line (`python -m evacrl.experiment`).

    from evacrl.experiment import Case, repeat_shortest_path, calibrate, evaluate_policy

Every run takes its seed from a base seed (`derive_seeds`), so an experiment is reproducible, and gives the same results for any
number of worker processes. See docs/manual.md, "Experiments".
"""
from evacrl.experiment.training import CalibrationResult, Checkpoint, calibrate, epsilon_at, evaluate_policy
from evacrl.experiment.case import Case
from evacrl.experiment.convergence import Convergence, convergence_trace
from evacrl.experiment.manifest import build_manifest, write_manifest
from evacrl.experiment.policy import PolicyComparison, compare_with_shortest_path, greedy_next_nodes, walks
from evacrl.experiment.runs import METHODS, RunResult, evaluate_state, make_model, run_episode, shortest_path_run
from evacrl.experiment.seeds import derive_seeds
from evacrl.experiment.sp import ShortestPathResult, repeat_shortest_path

__all__ = [
    "CalibrationResult", "Case", "Checkpoint", "Convergence", "METHODS", "PolicyComparison", "RunResult", "ShortestPathResult", "build_manifest", "calibrate",
    "compare_with_shortest_path", "greedy_next_nodes", "walks",
    "convergence_trace", "derive_seeds", "epsilon_at", "evaluate_policy", "evaluate_state", "make_model", "repeat_shortest_path",
    "run_episode", "shortest_path_run", "write_manifest",
]
