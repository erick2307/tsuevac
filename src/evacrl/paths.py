# -*- coding: utf-8 -*-
"""Where things live: the only module that knows the repository layout.

Everything that reads case inputs or writes run outputs resolves its location through
this module instead of through the current working directory. Moving a folder then means
changing the constants below (and nothing else).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/evacrl/paths.py -> repository root

# Study areas (kochi, new_kochi, arahama, ...) are folders directly under CASES_DIR; each holds
# `data/` (inputs, tracked) and `state_<name>/` (run outputs, not tracked).
CASES_DIR = REPO_ROOT / "cases"

FIGURES_DIR = REPO_ROOT / "figures"  # snapshots used to build videos (not tracked)
WEIGHTS_DIR = REPO_ROOT / "weights"  # link weights exported while learning
RESULTS_DIR = REPO_ROOT / "results"  # analysis outputs shared by the notebooks


def case_dir(area):
    """Folder of a study area; `area` is a case name ('kochi') or an absolute path."""
    area = Path(area)
    return area if area.is_absolute() else CASES_DIR / area


def case_path(area, *parts):
    """`os.path.join(case_dir(area), *parts)`, as a string."""
    return os.path.join(case_dir(area), *parts)
