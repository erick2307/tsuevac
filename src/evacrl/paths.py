# -*- coding: utf-8 -*-
"""Where things live: the only module that knows the repository layout.

Everything that reads case inputs or writes run outputs resolves its location through
this module instead of through the current working directory. Moving a folder then means
changing the constants below (and nothing else).
"""
import os
from pathlib import Path

def _find_repo_root():
    """The repository: $EVACRL_ROOT, else the checkout this file lives in (source tree or editable
    install), else the repository the program is run from (a regular `pip install`)."""
    override = os.environ.get("EVACRL_ROOT")
    if override:
        return Path(override).resolve()

    def is_root(folder):
        return (folder / "pyproject.toml").is_file() and (folder / "cases").is_dir()

    for folder in Path(__file__).resolve().parents:
        if is_root(folder):
            return folder
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if is_root(folder):
            return folder
    return Path(__file__).resolve().parents[2]  # not found: src/evacrl/paths.py -> repository root


REPO_ROOT = _find_repo_root()

# Study areas (kochi, new_kochi, arahama, ...) are folders directly under CASES_DIR; each holds
# `data/` (inputs, tracked) and `state_<name>/` (run outputs, not tracked).
CASES_DIR = REPO_ROOT / "cases"

FIGURES_DIR = REPO_ROOT / "figures"  # snapshots used to build videos (not tracked)
WEIGHTS_DIR = REPO_ROOT / "weights"  # link weights exported while learning
RESULTS_DIR = REPO_ROOT / "results"  # analysis outputs shared by the notebooks
CENSUS_DIR = REPO_ROOT / "datasets" / "census"  # census, household and building databases


def case_dir(area):
    """Folder of a study area; `area` is a case name ('kochi') or an absolute path."""
    area = Path(area)
    return area if area.is_absolute() else CASES_DIR / area


def case_path(area, *parts):
    """`os.path.join(case_dir(area), *parts)`, as a string."""
    return os.path.join(case_dir(area), *parts)
