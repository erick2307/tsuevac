# -*- coding: utf-8 -*-
"""Case selection for the entry-point scripts (scripts/main_*.py).

Each script defines one runner per study area; `run_case` picks the one named on the command line
(`python scripts/main_ql_mod.py kochi`) or the script's default, and explains what is wrong when the
case is unknown or its data is not part of this repository.
"""
import sys
from pathlib import Path

from evacrl import paths


def available_cases():
    """Names of the folders in paths.CASES_DIR that hold a `data/` folder."""
    if not paths.CASES_DIR.is_dir():
        return []
    return sorted(p.name for p in paths.CASES_DIR.iterdir() if (p / "data").is_dir())


def run_case(runners, default, argv=None, generic=None):
    """Call `runners[name]()` for the case in `argv[0]` (default: sys.argv[1:]) or else `default`. For a name without a runner
    that is a folder of `cases/` with a `data/` folder, call `generic(name)` if there is one."""
    argv = sys.argv[1:] if argv is None else list(argv)
    name = argv[0] if argv else default
    script = Path(sys.argv[0]).name or "script.py"
    if name not in runners and generic is not None and (paths.case_dir(name) / "data").is_dir():
        return generic(name)
    if name not in runners:
        extra = " (or, for a folder of cases/ with a data/ folder, its name)" if generic is not None else ""
        raise SystemExit(f"unknown case '{name}'; {script} defines: {', '.join(sorted(runners))}{extra}")
    if not (paths.case_dir(name) / "data").is_dir():
        hint = "" if argv else f" (it is the default of {script})"
        if not available_cases():
            raise SystemExit(
                f"no cases found in {paths.CASES_DIR}. Run from inside the repository, or set EVACRL_ROOT to the "
                f"repository folder (needed when evacrl is installed with a plain `pip install`)."
            )
        raise SystemExit(
            f"case '{name}'{hint} has no data in {paths.case_dir(name) / 'data'}.\n"
            f"Cases available here: {', '.join(available_cases()) or 'none'}. "
            f"Pass one as an argument, e.g.: python {script} {(available_cases() or ['kochi'])[0]}"
        )
    return runners[name]()
