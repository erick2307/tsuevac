# -*- coding: utf-8 -*-
"""Run manifests: what is needed to repeat an experiment and to know what produced a result."""
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from evacrl import paths


def _git():
    """Commit of the repository the code is run from (and whether it has uncommitted changes), or None outside a checkout."""
    try:
        run = lambda *args: subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return dict(commit=run("rev-parse", "HEAD"), uncommitted_changes=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception:
        return None


def _version():
    try:
        from importlib.metadata import version
        return version("evacrl")
    except Exception:
        return "source tree"


def _plain(value):
    """JSON-able copy of a value (dataclasses, NumPy numbers and arrays, tuples)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None                                  # JSON has no NaN or infinity
    return value


def build_manifest(kind, case, parameters, *, options=None, seeds=None, results=None, started=None, argv=None):
    """The manifest of one experiment (`kind`: "shortest_path", "calibration", "evaluation"): the case and the checksum of its files,
    the model options, every parameter, the seeds, the results summary, versions and timing. `started` is `time.time()` at the start."""
    out = dict(kind=kind, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               seconds=None if started is None else round(time.time() - started, 1),
               case=dict(name=case.name, files=case.checksums()),
               options=_plain(options) if options is not None else None, parameters=_plain(parameters), seeds=_plain(seeds),
               results=_plain(results), command=" ".join(argv if argv is not None else sys.argv),
               software=dict(evacrl=_version(), git=_git(), python=platform.python_version(), numpy=np.__version__,
                             platform=platform.platform()))
    return out


def write_manifest(folder, manifest, name="manifest.json"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return path
