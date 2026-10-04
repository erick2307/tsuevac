# -*- coding: utf-8 -*-
"""A case as the experiments see it: the files the model is built from, found by name or by folder."""
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from evacrl import paths

TABLES = ("agentsProfileName", "nodesdbFile", "linksdbFile", "transLinkdbFile", "transNodedbFile")


@dataclass(frozen=True)
class Case:
    """`name`, the model's input files (`files`: the keyword arguments of the model classes) and the shortest-path table."""
    name: str
    files: tuple            # ((keyword, path), ...): hashable, so a case can be passed between processes and compared
    nextnode: Optional[str] = None

    @property
    def kwargs(self):
        return dict(self.files)

    @classmethod
    def load(cls, spec, popfile=1):
        """`spec`: the name of a case under `cases/` (`kochi_area2`), or a folder. A folder holds either `data/` with
        `agentsdb.csv, nodesdb.csv, linksdb.csv, actionsdb.csv, transitionsdb.csv` (+ `nextnode.csv`), the layout of `cases/`, or
        the files of the 2024 Kochi study side by side (`population_<popfile>.csv, nodes.csv, edges.csv, actionsdb.csv,
        transitionsdb.csv, nextnode.csv`)."""
        folder = os.fspath(spec) if os.path.isdir(os.fspath(spec)) else str(paths.case_dir(spec))
        data = os.path.join(folder, "data")
        if os.path.isfile(os.path.join(data, "nodesdb.csv")):
            names = ("agentsdb.csv", "nodesdb.csv", "linksdb.csv", "actionsdb.csv", "transitionsdb.csv")
            root, next_file = data, "nextnode.csv"
        elif os.path.isfile(os.path.join(folder, "nodes.csv")) and os.path.isfile(os.path.join(folder, "edges.csv")):
            names = (f"population_{popfile}.csv", "nodes.csv", "edges.csv", "actionsdb.csv", "transitionsdb.csv")
            root, next_file = folder, "nextnode.csv"
        else:
            raise FileNotFoundError(f"{spec!r}: no case here (looked for {data}/nodesdb.csv, and for nodes.csv and edges.csv in {folder})")
        files = tuple((key, os.path.join(root, name)) for key, name in zip(TABLES, names))
        for _, path in files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"the case {spec!r} has no {path}")
        nextnode = os.path.join(root, next_file)
        return cls(name=os.path.basename(os.path.abspath(folder)), files=files, nextnode=nextnode if os.path.isfile(nextnode) else None)

    def checksums(self):
        """sha256 of every input file, by file name."""
        out = {}
        for path in [p for _, p in self.files] + ([self.nextnode] if self.nextnode else []):
            with open(path, "rb") as f:
                out[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
        return out
