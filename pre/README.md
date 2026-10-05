# pre/

**Legacy (2021) pre-processing, not tested as a whole and not linted.** Building a case is now `python -m evacrl.casebuild`
([Manual](../docs/manual.md#building-a-case)); these scripts remain because the old Kochi case (`cases/kochi`) was made with them.

| Script | What it does | Still used by |
|---|---|---|
| `SetPopDB.py` | builds `cases/<area>/data/agentsdb.csv` from the census databases in `datasets/census/` and the case's `nodesdb.csv` | `cases/kochi` (covered by `tests/test_data_provenance.py`) |
| `DisaggregationLibrary.py`, `SetDatabaseBldMeshCodes.py` | population disaggregation: census mesh to a synthetic population in buildings ([notes](../docs/preprocessing.md)); they produced the databases in `datasets/census/` | nothing: `SetPopDB.py` reads those databases, it does not call them |
| `defPathsFromNodes.py` | the next node on the shortest path from every node (the legacy way to make `nextnode.csv`) | replaced by `evacrl.casebuild.routing` |
| `DetectionShelters.py`, `makeUniformPedestrianProfiles.py`, `tests.ipynb` | shelter detection from a network, one agent per node, a scratch notebook | nothing |

Five of these files name another author in their header (`@author: Moya`, `@author: luismoya`): see
[docs/data-licences.md](../docs/data-licences.md#code-written-by-others). `ShelterCoordinates.csv` is a shelter register whose source is not
recorded (same page).
