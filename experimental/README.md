# experimental/

**Not maintained. Not tested, not linted, not part of the package. Do not build on it.**

* `new_model/`: an unfinished object-oriented rewrite (`tsuevac` package: `Environment`, `Agent`, `Evacuee`, `Node`, `Shelter`,
  `Model`). Several methods are stubs. The maintained engine is `src/evacrl/core.py`.
* `tdcontrol.py`: a toy temporal-difference control skeleton.
* `tests_mc.py`: ad-hoc runs of `MonteCarlo` (Arahama sequences, a shortest-path run, a video), not unit tests. The tests of the
  package are in `tests/`.

They are kept for their history. If one of them is wanted again, it should be brought into `src/evacrl` with tests first.
