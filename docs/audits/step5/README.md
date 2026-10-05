# Step 5 audit: release engineering

What a stranger gets when they clone the repository: install, lint, tests on four Python versions, the quick start, the wheel,
the documentation, the licence and the data attribution. The audit is `clean_checkout.sh` (a fresh clone, a new virtual
environment per Python version) and the GitHub Actions run of `.github/workflows/ci.yml`; both found defects, which are fixed.

```
sh clean_checkout.sh [REPO] [REF]       # PYVERS="3.10 3.11 3.12 3.13" by default; about 15 min on 4 cores
python regenerate_nextnode.py [--write] # the shortest-path tables, see finding 1
```

## What was done

| Part | Result |
|---|---|
| Lint | `ruff check src tests scripts` (pyflakes errors and syntax errors) clean: 13 findings fixed (unused imports and variables; the one random draw in `main_ShortPath.py` is kept, it advances the seeded stream). The pre-processing scripts, `variants/`, `experimental/`, the notebooks and `datasets/` are not linted (status table in [docs/repository.md](../../repository.md)) |
| Package | `pyproject.toml`: version from `evacrl.__version__` (0.2.0), `GPL-3.0-only` (PEP 639), authors, URLs, classifiers, a `dev` group. `python -m build` makes a wheel and an sdist and `twine check` passes on both |
| Citation | `CITATION.cff` (validated with `cffconvert`) |
| Changes | `CHANGELOG.md`: what 0.2.0 changed, with the defaults that make its results differ from 0.1.0 |
| Licence and data | [docs/data-licences.md](../../data-licences.md): every data folder, what the repository records about its source and what it does not; code of other authors |
| Documentation | new `README.md` (what it is, what the results say so far, install, quick start, documentation index); the layout moved to [docs/repository.md](../../repository.md) with a status table; `tests/test_docs.py` |
| Stubs | `experimental/`, `pre/` and `variants/app_2022/` carry a `README.md` that says they are legacy or unmaintained, not tested, not linted. Nothing was deleted |
| CI | lint + citation check, tests on Python 3.10, 3.11, 3.12 and 3.13 with the core install and with the `casebuild` extra (`python -W error -m unittest discover tests`), build + `twine check` |

`tests/test_docs.py` (8 tests): every relative link of every Markdown file leads to a file and every `#anchor` to a heading, every command of
`python -m evacrl.casebuild` and `python -m evacrl.experiment` answers `--help` and appears in the manual, and every option used in a
command in the README, the manual and the other documents exists. Mutation-checked: a broken file link, a broken anchor, a wrong option, a command
missing from the manual and a command that does not exist are each caught (5 of 5); the first anchor mutation was a no-op (the README has no anchor
link), repeated on a real one.

## Findings of the clean checkout and of CI, all fixed

1. **The shortest-path table depended on the SciPy version.** `kochi_area4` rebuilt on Python 3.10 (SciPy 1.15.3) differed from the shipped table
   (made with SciPy 1.18) in 1 entry of 2,220: where two walks are equally short, `scipy.sparse.csgraph.dijkstra` decides, and its
   choice changed between versions. Both choices are shortest walks, but "the shortest-path baseline of a case" must not depend on a library
   version. `next_nodes` now takes the lowest-numbered neighbour on a shortest walk (`routing._lowest_step_on_a_shortest_walk`), whatever the
   order of the links and the SciPy version. `regenerate_nextnode.py` rewrote the shipped tables after checking that every entry that changes
   is a step between two equally short options (`regenerate_nextnode.log`):

   | case | nodes | entries changed |
   |---|---:|---:|
   | `kochi_area0` | 537 | 3 |
   | `kochi_area1` | 423 | 3 |
   | `kochi_area2` | 297 | 0 |
   | `kochi_area4` | 1,110 | 13 |

   Three new tests (ties towards the lowest number whatever the link order, equally near shelters, and on random networks with many ties
   every walk of the table is as long as the shortest) and a mutation check of the new function (5 mutants: 4 killed, 1 equivalent and
   removed). On Python 3.10 `kochi_area4` now rebuilds to the shipped table. `method="allpairs"` (the 2024 study's way) keeps SciPy's ties, by
   design: it exists to reproduce that study's tables.
2. **`evacrl.casebuild` did not import after `pip install evacrl`**: SciPy was not a dependency of the core install, while the README
   and the manual lead with `python -m evacrl.casebuild`. SciPy is now a core dependency.
3. **The geospatial tests errored on Python 3.10 and 3.11 (13 tests)** under `-W error`: pyproj warns, on every coordinate transformation, of a
   NumPy 1.25 deprecation. Not ours; the test module filters that one message. Found by CI (on 3.11, where my local environment had a newer pyproj).
4. **The quick start of the first README took 14 minutes** on 4 cores; shortened, and its commands are run by the audit.
5. CI actions moved to the Node 24 versions (`checkout@v5`, `setup-python@v6`), after a warning in the first run.

## Not changed, for you

See "Decisions for you" at the end (to be completed with the independent review).
