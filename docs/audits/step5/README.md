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

## Independent review

A reviewer who had not seen the work read the commit, built the package, ran the quick start and read the documents against the code
(read-only). Twelve findings, two of them high. Every one was checked before it was acted on:

| # | Finding | Verdict | What was done |
|---|---|---|---|
| 1 | `pip install -e .` as the README says, then `evacrl-casebuild --help` and the first quick-start line fail (SciPy); CI and the audit script hid it by installing SciPy and pandas | **confirmed** (high) | SciPy in the core dependencies (finding 2 above); the CI core job is now the bare `pip install -e .` and runs both console scripts; the audit script installs nothing else and no longer pipes the output of the command that failed |
| 2 | Python 3.10 claimed as tested, red at HEAD; the old shortest-path figures of D9 not reproducible with a changed table | **confirmed** (high) | findings 1 and 3 above; the shortest-path figures of D9 measured again (below) |
| 3 | "Every command writes a manifest.json": false for `compare`, `policy` | **confirmed** | README and `evacrl.experiment` docstring name `sp`, `calibrate`, `evaluate` |
| 4 | Headline results selectively worded: "within 2 %" came from 30-minute training while "89 % / 82 %" came from 2-hour training; the 82 % policy is 13 % behind at 30 min; `kochi2` is not shipped | **confirmed** | README says which training each number is from, gives the 74 % of the 30-minute training and says `kochi2` is read from the study's repository |
| 5 | CHANGELOG: promises dates, says "bit for bit" for `legacy()` (only SARSA and Monte Carlo are pinned), omits that `run_ql*` results changed (real Q-learning), says "temporal-difference methods" for a change that includes Monte Carlo | **confirmed** (4 of 4) | corrected |
| 6 | Licence page: incomplete (the 3,086-row shelter shapefile with names and phone numbers, the other OSM-derived folders, the area polygons, basemaps in notebooks), wrong about "or any later version", "large local GIS data not tracked" false (277 MB are), a Dropbox "conflicted copy" file | **confirmed** | rows added, sentence reworded, repository.md corrected, the conflicted copy (byte-identical to `Censo_Code173.csv`) removed. **Not changed:** personal paths in legacy scripts and notebooks (`/home/emas/...`, `/Users/...`), and e-mail addresses in the git history |
| 7 | `test_docs.py` checked only part of the Markdown files; headings with code gave the wrong anchor | **confirmed** (latent) | walks every Markdown file outside `datasets/`; anchors computed from the heading as written, tested through `anchors()`. Known limits, kept: the option check is a substring match on the help text, ignores short options and sees only fenced commands at the start of a line |
| 8 | An installed wheel run outside a clone resolved `cases/`, `figures/`, `weights/` inside the interpreter's library folder | **confirmed** | the fallback is the current folder; the case-lookup error names `EVACRL_ROOT` |
| 9 | Stale lines in repository.md (byte-identical recordings, test list, "most methods are stubs": 15 of 53 function bodies are, "four" files of another author: five) | **confirmed** | corrected. **Not changed:** process wording in `roadmap.md` and the audits ("awaiting your confirmation", "decisions for you"), which reads oddly in a public repository |
| 10 | Quick start about 13 minutes, writes `runs/` untracked | **confirmed** | shortened to what was timed (about 6 minutes), `/runs/` ignored |
| 11 | The sdist has tests but not their fixtures, and no changelog | **confirmed** | `MANIFEST.in`: no tests, cases or docs; the changelog, citation and licence in. The wheel was already clean (41 files) |
| 12 | `clean_checkout.sh` clones the origin, not the local HEAD | **confirmed** | it refuses to run if they differ |
| | Could not verify: the GitHub runner (libGL for OpenCV) | **now verified** | CI runs on `ubuntu-latest` |
| | Could not verify: Office "public domain", OSM download dates, basemaps in notebooks | open | in the licence page |
| | `CITATION.cff` named one author for the 2024 study's repository | **removed** | authorship of that repository is not mine to state |

## The shortest-path figures of D9, measured again

After finding 1 the shortest-path baseline of `kochi_area4` was run again (`crowded_remeasure.py`, same seeds as D9). The change is under one standard deviation:

| | before | with the 0.2.0 tie-break |
|---|---:|---:|
| safe at 30 min, 10 runs | 7,517 ± 89 (55.7 %) | 7,507 ± 96 (55.6 %) |
| 5 runs of 2 h: safe at 60 min | 9,377 | 9,483 |
| everybody safe by | 5,781 ± 94 s | 5,828 ± 73 s |
| learned `link` / `segment` policy, walk against the shortest path; first choice equal | +0.4 % / +1.8 %; 88 % / 87 % | the same |

The D9 text and the README quote the new values. The conclusion is unchanged: learning does not beat the shortest path on this area within the
budget tried.

## Result of the checks

* Local, on the final tree: 309 tests with the geospatial packages and 277 without (1 skipped), `python -W error`, Python 3.11; `ruff` clean.
* GitHub Actions on the pushed branch: lint, build (with the wheel installed and run away from the checkout), and the tests on Python 3.10, 3.11, 3.12 and 3.13
  in both configurations: first run (before the fixes) 7 of 10 jobs green and the 3 red ones were the two defects above; after the fixes 10 of 10 on two
  successive commits.
* The clean-checkout script on the final commit (`49e6835`, below).

## Clean checkout of the final commit

`clean_checkout.sh` on `49e6835` (a fresh clone of the pushed branch, a new environment per Python version, nothing installed beyond the README's install):

| | 3.10 | 3.11 | 3.12 | 3.13 |
|---|---|---|---|---|
| `pip install -e .`, `python -W error -m unittest discover tests` | 275 tests OK (2 skipped) | 275 OK (2) | 275 OK (2) | 275 OK (2) |
| `pip install -e ".[casebuild]"`, same | 309 OK | 309 OK | 309 OK | 309 OK |

`ruff` clean, `CITATION.cff` valid, wheel and sdist build and pass `twine check`; the wheel installed in another environment imports from `site-packages`,
`evacrl-casebuild` and `evacrl-experiment` answer `--help`, and `validate cases/kochi_area2` reports the case valid. The quick start of the README, run
with that wheel from the clone: `sp` 26 s, `calibrate` 331 s, `evaluate` 17 s, `compare` and `policy` 1 s each (about 6 minutes, 4 cores). GitHub Actions on the same
commit: all jobs green.

## Decisions for you

| | Decision | Recommendation |
|---|---|---|
| D12 | Version **0.2.0** (the defaults changed, so results differ from 0.1.0) and the licence **`GPL-3.0-only`** (the repository carries the GPLv3 text and no "or later" notice; widening it later is possible, narrowing is not) | both as they are |
| D13 | The Kochi Prefectural Office layers (shelters, census mesh) are recorded as "public domain, as stated by you"; Japanese open data is often CC BY-like and then asks for attribution | look up the Office's terms and put the wording in [docs/data-licences.md](../../data-licences.md) before the repository is shared |
| D14 | Rows of the licence page that say "source and terms not recorded" (census and building databases, tsunami inundation, shelter register, area polygons) and the five files in `pre/` that name Moya as author: their agreement to GPL-3.0 | fill in, or remove the data / files that cannot be cleared |
| D15 | 277 MB of GIS data are tracked (three files over 65 MB; a clone is about 100 MB compressed). Moving them out needs a history rewrite to help, which I did not do | leave, or move to a release asset / Git LFS and say so in the README |
| D16 | `experimental/` (an unfinished rewrite), `pre/`, `variants/app_2022/` are labelled legacy, nothing deleted | label (done); delete `experimental/` if you do not mean to continue it |
| D17 | `CITATION.cff` has one author, no affiliation, no ORCID and no paper (`preferred-citation`) | add them; I did not invent any |
| D18 | Process wording in `roadmap.md` and the audits; personal paths in legacy scripts and notebooks; the e-mail addresses in the git history | a tidy-up commit in Step 6 if you want the repository to read as a finished release |
| D11 | (from Step 4, not yet answered) the 2024 Kochi study's conclusion about learned policies was made under the 0.9-per-decision discount that capped them at 81 %: re-examine it before it is cited | yes |

D10 (a long training run on `kochi_area4`) was declined (your choice (a)).
