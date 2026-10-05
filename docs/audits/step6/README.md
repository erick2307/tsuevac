# Step 6 audit (in progress)

## The 2024 study's conclusions (D11)

## D15: smaller clones

Where the size is (measured on the repository as pushed, `shrink_history.sh`):

| | download | on disk (files) |
|---|---:|---:|
| full clone, before | 154 MB | 343 MB |
| `git clone --depth 1`, before | 61 MB | 343 MB |
| full clone, after the GIS layers left the tip (this commit) | unchanged until the history is rewritten | about 66 MB |
| full clone, after the history is rewritten (below) | **54 MB** | about 66 MB |

The GIS layers (277 MB) are mostly zeros in a pack (13 MB compressed), so removing them from the tip shrinks the *checkout* by 277 MB but the *download* little. What
weighs in the download is history: files that were deleted long ago and are still in it (two AVI videos 51 MB, the 2021 Arahama and Kochi state dumps about 50 MB).

**Done now, without touching history:** `datasets/gis/data/qgis` and `qgis_1` are no longer tracked and are ignored (`datasets/gis/README.md` says how to restore them; an 8 MB
backup archive was sent to you); `README.md` tells how to make a light clone (`--depth 1`).

**Prepared, not applied: rewriting the history** (`shrink_history.sh`). It removes from all four branches (`main`, `dev`, `SPvsRL`, `regid/trusting-curie-ogh40k`) the files listed in it,
leaves the tip of every branch identical apart from those paths (checked branch by branch: 1,238 to 1,401 files each, no difference) and shrinks a full clone from 153 MiB
to 54 MiB. It is **not pushed**: it needs a force-push to branches that are not this session's. What a push does:

* every commit of every branch gets a new hash; clones and forks have to be made again, and open pull requests and links to old commits break;
* the old objects stay reachable on GitHub by their old hashes (and in forks) until GitHub collects them; only GitHub Support can purge them early;
* the e-mail addresses in the author fields of the history stay (changing them would rewrite attribution, which is a separate choice);
* the removed files are gone from the repository: keep the backup archive (the GIS layers) and, if the Arahama state dumps matter, a copy of the old history.

To apply it, say so, or run `sh docs/audits/step6/shrink_history.sh --push` yourself with `git-filter-repo` installed.

