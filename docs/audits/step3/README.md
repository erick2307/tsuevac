# Step 3 audit: the case-building pipeline

`evacrl.casebuild` rebuilds, tests and extends the pipeline of the 2024 Kochi study
([erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara), `EVACMODEL3_FocalPoints/preprocess.py` and the notebook that
calls it). Its stages are checked against that study's committed tables, case by case, and the tables of the study are checked with
the new validator. The scripts here regenerate every number below.

```
git clone https://github.com/erick2307/2024_urushibara ~/2024_Urushibara     # or set URUSHIBARA_DIR=<its EVACMODEL3_FocalPoints folder>
pip install -e ".[casebuild]"                       # osmnx < 2, geopandas, shapely, scipy (+ the base requirements)
cd docs/audits/step3
python legacy_exactness.py        # A1  core only
python validate_committed.py      # A6  core only
python geo_checks.py              # A2-A4  needs the casebuild extra, works offline
python effect_on_results.py 30    # A5  about 6 min on 4 cores
python shelter_access.py 20       # A7  snapped against attached shelters, four areas, about 25 min on 4 cores
```

**What could not be tested here.** The sandbox cannot reach OpenStreetMap, so `download_network` (one call to
`osmnx.graph_from_polygon`, the call the study made) was not run against the live service. Everything after the download is tested
from stored graphs: the study's own `Graph/Gnodes.geojson` and `Gedges.geojson` of six areas, and a synthetic graph in the unit tests.

## What was checked, and the result

| | Stage | Reference | Result |
|---|---|---|---|
| A1 | `merge_short_links(method="legacy")` | the study's `nodes.csv`, `edges.csv` from its `nodes0.csv`, `edges0.csv` | **identical**, 5 of 5 areas |
| A1 | `actions_and_transitions` | its `actionsdb.csv`, `transitionsdb.csv` | **identical**, 5 of 5 |
| A1 | `next_nodes` (all pairs, last parallel link; and multi-source) | its `nextnode.csv` | **identical**, 5 of 5, both methods |
| A1 | `actions_and_transitions` | tsuevac's own `cases/kochi` (made by another script, years earlier) | **identical** |
| A2 | stored OSM graph → project → undirected → raw network | the study's raw tables, compared by OSM id | **identical**, 6 of 6: same nodes, coordinates to 1 mm, same links and lengths, same shelters |
| A3 | shelter points → nearest node, inside the network's box | the study's shelter nodes | **identical**, 6 of 6 |
| A4 | census population of an area, cells entirely inside | the totals printed by the study's notebook (2303, 1078, 622, 240, 12288) | **identical**, 5 of 5 |
| A6 | the validator on the study's tables | | finds the defects below |
| A7 | shelters attached by an access link, against snapped (the shipped cases use attach) | the same inputs built both ways | access links of 0–360 m; the walk to the nearest shelter grows by 22–57 m on average and the last evacuee is 25–56 s later (table below) |

The numbering of the nodes in a stored graph differs from the study's (the saved graph's node order is not the one the tables
were made from), so A2 compares the networks as graphs by OSM id; the shipped cases keep the numbering of their own `raw/`.

## What the validator finds in the 2024 tables (A6, and the clean-up)

| Area | Short links | Orphan rows (nodes with no link) | Chains of 3+ nodes | Nodes without a way out: 2024 / clusters | Agents at a shelter at t=0 | Agents at an *orphan* shelter | Next nodes off a shortest walk |
|---|---:|---:|---:|---:|---:|---:|---:|
| kochi0 | 16 | 16 | 1 | 16 / **0** | 1.3 % | 0 | 2 |
| kochi1 | 20 | 18 | 2 | 18 / **0** | 0.7 % | 0 | 0 |
| kochi2 | 16 | 16 | 1 | 16 / **0** | 2.3 % | 0 | 0 |
| kochi4 | 39 | 39 | 2 | 38 / **0** | 1.3 % | **9** | 16 |
| kochi42 | 33 | 33 | 2 | 32 / **0** | 1.9 % | **11** | 13 |

* **The short-link merge works on pairs.** For every short link it writes the merged position to a new row, and keeps the old row of the
  first node: one orphan node per short link, with the merged position that no link uses. A chain of short links is not merged: the
  nodes of the chain end up on different sides, cut off from each other (1–2 chains in each area). `merge_short_links` merges clusters
  instead: one node at the mean position, a shelter if any member is one, the other links re-attached, and none left over.
* **The orphans can be shelters.** In kochi4 and kochi42 a merged shelter leaves an orphan with the shelter flag, and the random
  population put 9 and 11 agents on it: they stand at an isolated "shelter" and count as evacuated at time 0. Overall 0.7–2.3 % of
  the agents of every area start at a shelter, a real or an orphan one, and count the same way; the builder no longer places agents
  at shelters unless asked.
* **Parallel links** between the same two nodes keep the *last* length in the 2024 shortest-path table, not the shortest: 2–16 next
  nodes per area are not on a shortest walk. The new table keeps the shortest.
* **The population file of `kochi4`** has 13,244 agents although the area's census total is 12,288, which is the number of agents of
  `kochi42`. The two folders do not come from the same run.

## The population (A4)

The study sums the census cells lying *entirely inside* an area. Cells on the boundary are lost, and the cells are large: the standard 500 m mesh (`MESH4_ID`, 9-digit code), about 460 m by 580 m at Kochi (an earlier version of this report said 250 m; that was wrong):

| Area | Study (`within`) | Boundary cells by area (`weighted`) | More |
|---|---:|---:|---:|
| 0 | 2,303 | 4,196 | +82 % |
| 1 | 1,078 | 2,743 | +155 % |
| 2 | 622 | 1,704 | +174 % |
| 3 | 240 | 546 | +128 % |
| 4 | 12,288 | 13,502 | +10 % |

(area-weighting assumes the people are spread evenly over a cell.) The smaller the area, the larger the error.

## The shelters (A3 and S2)

Shelter points (53 shelters and 339 evacuation buildings) are matched to the network of an area. The study snapped each one to the
nearest node, however far it is (it took every point inside the *box* of the network): **A3** reproduces that exactly
(`shelters="snap"`). Distances to the node: median 50–190 m. In area 1 two of the four points are more than 300 m from any node
(360 m at most), in area 4 six of 17 are more than 100 m: the model puts such a shelter on the edge of the area, so the walk from
there to the building is not part of the evacuation, and the street node that received it stops being a place people can pass
through (a shelter has no way out). Area 3 has no shelter inside its box at all.

**S2, built after your confirmation: attach.** `shelters="attach"` (the default now; `--shelters-as attach`) puts a node at the
shelter and joins it to the nearest street node by one link as long as the distance (whole metres, width 3 m). Points within 5 m of
each other are one shelter (in area 4 three points sit at the same place), a point closer than 1 m to a node is that node, and a shelter is never
attached to another shelter. The street node stays an ordinary node. Nothing is left out and nothing is moved. A7 compares the two
ways on the same inputs, with one agent at every start node (so that both variants have the same people on the same street
nodes, up to the few nodes the snapping turned into shelters), shortest path, 20 departure-time seeds, 120 min simulated:

| Area | Mode | Nodes | Shelters | Access link, m: median / max (over 100 m) | Walk to the nearest shelter, m: mean / max | Agents | Last evacuee, s | Safe at 30 min |
|---|---|---:|---:|---|---|---:|---:|---:|
| 0 | snap | 531 | 6 | 54 / 71 (snapped; 0) | 693 / 1,392 | 525 | 1,819 ± 93 | 99.8 % |
| 0 | attach | 537 | 6 | 54 / 70 (0) | 723 / 1,438 | 531 | 1,875 ± 94 | 99.8 % |
| 1 | snap | 419 | 3 | 189 / 360 (snapped; 2) | 471 / 1,477 | 416 | 1,805 ± 71 | 99.9 % |
| 1 | attach | 423 | 4 | 189 / 360 (2) | 528 / 1,513 | 419 | 1,861 ± 142 | 99.8 % |
| 2 | snap | 293 | 4 | 50 / 79 (snapped; 0) | 1,002 / 2,418 | 289 | 2,503 ± 97 | 84.9 % |
| 2 | attach | 297 | 4 | 49 / 78 (0) | 1,033 / 2,469 | 293 | 2,549 ± 83 | 83.4 % |
| 4 | snap | 1,095 | 15 | 60 / 221 (snapped; 6 of 17 points) | 1,361 / 4,558 | 1,080 | 4,171 ± 112 | 70.1 % |
| 4 | attach | 1,110 | 15 | 59 / 221 (4 of 15 shelters) | 1,383 / 4,586 | 1,095 | 4,196 ± 120 | 70.0 % |

(`shelter_access.py`; the "over 100 m" counts are of snapped *points* for snap and of *shelters* after merging for attach.) What it
shows: the walk to the nearest shelter grows by 22–57 m on average (the length of the access links of the shelters people go to) and
the last evacuee is 25–56 s later (0.6–3 %), 0.2–0.8 of a run-to-run standard deviation, and the share safe after 30 min moves by
0–1.5 percentage points. The effect is small on these networks and for this metric because a node pays only the access link of the
shelter it goes to, and that is short next to walks of 0.5–1.4 km on average (4.6 km at most). These are means over 20
seeds, not tested for significance; the two variants differ in a few agents (the street nodes that snapping turned into shelters).
Where it will matter is a shelter far from the network with many people near it, and a learned policy that has to choose between
shelters. Limits: the length is the straight line to the nearest *node*, not a walk (it can cross a river or a block), and it is
not the nearest point of the nearest street (that would split the street and is not done). `--max-snap-distance` still
removes shelters beyond a distance (in area 1, 100 m keeps 2 of the 4 points; in area 4, 11 of 17 points).

**Found by an independent review of the attach code, and fixed.** (1) The default clean-up merged an access link shorter than 5 m, which
turned the street node into the shelter and moved it (a shelter 1–4 m from a junction is common in a town; none of the shipped cases
has one): such links are now never merged. (2) The 2024 pairwise merge loses a shelter attached by a link under the threshold: the
CLI refuses `--merge legacy` with `attach`. (3) `node_weights` shared a cell's people with the shelter nodes in it, and
`proportional` then dropped them (a cell holding only an attached shelter lost all its people): shelters get no share now, snapping
had a milder form of the same loss. (4) A shelter could be attached to a node that has no street. (5) A coordinate array of the wrong
shape was silently reshaped. The shipped cases did not change.

## Does it matter for a result? (A5)

Shortest-path evacuation of area 2 with the default `ModelOptions`, 30 departure-time seeds, 120 min simulated:

| Variant | Agents | Last evacuee (s) | Safe at 30 min |
|---|---:|---:|---:|
| 1  the 2024 tables and population file | 622 | 2,645 ± 104 | 529.2 (85.1 %) |
| 2  rebuilt, same recipe (clusters; none at a shelter) | 622 | 2,609 ± 112 | 527.9 (84.9 %) |
| 3  rebuilt, area-weighted census total, uniform | 1,704 | 2,684 ± 66 | 1,426.6 (83.7 %) |
| 4  rebuilt, area-weighted census total, placed by the census | 1,704 | 2,672 ± 104 | 1,440.5 (84.5 %) |

For the shortest-path baseline on this area the differences are small: the rebuilt tables change the last evacuee by 36 s (1.4 %,
inside the run-to-run spread), and 2.7 times as many agents by 75 s (3 %): the streets of area 2 do not saturate. (Variant 4 read
2,638 ± 83 and 83.8 % before the review fix that keeps shelter nodes out of the census weights moved a few people; variants 1–3 did
not change when the table was measured again.) These are
errors of *data quality* rather than of this metric. They matter for the absolute numbers (622 people against 1,704), for areas where
crowding does bind, and for the learned policy, whose state contains the density of the links; none of that was measured here.

## Also found

* `cases/new_kochi` has a node (3106) with 11 links; the model holds 10 choices, so its table drops the last link at that end only
  (the old script says "Not the best solution", Aug 2021) and an agent could walk it one way. The builder now stops on such a node
  (`--excess prune` removes the longest links there instead). `new_kochi` itself was left as it is.
* The old all-pairs shortest-path step needs n × n matrices: about 3 GB each for `new_kochi`. `next_nodes` searches from all shelters at
  once and gives the same table (with the same parallel-link rule) in the memory of the network: identical on all five areas above.

## Decisions for you

| | Decision | Recommendation |
|---|---|---|
| S1 | Population of the shipped `kochi_area*`: the study's (`within`, uniform) or the census total by area-weighting, placed by the census (`weighted`, `proportional`) | the second is closer to the people there; the first allows comparison with the study. Both are one command (see `cases/README.md`); I shipped the first |
| S2 | Shelters far from the network: keep snapping (study), leave out beyond a distance, or attach each shelter by a link whose length is the distance | **built and applied** (A7): the four shipped cases were rebuilt with it; `--shelters-as snap` and `--legacy` give the study's version |
| S3 | The licence and terms of the OSM-derived network and of the shelter and census layers (`cases/README.md`) | the shelter and census layers are from the Kochi Prefectural Office and public domain, **as stated by you** (recorded in `cases/README.md`, not verified against the Office's own terms); still open: the ODbL notice for the OpenStreetMap network and the origin of the area polygons |
| S4 | `new_kochi`: rebuild its tables without the 11th-link inconsistency | later, with the experiment layer |
