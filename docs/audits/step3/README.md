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

The study sums the census cells lying *entirely inside* an area. Cells on the boundary are lost, and the cells are 250 m wide:

| Area | Study (`within`) | Boundary cells by area (`weighted`) | More |
|---|---:|---:|---:|
| 0 | 2,303 | 4,196 | +82 % |
| 1 | 1,078 | 2,743 | +155 % |
| 2 | 622 | 1,704 | +174 % |
| 3 | 240 | 546 | +128 % |
| 4 | 12,288 | 13,502 | +10 % |

(area-weighting assumes the people are spread evenly over a cell.) The smaller the area, the larger the error.

## The shelters (A3)

Shelter points (53 shelters and 339 evacuation buildings) are snapped to the nearest node of the area's network, however far it is
(the study took every point inside the *box* of the network). Distances to the node: median 50–190 m. In area 1 two of the four points
are more than 300 m from any node (360 m at most), in area 4 six of 17 are more than 100 m: the model puts such a shelter on the
edge of the area, so the walk from there to the building is not part of the evacuation. Area 3 has no shelter inside its box at all.
`--max-snap-distance 100` leaves the far ones out (area 1: 2 of 3 shelters kept; area 4: 11 of 15).

## Does it matter for a result? (A5)

Shortest-path evacuation of area 2 with the default `ModelOptions`, 30 departure-time seeds, 120 min simulated:

| Variant | Agents | Last evacuee (s) | Safe at 30 min |
|---|---:|---:|---:|
| 1  the 2024 tables and population file | 622 | 2,645 ± 104 | 529.2 (85.1 %) |
| 2  rebuilt, same recipe (clusters; none at a shelter) | 622 | 2,609 ± 112 | 527.9 (84.9 %) |
| 3  rebuilt, area-weighted census total, uniform | 1,704 | 2,684 ± 66 | 1,426.6 (83.7 %) |
| 4  rebuilt, area-weighted census total, placed by the census | 1,704 | 2,638 ± 83 | 1,428.8 (83.8 %) |

For the shortest-path baseline on this area the differences are small: the rebuilt tables change the last evacuee by 36 s (1.4 %,
inside the run-to-run spread), and 2.7 times as many agents by 75 s (3 %): the streets of area 2 do not saturate. These are
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
| S2 | Shelters far from the network: keep snapping (study), leave out beyond a distance, or attach each shelter by a link whose length is the distance | attach by a link: nothing is lost and the walk counts. Not implemented yet; `--max-snap-distance` exists meanwhile |
| S3 | The licence and terms of the OSM-derived network and of the shelter and census layers (`cases/README.md`) | yours to confirm before the cases are public |
| S4 | `new_kochi`: rebuild its tables without the 11th-link inconsistency | later, with the experiment layer |
