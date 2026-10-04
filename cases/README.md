# Cases

One folder per study area: `data/` holds the tables the model reads. Run outputs go to `state_<name>/` (not tracked).

| Case | What it is |
|------|------------|
| `kochi` | The old Kochi network (4,315 nodes), census-based population of 35,930. |
| `new_kochi` | Kochi with 19,207 nodes. One node (3106) had 11 links, the model holds 10: its table used to drop the 11th at that end only. `fixExcessLinks.py` removed the longest link there (link 4779, 22 m, node 3106 to node 4588) from both ends and renumbered the links; see [docs/engine-reconciliation.md](../docs/engine-reconciliation.md). |
| `kochi_area0`, `1`, `2`, `4` | Four of the tsunami-evacuation areas of Kochi City, built with `evacrl.casebuild` from the 2024 Kochi study ([erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara)). Below. |

## `kochi_area*`

| Case | Nodes | Links | Shelters | Agents | Short links merged | Shelter points used / shelters | Longest access link |
|------|------:|------:|---------:|-------:|-------------------:|--------------------------------|---------------------|
| `kochi_area0` | 537 | 721 | 6 | 4196 | 16 | 6 / 6 | 70 m, 0 over 100 m |
| `kochi_area1` | 423 | 665 | 4 | 2743 | 20 | 4 / 4 | 360 m, 2 over 100 m |
| `kochi_area2` | 297 | 439 | 4 | 1704 | 16 | 4 / 4 | 78 m, 0 over 100 m |
| `kochi_area4` | 1110 | 1634 | 15 | 13502 | 39 | 17 / 15 (three points at one location: one shelter) | 221 m, 4 over 100 m |

Each folder has `data/` (nodes, links, actions, transitions, agents, `nextnode.csv` for the shortest-path baseline), `raw/` (the
network before the clean-up, with OpenStreetMap ids, from which `data/` is rebuilt offline), `node_population.csv` (the people
placed at each node of the cleaned network, which is how the agents are rebuilt without the census) and `provenance.json`
(settings, counts, input checksums, software versions). `tests/test_casebuild_cases.py` rebuilds every case from its `raw/` and compares.

**What was chosen**, so that the cases agree with the 2024 study where it is sound (details and measurements:
[docs/audits/step3](../docs/audits/step3/README.md)):

* Network: the OSM graphs of the study. Shelters: the shelter points of the study, **attached to the network by an access
  link** (`--shelters-as attach`): a node at the shelter, joined to the nearest street node by a link as long as the distance,
  instead of the shelter being the nearest node (the study; `--shelters-as snap`). Nothing is moved onto the edge of the area any
  more and the walk from the street to the shelter counts: the longest is 360 m, in area 1. Limits: the link is the straight line
  to the nearest *node* (a lower bound of the walk, through whatever lies between), and the shelter is a dead end, as before.
* **Population: the area-weighted census** (`--census-method weighted --strategy proportional`). The number of agents is the
  census total of the area counting every 500 m cell that touches it by the part inside (4,196 / 2,743 / 1,704 / 13,502; the
  study's 2,303 / 1,078 / 622 / 12,288 counted only the cells lying entirely inside, and so lost the boundary cells, most of the
  people of a small area), and each cell's people are shared equally by the street nodes inside it: 4,196 etc. agents, the same
  every time, none at a shelter. A cell with people and no node goes to the nearest node. The mesh is coarse (a cell is about
  460 m by 580 m and holds many nodes, up to 2.5 % of area 4's people end on one node), so the distribution inside a cell is an
  assumption, not data. For the study's population use `--census-method within --strategy uniform --seed 0`.
* Other differences from the study: the short-link clean-up merges clusters of nodes instead of pairs (no orphan nodes, no
  cut-off chains), the shortest-path table uses the shortest of several parallel links, and no agent starts at a shelter.

**Not shipped:** area 3 has no shelter inside its box, so no evacuation is possible; `kochi42` of the study has no clear origin
(its population file has exactly the census total of area 4, while the file of its `kochi4` has 13,244 agents).

## Data sources: check before publishing

* Road network: © OpenStreetMap contributors, ODbL 1.0 (<https://www.openstreetmap.org/copyright>), downloaded with OSMnx for the 2024
  study; the date of that download is not recorded. `raw/` and `data/` are derived from it, so the ODbL attribution (and possibly
  its share-alike) conditions may apply to them.
* Shelters and tsunami evacuation buildings (`kochi_tsunami_shelters`, `kochi_tsunami_evacbldg`) and the census mesh
  (`M_TOTPOP_H` of `kochi-shi_census`): provided by the **Kochi Prefectural Office**, and **stated by the repository owner
  (2026-10-04) to be in the public domain**. That is recorded here as it was given: the licence text or terms of use of the
  Office were not looked at, so cite the Office as the source and add its exact wording (an open-data licence, if it has one)
  before publishing. The tables here keep only coordinates, lengths and start nodes (no names or addresses).
* The area polygons were drawn for the 2024 study.

## Rebuilding

```
python -m evacrl.casebuild validate cases/kochi_area2
python -m evacrl.casebuild from-raw cases/kochi_area2/raw cases/kochi_area2 --strategy proportional \
    --weights cases/kochi_area2/node_population.csv                                                 # offline
python -m evacrl.casebuild from-snapshot GRAPH_DIR cases/kochi_area2 --areas AREAS.geojson --index 2 \
    --shelters EVACBLDG.geojson SHELTERS.geojson --shelters-as attach --census CENSUS.geojson \
    --census-method weighted --strategy proportional
```

`GRAPH_DIR` holds `Gnodes.geojson` and `Gedges.geojson` (a stored OSM download); the GeoJSON files are those of the 2024 study
(`EVACMODEL3_FocalPoints/kochi_data`). Run a case from Python, for instance
`run_ql_mod(area="kochi_area2", simtime=30, meandeparture=5, numBlocks=1, simPerBlock=100, name="a")` from `scripts/main_ql_mod.py`.
