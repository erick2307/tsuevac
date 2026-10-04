# Cases

One folder per study area: `data/` holds the tables the model reads. Run outputs go to `state_<name>/` (not tracked).

| Case | What it is |
|------|------------|
| `kochi` | The old Kochi network (4,315 nodes), census-based population of 35,930. |
| `new_kochi` | Kochi with 19,207 nodes. One node has 11 links, the model holds 10: its table drops the 11th, see [docs/engine-reconciliation.md](../docs/engine-reconciliation.md). |
| `kochi_area0`, `1`, `2`, `4` | Four of the tsunami-evacuation areas of Kochi City, built with `evacrl.casebuild` from the 2024 Kochi study ([erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara)). Below. |

## `kochi_area*`

| Case | Nodes | Links | Shelters | Agents | Short links merged | Shelter points used / shelter nodes | Farthest shelter snapped |
|------|------:|------:|---------:|-------:|-------------------:|-------------------------------------|--------------------------|
| `kochi_area0` | 531 | 715 | 6 | 2303 | 16 | 6 / 6 | 70 m, 0 over 100 m |
| `kochi_area1` | 419 | 661 | 3 | 1078 | 20 | 4 / 3 | 360 m, 2 over 100 m |
| `kochi_area2` | 293 | 435 | 4 | 622 | 16 | 4 / 4 | 79 m, 0 over 100 m |
| `kochi_area4` | 1095 | 1619 | 15 | 12288 | 39 | 17 / 15 | 221 m, 6 over 100 m |

Each folder has `data/` (nodes, links, actions, transitions, agents, `nextnode.csv` for the shortest-path baseline), `raw/` (the
network before the clean-up, with OpenStreetMap ids, from which `data/` is rebuilt offline) and `provenance.json` (settings,
counts, input checksums, software versions). `tests/test_casebuild_cases.py` rebuilds every case from its `raw/` and compares.

**What was chosen**, so that the cases agree with the 2024 study where it is sound (details and measurements:
[docs/audits/step3](../docs/audits/step3/README.md)):

* Network, shelters and the population totals are those of the study: the same OSM graphs, the same shelter points snapped to the
  nearest node, the census cells lying *entirely inside* the area (`--census-method within`), agents placed at random on the
  nodes (`--strategy uniform`, seed 0).
* Differences: the short-link clean-up merges clusters of nodes instead of pairs (no orphan nodes, no cut-off chains), the
  shortest-path table uses the shortest of several parallel links, and no agent starts at a shelter.

**Two things to decide before relying on these cases**

1. *Population.* `within` loses the census cells on the boundary of an area, so it undercounts: area-weighting gives 4,196 (area 0),
   2,743 (1), **1,704 (2, against 622)**, 13,502 (4, against 12,288). `--census-method weighted --strategy proportional` places the
   weighted total by the census. The study's figures were kept here so that results can be compared with it.
2. *Shelters far from the network.* A shelter point is snapped to the nearest node however far it is; in area 1 two of the four
   points are more than 300 m from any node, so the model puts them on the edge of the area. `--max-snap-distance 100` leaves such
   points out (area 1 would keep two of its three shelters, area 4 eleven of its fifteen, areas 0 and 2 all of theirs).

**Not shipped:** area 3 has no shelter inside its box, so no evacuation is possible; `kochi42` of the study has no clear origin
(its population file has exactly the census total of area 4, while the file of its `kochi4` has 13,244 agents).

## Data sources: check before publishing

* Road network: © OpenStreetMap contributors, ODbL 1.0 (<https://www.openstreetmap.org/copyright>), downloaded with OSMnx for the 2024
  study; the date of that download is not recorded. `raw/` and `data/` are derived from it, so the ODbL attribution (and possibly
  its share-alike) conditions may apply to them.
* Shelters and tsunami evacuation buildings (`kochi_tsunami_shelters`, `kochi_tsunami_evacbldg`) and the census mesh
  (`M_TOTPOP_H` of `kochi-shi_census`) were taken from the 2024 study as they are. **Their licences and terms of use have not been
  checked.** The tables here keep only coordinates, lengths and start nodes (no names or addresses).
* The area polygons were drawn for the 2024 study.

## Rebuilding

```
python -m evacrl.casebuild validate cases/kochi_area2
python -m evacrl.casebuild from-raw cases/kochi_area2/raw cases/kochi_area2 --agents 622 --seed 0     # offline
python -m evacrl.casebuild from-snapshot GRAPH_DIR cases/kochi_area2 --areas AREAS.geojson --index 2 \
    --shelters EVACBLDG.geojson SHELTERS.geojson --census CENSUS.geojson --census-method within --strategy uniform --seed 0
```

`GRAPH_DIR` holds `Gnodes.geojson` and `Gedges.geojson` (a stored OSM download); the GeoJSON files are those of the 2024 study
(`EVACMODEL3_FocalPoints/kochi_data`). Run a case from Python, for instance
`run_ql_mod(area="kochi_area2", simtime=30, meandeparture=5, numBlocks=1, simPerBlock=100, name="a")` from `scripts/main_ql_mod.py`.
