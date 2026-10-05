# datasets/gis

QGIS layers and figures of the tsunami inundation and the road network of Kochi, and `fromatconvert.ipynb`, the notebook that reads them (it uses `./data`).

**`data/qgis/` and `data/qgis_1/` (277 MB: shapefiles of the nodes, edges, shelters and areas, and the rasters `inund5.tif`, the maximum inundation
depth, and `etat_utm_6690.tif`) are no longer tracked**, to keep a clone small. Nothing in `evacrl`, the cases or the tests uses them. As long as the git history has not been rewritten they are in it, up to commit
`e16d91d`; otherwise they are in the maintainer's backup archive (8 MB compressed):

```
git archive e16d91d datasets/gis/data | tar -x          # restores datasets/gis/data/qgis and qgis_1
```

(`.gitignore` keeps them out if you put them back.) The source and terms of these data are not recorded: see [docs/data-licences.md](../../docs/data-licences.md).
