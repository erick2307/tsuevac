# Tsunami Evacuation Guidance System

## PreProcess Manual (2021.Aug2)

- Run `preProcess.py` in the case folder
- Requires `createLinksAndNodes.py`, `getPopulation.py`,`setActionsAndTransitions.py`
- Run `main.py` in the main folder


## Tsunami Simulation outputs:  
1. `arrivaltime.dat.asc`:
	This is a raster file with the arrival time.  
	What is the **arrival time**?  
	==It is the first time when an inland grid is inundated.==  
	
2. `inund5.dat.asc`:  
	This is a raster file with the **maximum** inundation depth at each inland grid.  
	
## Preprocess for Evacuation:
### Tsunami attribute to road network
Tsunami inundation depth and the arrival time in a **10m resolution** size are included as attributes in the nodes contain in a grid.
> Possible problems here are:  
> 1. One node is exactly in the middle of two or more grids. Duplicate attribute.  

### Area split
Divide the computation area.  
> Some notes here:   
> 1. How to divide the areas?  
> 2. Based on road-connectivity (graph-theory) or with regular grid.
> 3. If regular grid, what to do with interrupted graphs?


### Population selection
Find the population inside the inundation area.  
- Are we using only these population? Shall we include a shadow?
- How big of a shadow? Tsunami runup uncertainty = buffer

# Notes (2021.07.20)

1. Abe san data is on EPSG: 2446 (JGD2000 / JPR CS IV) and OSMNX downloads the graph in WGS84 EPSG 4326. 
2. The OSMNX data can be projected and exported to shp, then the CRS is EPSG 32653 - WGS84 / UTM zone 53N
3. Kochi polygon shp is in EPSG 4612 - JGD2000  
4. The file `lib_ImportOSM.py` works in Terminal but not from VS Code. VS Code terminal shows another interpreter Python 3.8.2 compared to one in conda 3.7.7
5. The file `preprocess.ipynb` also can generate a `linksdb.csv` file.



## Issues

1. Project all to same CRS (WGS84? JGD2000?)

https://epsg.org/home.html

- ==**JGD2011 / UTM zone 53N (EPSG: 6690)**==
- JGD2000 / JPR CS IV (EPSG: 2446)
- WGS84 / UTM zone 53N (EPSG: 32653)
- ==**JGD2011 (EPSG: 6668) (Geographic 2D)**==
- JGD2000 (EPSG: 4612) (Geographic 2D)
- WGS84 (EPSG: 4326) (Geographic 2D)
	
# Preprocess for Population

1. `SetDatabaseBldMeshCodes.py`
2. `SetpopDB.py`
3. `DisaggregationLibrary.py`

# Kochi data

- Area code polygons (WGS84 - EPSG:4326):  
- 
> Dropbox/zDATA/PAREA_Town_2018/Shape形式/Shape形式/世界測地系/39/A3924POL.shp

# to Obtain Population within inundation area
1. Use `getPopulatio.py`>`getPopulationArea` to extract population within the 'aos' --> `areaPop`
2. Use the feature `areaPop` and the raster `inund5` to `Add values to feature`
3. Select features `>-99`
Note: This is a bit overestimated since areas near the river or at the edge of the inundation line are also included. A shadow for evacuation.
