 # Heatwave-sensitive multi-temporal Local Climate Zone mapping in Tallinn

- this repository was created to document a workflow established by Moritz Mühlbauer as part of the Master's Thesis `Mapping Surface Urban Heat Island intensity and the influence of urbanization and heatwaves through Local Climate Zone mapping Tallinn - 2014 and 2022` at the University of Tartu in 2025
- to support training area selection for supervised classification, building and landcover vector data are aggregated to `Local Climate Zone` (LCZ) relevant metrices
- to further improve the understanding of the study areas urban morphology, spectral indices from Landsat 8 satellite imagery are incorporated and combined with vector data
- the supervised classification into LCZ is conducted using an updated workflow established by Demuzere (2020)
- based on the LCZ, the `Surface Urban Heat Island` (SUHI) intensity is estimated with Landsat 8 land surface temperature (LST)

## Requirements

- Conda/Micromamba environment. Instruction [here](https://www.anaconda.com/download) and [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Install the necessary libraries from `requirements.yml` 
- Google Earth Engine account. Sign-up instructions [here](https://courses.spatialthoughts.com/gee-sign-up.html)
- Prior to using the Earth Engine Python client library, you need to authenticate (verify your identity) and use the resultant credentials to initialize the Python client. Instructions [here](https://developers.google.com/earth-engine/guides/python_install?hl=de)

## Buildings and Landcover Processing

- the buildings and landcover processing can be executed running `?` with `?` containing the helper functions
- the main final rasterized outputs are: **Built fraction**, **Mean Building Height** and **Dominant Landcover**
- all vector datasets are aggregated to 30 meters Landsat 8 resolution to ensure further combined processing

### Relevant coding highlights

```python
def calculate_built_fraction(buildings, grid):
    """
    Calculates the percentage of built-up area per 30x30 m grid cell.

    Performs a spatial intersection between buildings and grid cells,
    sums the intersected areas per cell, and computes the built-up
    fraction as a percentage. Results are saved to a GeoPackage.

    Parameters:
        buildings (GeoDataFrame): Building footprint geometries.
        grid (GeoDataFrame): Grid with a 'grid_id' column.

    Returns:
        GeoDataFrame: Grid with an added 'built_fraction' column.
    """
    
    intersection = grid.overlay(buildings, how="intersection")
    # calculating the area of the intersect building fraction per grid cell
    intersection["fraction_area"] = intersection["geometry"].area
    # grouping by grid cell id and summing up the fraction area on the fly
    grouped = intersection.groupby("grid_id")["fraction_area"].sum().reset_index()
    # calculating built fraction on previuosly summed building area per grid cell (900 due to 30x30 resolution)
    grouped["built_fraction"] = ((grouped["fraction_area"] / 900) * 100).round().astype(int)
    # merging grouped DataFrame to grid to create GeoDataFrame
    built_fraction = grid.merge(grouped, on="grid_id", how="inner")

    built_fraction.to_file(BUILT_FRACTION_DIRECTORY / "built_fraction.gpkg")

    return built_fraction
```

```python
def find_dominant_landcover(landcover, grid):
    """
    Identifies the dominant land cover type within each 30x30 m grid cell.

    Intersects land cover polygons with grid cells, calculates area per 
    land cover type within each cell, and selects the type with the 
    largest area as dominant. Results are saved to a GeoPackage.

    Parameters:
        landcover (GeoDataFrame): Land cover polygons with a 'raster_code' column.
        grid (GeoDataFrame): Grid with a 'grid_id' column.

    Returns:
        GeoDataFrame: Grid with an added 'raster_code' column indicating 
                      the dominant land cover type per cell.
    """
    
    # intersection between grid and landcover
    landcover_intersect = grid.overlay(landcover, how="intersection")

    # Calculate area of clipped land use per grid cell
    landcover_intersect["dominant_area"] = landcover_intersect.geometry.area

    # Group by grid cell and land use type, summing areas
    landcover_areas = landcover_intersect.groupby(["grid_id", "raster_code"])["dominant_area"].sum().reset_index()

    # Select the dominant land use (largest area per grid cell)
    dominant_landcover = landcover_areas.loc[landcover_areas.groupby("grid_id")["dominant_area"].idxmax()]

    # Merge with grid
    landcover_grid_dominant_area = grid.merge(
        dominant_landcover[["grid_id", "raster_code"]],
        on="grid_id",  # Merging on the grid ID column
        how="inner"  # Only keep matching grid cells
    )
    
    landcover_grid_dominant_area.to_file(DOMINANT_LANDCOVER_DIRECTORY / 'landcover_grid_dominant_area.gpkg')
    
    return landcover_grid_dominant_area
```

## Satellite Imagery selection, spectral indices calculation and LST extraction 


#### **3. Select the `Copernicus Marine Toolbox` Kernel in the right upper corner of your browser window inside the IDE:**

<img src="images/explain1.png" alt="Description" width="400"/>      <img src="images/explain2.png" alt="Description" width="400"/>


## Usage

### Step 1: Built a query

To use the code provided, clone this repository to your local machine and open `hot_cold_spot_analysis.ipynb` in the previously created set-up.

To run a Hot-Cold-Spot Analysis over different datasets, geographic extents or date ranges, find the `USER_INPUT` section at the beginning of `hot_cold_spot_analysis.ipynb`

The  `Copernicus Marine Toolbox` sends a get_feature request through a Python API to the Copernicus Server. The query can be customized by passing a dictionary (`input_dict` compare code snippet). It consists of the following components:
  - `dataset_id` defines the product to be queried, by default set to *Mediterranean Sea - High Resolution and Ultra High Resolution L3S Sea Surface Temperature*
  - `variables` defines a list of variables selected from the product (identifer = dataset_id), by default set to *adjusted sea surface temperature*
  - `max and min lon/lat` define the bounding box of the query, by default set to *whole Agean sea*
  - `start/end_datetime` define the time period covered by the dataset, by default set to *winter 2023/24*

Customize your query and find a link where to get information about available datasets in the reference section.


```python
input_dict = {
    "dataset_id": "SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_b", 
    "variables": ["adjusted_sea_surface_temperature"], 
    "minimum_longitude": 19.22659983450641, 
    "maximum_longitude": 28.439441984120553, 
    "minimum_latitude": 34.62160284496615, 
    "maximum_latitude": 40.9634662781889, 
    "start_datetime": "2023-12-01T00:00:00", 
    "end_datetime": "2024-02-28T00:00:00", }
```


After modifying `input_dict`, adjust `units`, `variable_abreviation`, `DOI`, `spatial_resolution` and `temporal_resolution` to your queried dataset. 

Since *sea surface temperature* is queried by default, `unit` is assigned "°C" and the abreviation equals "sst". Those input parameters will appear on the output plots. 

Make sure to extract copyright information (`DOI`) and properties (`spatial` and `temporal resolution`) to match your dataset (compare reference section). Find metadata on the products website (for this example case compare [here](https://data.marine.copernicus.eu/product/SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012/description))


```python
unit = "°C"

variable_abreviation = "sst"

DOI = "https://doi.org/10.48670/moi-00171"

spatial_resolution = "0.01° × 0.01°"

temporal_resolution = "Daily"
```

### Step 2: Conduct temporal comparison

Depending on the extent and variable selection of the query, fetched datasets can get heavy very quickly. To avoid crashing, this workflow allows one date range selection per iteration. 

Therefore, when conducting a temporal comparison (e.g. of the same extent) it is recommended to run the `hot_cold_spot_analysis.ipynb` twice (with different `input_dict` modifications of `start/end_datetime`).

### Step 3: Conduct variable comparison

**NOTE**: This experimental workflow was designed for sea surface temperature analysis. To adapt it for a different variable, modify the preprocessing function accordingly:

```python
def pre_processing(dataset, variable_abreviation):

    # renaming data variable
    dataset = dataset.rename({"adjusted_sea_surface_temperature": variable_abreviation})

    # converting Kelvin to Celsius
    dataset[variable_abreviation] = dataset[variable_abreviation]-273.15

    # handle projections
    dataset = dataset.rio.write_crs(CRS.from_epsg(4326))
    dataset = dataset.rio.reproject(CRS.from_epsg(3857))

    print("\nDataset pre-processing finished")

    return dataset
```

## Output

`hot_cold_spot_analysis.ipynb` has the following outputs:
  - **Plots:**
      - Mean, Median and Standard Deviation
      - 5 % highest values per grid cell (collapsed over the time dimension)
      - GETIS-ORD G* Hot-Cold Spot over the Median of the 5 % highest values AND the entire dataset
  - **Data:**
      - Mean, Median, Standard Deviation and 5 % highest values as GeoTiff
      - GETIS-ORD G* Statistics as GPKG

File naming convention:
```
[statistical metric, eg. median]_{variable_abreviation}_{start_date}-{end_date}.[extension, eg. .gpkg]
```

**NOTE:** Since plot and data output names are generated based on the provided input data, running the notebook multiple times with different metadata selections **does not** overwrite existing files.

Find the style file `hot_cold_spot_style_qgis.qml` for GETIS-ORD G* GPKG plotting (e.g in QGIS) in this repository.

Find a temporal comparison between winter 23/24 and summer 24 of *sea surface temperature* over the whole *Agean Sea* produced with `hot_cold_spot_analysis.ipynb` in the `output` folder of this repository. 

## Relevant References

### Copernicus Marine Ocean Products

- access the Copernicus Marine Data Storage and find data compatible with this script [here](https://data.marine.copernicus.eu/products)
- explore quality of Copernicus Marine Data [here](https://pqd.mercator-ocean.fr/?pk_vid=161106812679b150)

### Complementary 

- read about xarray [here](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html)
- watch a video about Hot-Spot Anylsis [here](https://www.youtube.com/watch?v=sjLyJW95fHM)
- learn about geospatial Python [here](https://geog-312.gishub.org)
