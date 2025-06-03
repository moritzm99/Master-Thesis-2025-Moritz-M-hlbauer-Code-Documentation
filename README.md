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
        
    return landcover_grid_dominant_area
```

```python
def calculate_mean_building_height(buildings, grid):
    """
    Calculates the mean building height within each 30x30 m grid cell.

    Intersects building footprints with the grid, computes the average height 
    of buildings per cell, and merges the result back to the grid. The output 
    is saved as a GeoPackage.

    Parameters:
        buildings (GeoDataFrame): Building footprints with a 'height' column.
        grid (GeoDataFrame): Grid with a 'grid_id' column.

    Returns:
        GeoDataFrame: Grid with an added 'height' column representing the 
                      mean building height (rounded to the nearest integer).
    """
    # Intersect buildings with grid to assign buildings to grid cells
    grid_buildings_intersection = grid.overlay(buildings, how="intersection")
    # Calculate mean height of buildings per grid cell
    mean_building_height_grouped = grid_buildings_intersection.groupby(["grid_id"])["height"].mean().reset_index()
    # Merge mean height values back into the original grid
    mean_building_height = grid.merge(mean_building_height_grouped, on="grid_id", how="inner")
    # Round heights to the nearest integer for cleaner representation
    mean_building_height["height"] = mean_building_height["height"].round().astype(int)

    return mean_building_height
```
## Satellite Imagery selection, spectral indices calculation and LST extraction

- image selection of Landsat 8 images for spectral indices calculation (*NDBI*, *NDVI* and *NDWI*) and LST extraction for heatwave and non-heatwave days with `extract_spectral_indices.ipynb`
- heatwave days are identified through maximum daily air temperature with functions contained in `data_wrangling.py` and `01_JSONfromHTML_v2.ipynb` for accessing meteorological data from Estonian Weather Agency API

### Relevant Coding Highlights

- heatwave identification using threshold of >27 °C for three consecutive days:

```python
def get_heatwave(air_temp):
    # Create a copy of the DataFrame to avoid modification issues
    air_temp = air_temp.copy()

    threshold = 27
    
    # Step 1: Identify if each day exceeds the threshold
    air_temp["above_threshold"] = air_temp["air_temp"] > threshold
    
    # Step 2: Compute rolling sum over 3-day periods
    air_temp["rolling_sum"] = air_temp["above_threshold"].rolling(window=3, min_periods=1).sum()
    
    # Step 3: Initialize the Heatwave column
    air_temp["Heatwave"] = False
    
    # Step 4: Iterate over the dataset and apply back-labeling when a heatwave is detected
    for i in range(len(air_temp)):
        if air_temp.iloc[i]["rolling_sum"] >= 3:  # If a heatwave is detected
            j = i
            while j >= 0 and air_temp.iloc[j]["above_threshold"]: 
                air_temp.iloc[j, air_temp.columns.get_loc("Heatwave")] = True
                j -= 1  

    # Saving only heatwave period
    heatwave_period = air_temp.loc[air_temp["Heatwave"] == True]

    return heatwave_period
```

- visual screening of satellite imagery for no cloud-cover and full study area coverage of Tallinn municipality:
```python
def create_maps_from_collections(collections_by_year, vis_params, boundary):
    maps = {}

    for year, collection_list in collections_by_year.items():
        # Create a new interactive map
        map_selection = geemap.Map(center=[59.430752, 24.751965], zoom=11)

        # Iterate over image list and add to map
        for i in range(collection_list.size().getInfo()):
            image = ee.Image(collection_list.get(i))
            label = f"{image.get('DATE_ACQUIRED').getInfo()} / list position: {i}"
            map_selection.add_layer(image, vis_params, label)

        # Add boundary overlay
        map_selection.add_layer(boundary, {}, "Boundary")

        # Store map
        maps[year] = map_selection

    return maps
```

## Local Climate Zone supervised classification

- for multi-temporal LCZ classification, access the full WUDAPT created by Demuzere (2020) and updated by Mühlbauer (2025) [here](https://github.com/matthiasdemuzere/multitemporal-lcz-mapping)

## Surface Urban Heat Island Intensity estimation 

- the SUHI itensity is estimated as the difference of the mean LST of a given LCZ with the means LST of the reference zone LCZ D/14 (low plants)
- the processing of Landsat 8 LST and the LCZ classification to obtain SUHI as thermal LCZ-LST difference is conducted using `SUHII.ipynb`

### Relevant Coding Highlights

- SUHI intensity estimation from reference zone:

```python
def calculate_SUHII(LCZ, LST, nodata_lcz=None, nodata_lst=None, lcz_d_code=14):
    
    # creating a mask of valid data
    mask = (~np.isnan(LST)) & (~np.isnan(LCZ))
    if nodata_lst is not None:
        mask &= (LST != nodata_lst)
    if nodata_lcz is not None:
        mask &= (LCZ != nodata_lcz)

    # Also exclude LCZ == 0
    mask &= (LCZ != 0)

    # Flatten for DataFrame
    df = pd.DataFrame({
        'LCZ': LCZ[mask].astype(int),
        'LST': LST[mask]
    })

    mean_lst = df.groupby('LCZ')['LST'].mean()
    mean_d = mean_lst[lcz_d_code]

    SUHII = np.full_like(LST, np.nan, dtype=np.float32)
    for lcz_class in mean_lst.index:
        SUHII[LCZ == lcz_class] = mean_lst[lcz_class] - mean_d

    return SUHII, mean_lst, mean_d, mask
```

- Quantify urbanization from LCZ area:

```python
def calculate_lcz_areas(lcz_path, class_range=range(1, 11)):
    with rasterio.open(lcz_path) as src:
        lcz = src.read(1)
        transform = src.transform
        pixel_area = abs(transform.a * transform.e)
        nodata = src.nodata

    if nodata is not None:
        lcz = lcz[lcz != nodata]

    mask = np.isin(lcz, class_range)
    lcz_filtered = lcz[mask]

    unique, counts = np.unique(lcz_filtered, return_counts=True)
    areas_km2 = counts * pixel_area / 1e6

    return dict(zip(unique, areas_km2))
```

- Calculate area of SUHI by intensity class:

```python
def calculate_suhi_area_by_class(suhi_path, pixel_area_km2=0.0009):
    """
    Calculate SUHI area in km² for defined intensity ranges from a raster file.

    Intensity classes:
    1. (0–2]
    2. (2–4]
    3. (4–6]
    4. (6–8]
    5. >8

    Parameters:
    - suhi_path (str): Path to SUHI raster file
    - pixel_area_km2 (float): Area of a single pixel in km²

    Returns:
    - dict: Area in km² for each SUHI intensity class
    """
    with rasterio.open(suhi_path) as src:
        suhi_array = src.read(1)
        nodata = src.nodata

    # Mask out nodata values
    if nodata is not None:
        suhi_array = suhi_array.astype(np.float32)
        suhi_array[suhi_array == nodata] = np.nan

    classes = {
        "1: >0–2": (suhi_array > 0) & (suhi_array <= 2),
        "2: >2–4": (suhi_array > 2) & (suhi_array <= 4),
        "3: >4–6": (suhi_array > 4) & (suhi_array <= 6),
        "4: >6–8": (suhi_array > 6) & (suhi_array <= 8),
        "5: >8": (suhi_array > 8)
    }

    area_by_class = {
        label: np.sum(mask) * pixel_area_km2
        for label, mask in classes.items()
    }

    return area_by_class
```

- identify shifting LCZ between years:

```python
# Mask only valid classes
mask = np.isin(lcz_2014, list(valid_classes)) & np.isin(lcz_2022, list(valid_classes))

# Encode transitions
transitions = (lcz_2014 * 100 + lcz_2022).astype(np.int32)

# Mask unchanged values
changed = (lcz_2014 != lcz_2022) & mask
transitions[~changed] = 0  # or np.nan
```

## Relevant References

- Demuzere, M., Kittner, J., & Bechtel, B. (2021). LCZ Generator: A Web Application to Create Local Cli-
mate Zone Maps. FRONTIERS IN ENVIRONMENTAL SCIENCE, 9.
https://doi.org/10.3389/fenvs.2021.637455
- Demuzere, M., Mihara, T., Redivo, C. P., Feddema, J., & Setton, E. (2020). Multi-temporal LCZ maps for
Canadian functional urban areas. https://doi.org/10.31219/OSF.IO/H5TM6
- Stewart, I. D., & Oke, T. R. (2012). Local Climate Zones for Urban Temperature Studies. Bulletin of the
American Meteorological Society, 93(12), 1879–1900. https://doi.org/10.1175/BAMS-D-11-00019.1
- TLV. (n.d.). Tallinn Geoportal. Retrieved 26 May 2025, from https://www.tallinn.ee/et/geoportaal/ruumi-
andmed
-Žgela, M., Herceg-Bulić, I., Lozuk, J., & Jureša, P. (2024). Linking land surface temperature and local cli-
mate zones in nine Croatian cities. Urban Climate, 54. https://doi.org/10.1016/J.UCLIM.2024.101842
