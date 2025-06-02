# library import
import pandas as pd 
import geopandas as gpd
import pathlib

import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
from numpy import int16

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#year = 2014
year = 2022

# defining relative paths
NOTEBOOK_DIRECTORY = pathlib.Path().resolve()

MEAN_BUILDING_HEIGHT_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "mean_building_height"
BUILT_FRACTION_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "built_fraction"
DOMINANT_LANDCOVER_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "dominant_landcover"

RASTER_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "rasterized"
RECLASSIFIED_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "reclassified"

WRANGLED_DIRECTORY = NOTEBOOK_DIRECTORY / "output" / f"{year}" / "wrangled"

TEST_DIRECTORY = NOTEBOOK_DIRECTORY / "test"


########################################Pre-Processing#######################################################



# formating GeoDataFrames
def formating(buildings, landcover, grid):
    
    buildings.columns = buildings.columns.str.lower()
    landcover.columns = landcover.columns.str.lower()
    
    # Drop rows with missing geometries
    landcover = landcover.dropna(subset=["geometry"])
    buildings = buildings.dropna(subset=["geometry"])

    
    if year == 2014:

        buildings = buildings.rename(columns={"tyyp_id": "type_id", "korruselis": "stories", "materjal": "material", "k6rgus": "height", "abs_k6rgus": "abs_height", "m_korrusel": "undg_stories"})
        buildings = buildings.drop(columns=["tar_id", "vajalik", "markused", "lisamis_kp", "muutmis_kp", "lisaja", "muutja", "andmeallik", "korgusalli", "ruumikujua", "objectid", "a2_esitusr", "globalid", 'ads_lahiaa', 'ehr_gid', 'ads_oid', 'kmr_id', 'kpo_id', 'shape_leng', 'shape_area'])
        
        landcover = landcover.rename(columns={"tyyp_id": "type_id"})
        landcover = landcover.drop(columns=["tar_id", "liik", "kate", "nimi", "knr_kood", "vajalik", "markused", "lisamis_kp", "muutmis_kp", "lisaja", "muutja", "andmeallik", "korgusalli", "ruumikujua", "objectid", "a2_esitusr", "globalid", "symbol_3d", 'shape_leng', 'shape_area'])


    if year == 2022:
        
        buildings = buildings.rename(columns={"tyyp_id": "type_id", "korruselisus": "stories", "materjal": "material", "k6rgus": "height", "abs_k6rgus": "abs_height", "m_korruselisus": "undg_stories"})
        buildings = buildings.drop(columns=["tar_id", "vajalik", "markused", "lisamis_kp", "muutmis_kp", "lisaja", "muutja", "andmeallika_id", "korgusallika_id", "ruumikujuallika_id", "objectid", "a2_esitusreegel", "globalid", 'shape_length', 'shape_area', 'a2_esituserand'])
    
        landcover = landcover.rename(columns={"tyyp_id": "type_id"})
        landcover = landcover.drop(columns=["tar_id", "liik", "kate", "nimi", "knr_kood", "vajalik", "markused", "lisamis_kp", "muutmis_kp", "lisaja", "muutja", "andmeallika_id", "korgusallika_id", "ruumikujuallika_id", "objectid", "a2_esitusreegel", 'a2_esituserand', "globalid", "symbol_3d", 'shape_length', 'shape_area'])
    
    grid = grid.rename(columns={"id": "grid_id"})

    buildings['area'] = buildings.geometry.area
    landcover['area'] = landcover.geometry.area
    
    return buildings, landcover, grid



# cleaning building data
def clean_buildings(buildings):
    
    # dropping rows where stories and height have an NA because no height information can be extracted 
    buildings = buildings[~buildings.apply(lambda row: pd.isna(row['stories']) and pd.isna(row['height']), axis=1)]
    # dropping rows where stories and height are 0 because building doesnt seem to exist
    buildings = buildings[~buildings.apply(lambda row: (row['stories'] == 0) and (row['height'] == 0), axis=1)]
    
    # calculating mean 1 story height
    buildings['height_threshold'] = buildings.apply(
        # formula for calculating height of one story
        lambda row: (row['height'] / row['stories'])
        # calculate only for rows that have height and stories assigned, exlcude rows that have 0 stories or height
        if pd.notna(row['height']) and pd.notna(row['stories']) and row['height'] != 0 and row['stories'] != 0
        else None,
        axis=1
    )

    # calculating mean story height of all buildings higher than 0 stories
    mean_ht = buildings['height_threshold'].mean()

    # estimating building height
    buildings['height'] = buildings.apply(
        # assign new height
        lambda row: (
            (row['stories'] * mean_ht) if row['stories'] != 0 else (row['stories'] + mean_ht)
        ) if pd.isna(row['height']) else row['height'], # only for buildings that miss height
        axis=1
    )

    # masking out buildings smaller 2m to fit LCZ classification properties 
    buildings = buildings.loc[buildings['height'] >= 2]

    buildings['height'] = buildings['height'].round().astype(int)


    return buildings



# decrypting type_id column 
def decrypt(landcover, buildings):

    buildings["type"] = None
    landcover["landcover"] = None
    landcover["raster_code"] = None
    
    for row in landcover.itertuples():
        
        if row.type_id == 11:
            landcover.at[row.Index, "landcover"] = "as_with_veg"
        if row.type_id == 12:
            landcover.at[row.Index, "landcover"] = "as_without_veg"
        if row.type_id == 13:
            landcover.at[row.Index, "landcover"] = "ns_with_veg"
        if row.type_id == 14:
            landcover.at[row.Index, "landcover"] = "ns_without_veg"
        if row.type_id == 15:
            landcover.at[row.Index, "landcover"] = "as_without_veg"
        if row.type_id == 21:
            landcover.at[row.Index, "landcover"] = "water"
        if row.type_id == 22:
            landcover.at[row.Index, "landcover"] = "water"
        if row.type_id == 23:
            landcover.at[row.Index, "landcover"] = "water"
        if row.type_id == 100:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 110:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 120:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 130:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 210:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 220:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 230:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 240:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 250:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 310:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 320:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 330:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 410:
            landcover.at[row.Index, "landcover"] = "transportation"
        if row.type_id == 420:
            landcover.at[row.Index, "landcover"] = "transportation"

    for row in landcover.itertuples():

        if row.landcover == "as_with_veg":
            landcover.at[row.Index, "raster_code"] = 1
        if row.landcover == "as_without_veg":
            landcover.at[row.Index, "raster_code"] = 2
        if row.landcover == "ns_with_veg":
            landcover.at[row.Index, "raster_code"] = 3
        if row.landcover == "ns_without_veg":
            landcover.at[row.Index, "raster_code"] = 4
        if row.landcover == "water":
            landcover.at[row.Index, "raster_code"] = 5
        if row.landcover == "transportation":
            landcover.at[row.Index, "raster_code"] = 6

    for row in buildings.itertuples():
          
        if row.type_id == 1:
            buildings.at[row.Index, "type"] = "residential"
        if row.type_id == 2:
            buildings.at[row.Index, "type"] = "public"
        if row.type_id == 3:
            buildings.at[row.Index, "type"] = "outbuilding/auxiliary"
        if row.type_id == 4:
            buildings.at[row.Index, "type"] = "production"
        if row.type_id == 5:
            buildings.at[row.Index, "type"] = "underground"
        if row.type_id == 6:
            buildings.at[row.Index, "type"] = "underground"
        if row.type_id == 7:
            buildings.at[row.Index, "type"] = "underground"
        if row.type_id == 8:
            buildings.at[row.Index, "type"] = "other"
        if row.type_id == 0:
            buildings.at[row.Index, "type"] = "unspecified"
        
        
            
    return landcover["landcover"], landcover["raster_code"], buildings["type"]
    


# function for clipping landcover and buildings to the extent of Tallinn
def clip(buildings, landcover, tallinn):
    buildings = buildings.clip(tallinn)
    landcover = landcover.clip(tallinn)
    return buildings, landcover



# function for extracting unique landcovers
def landcover_extract(landcover):
    
    transportation = landcover.loc[landcover["landcover"] == "transportation"]
    as_with_veg = landcover.loc[landcover["landcover"] == "as_with_veg"]
    as_without_veg = landcover.loc[landcover["landcover"] == "as_without_veg"]
    ns_without_veg = landcover.loc[landcover["landcover"] == "ns_without_veg"]
    ns_with_veg = landcover.loc[landcover["landcover"] == "ns_with_veg"]
    water = landcover.loc[landcover["landcover"] == "water"]
    
    return transportation, as_with_veg, as_without_veg, ns_without_veg, ns_with_veg, water




def safe_gpkg(transportation, as_with_veg, as_without_veg, ns_without_veg, ns_with_veg, water, landcover, buildings):

    transportation.to_file(WRANGLED_DIRECTORY / f"transportation_{year}.gpkg")
    as_with_veg.to_file(WRANGLED_DIRECTORY / f"as_with_veg_{year}.gpkg")
    as_without_veg.to_file(WRANGLED_DIRECTORY / f"as_without_veg_{year}.gpkg")
    ns_without_veg.to_file(WRANGLED_DIRECTORY / f"ns_without_veg_{year}.gpkg")
    ns_with_veg.to_file(WRANGLED_DIRECTORY / f"ns_with_veg_{year}.gpkg")
    water.to_file(WRANGLED_DIRECTORY / f"water_{year}.gpkg")
    landcover.to_file(WRANGLED_DIRECTORY / f"landcover_wrangled_{year}.gpkg")
    buildings.to_file(WRANGLED_DIRECTORY / f"buildings_wrangled_{year}.gpkg")


#########################################Processing#######################################################


# function to calculate the built fraction per 30x30 m grid cell
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

    
    
# function to find dominant landcover per 30x30 m grid cell
def find_dominant_landcover(landcover, grid):
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



# function to calculate mean building height per 30x30 m grid cell
def calculate_mean_building_height(buildings, grid):
    
    grid_buildings_intersection = grid.overlay(buildings, how="intersection")
    mean_building_height_grouped = grid_buildings_intersection.groupby(["grid_id"])["height"].mean().reset_index()
    mean_building_height = grid.merge(mean_building_height_grouped, on="grid_id", how="inner")
    mean_building_height["height"] = mean_building_height["height"].round().astype(int)

    mean_building_height.to_file(MEAN_BUILDING_HEIGHT_DIRECTORY / 'mean_building_height.gpkg')

    return mean_building_height


################################################################################################


# function to rasterize landcover
def rasterize_landcover(dominant_landcover, base_raster):

    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value_2 = ((geom,value) for geom, value in zip(dominant_landcover.geometry, dominant_landcover['raster_code']))

    # Rasterize vector using the shape and transform of the raster
    landcover_raster = features.rasterize(geom_value_2,
                                    out_shape = base_raster.shape,
                                    transform = base_raster.transform,
                                    all_touched = True,
                                    fill = -2,   # background value
                                    merge_alg = MergeAlg.replace,
                                    dtype = int16)

    # Save the new raster using rasterio
    with rasterio.open(
        RASTER_DIRECTORY / "landcover.tif",
        "w",  # Write mode
        driver="GTiff",
        height=base_raster.shape[0],
        width=base_raster.shape[1],
        count=1,  # Single-band raster
        dtype=rasterio.int16,  # Match data type
        crs=dominant_landcover.crs,  # Copy CRS from the original raster
        transform=base_raster.transform,  # Copy spatial transform
        nodata=-2,  # Assign NoData value
    ) as dst:
        dst.write(landcover_raster, 1)  # Write data to band 1
        
    return landcover_raster



# function to rasterize built fraction
def rasterize_built_fraction(built_fraction, base_raster):
    
    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value = ((geom,value) for geom, value in zip(built_fraction.geometry, built_fraction['built_fraction']))

    # Rasterize vector using the shape and transform of the raster
    built_fraction_raster = features.rasterize(geom_value,
                                    out_shape = base_raster.shape,
                                    transform = base_raster.transform,
                                    all_touched = True,
                                    fill = 0,   # background value
                                    merge_alg = MergeAlg.replace,
                                    dtype = int16)

    # Save the new raster using rasterio
    with rasterio.open(
        RASTER_DIRECTORY / "built_fraction.tif",
        "w",  # Write mode
        driver="GTiff",
        height=base_raster.shape[0],
        width=base_raster.shape[1],
        count=1,  # Single-band raster
        dtype=rasterio.int16,  # Match data type
        crs=built_fraction.crs,  # Copy CRS from the original raster
        transform=base_raster.transform,  # Copy spatial transform
        nodata=0,  # Assign NoData value
    ) as dst:
        dst.write(built_fraction_raster, 1)  # Write data to band 1
    
    return built_fraction_raster



# function to rasterize mean building height
def rasterize_mean_building_height(mean_building_height, base_raster):

    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value_2 = ((geom,value) for geom, value in zip(mean_building_height.geometry, mean_building_height['height']))

    # Rasterize vector using the shape and transform of the raster
    mean_building_height_raster = features.rasterize(geom_value_2,
                                    out_shape = base_raster.shape,
                                    transform = base_raster.transform,
                                    all_touched = True,
                                    fill = 0,   # background value
                                    merge_alg = MergeAlg.replace,
                                    dtype = int16)

    # Save the new raster using rasterio
    with rasterio.open(
        RASTER_DIRECTORY / "mean_building_height.tif",
        "w",  # Write mode
        driver="GTiff",
        height=base_raster.shape[0],
        width=base_raster.shape[1],
        count=1,  # Single-band raster
        dtype=rasterio.int16,  # Match data type
        crs=mean_building_height.crs,  # Copy CRS from the original raster
        transform=base_raster.transform,  # Copy spatial transform
        nodata=0,  # Assign NoData value
    ) as dst:
        dst.write(mean_building_height_raster, 1)  # Write data to band 1
        
    return mean_building_height_raster
