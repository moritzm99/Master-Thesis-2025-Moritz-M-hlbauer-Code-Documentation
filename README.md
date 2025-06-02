This repository documents coding conducted by Moritz Mühlbauer for the Masters Thesis 'Mapping Surface Urban Heat Island intensity and the influence of urbanization and heatwaves through Local Climate Zone mapping  Tallinn - 2014 and 2022'
# Hot-Spot Analysis and data visualization with `Copernicus Marine Toolbox`

- this repository was mainly created for `Archipelagos` interns to maintain data harvesting from `Copernicus Ocean Products` for marine research purposes
- this is a 'from scratch guide' starting from anaconda installation and copernicus API setup to more sophisticated data querying and visualization
- expected runtime on a 32 GB RAM machine: ~6 min

## Requirements
### Step 0:
- a working package/file/environment manager like anaconda/miniconda or micromamba with a Python > 3 installation:
  - refer to the Archipelagos GIS Google Drive (contact Tim Grandjean for access [here](mailto:t.grandjean@archipelago.gr)) or get instructions directly from the providers:
    - [here](https://www.anaconda.com/download) for anaconda/miniconda
    - [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for micromamba

- an Interactive Development Environment (IDE) or a Notebook-based Interactive Computing Environment:
  - this repository was built using Jupyter lab (read more details [here](https://jupyter.org))
  - after sucessful conda installation follow the next steps to get your instance
  - it is highly recommended to install Juypter lab in a seperate environment and link other environment Kernels to this one instance
 
    ```bash
    # Create and Activate a jupyter Environment
    conda create -n jupyter python=3.9 jupyterlab ipykernel -y
    conda activate jupyter
    ```
    
    ```bash
    # Install Juypter
    conda install -c conda-forge jupyterlab
    ```

### Step 1: Establish connection to Copernicus Server

A video tutorial can be found [here](https://vimeo.com/943705586) and in more detailed written form [here](https://help.marine.copernicus.eu/en/articles/7970514-copernicus-marine-toolbox-installation)

#### **1. Install the Toolbox from the conda-forge channel**
```bash
conda create --name copernicusmarine -c conda-forge copernicusmarine --yes
```

#### **2. Activate copernicusmarine environment**
```bash
conda activate copernicusmarine
```

#### **3. Verify installation**
```bash
conda list  # Lists installed packages
```

#### **4. Install a ipykernel**
```bash
# Inside the copernicusmarine environment, install the ipykernel package
conda install -c conda-forge ipykernel -y
```

#### **5. Register this environment as a Jupyter Kernel**
```bash
# This registers the copernicusmarine environment inside Jupyter
python -m ipykernel install --user --name copernicusmarine --display-name "CMT_Kernel"
```

### Step 2: Install necessary libraries 
```bash
conda env update --name copernicusmarine --file requirements.yml # find requirments.yml in this repository
```

### Step 3: Start the installation

#### **1. Activate juypter environment**
```bash
conda activate jupyter # navigate to your terminal/shell 
```

#### **2. Start Juypter lab**
```bash
jupyter lab # this will automatically open a new browser window and host the Jupter IDE
```

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
