# AIWUM 2.1

Authors: [Sayantan Majumdar](https://scholar.google.com/citations?user=iYlO-VcAAAAJ&hl=en) [sayantan.majumdar@dri.edu], [Ryan Smith](https://scholar.google.com/citations?user=nzSrr8oAAAAJ&hl=en) [ryan.g.smith@colostate.edu], and [Vincent E. White](https://www.usgs.gov/staff-profiles/vincent-e-white) [vwhite@usgs.gov]

<img src="../Readme_Figures/USGS_logo.png" height="40"/> &nbsp; <img src="../Readme_Figures/CSU-Signature-357-617.png" height="50"/> &nbsp; <img src="../Readme_Figures/official-dri-logotag-trans-bkgd.png" height="40"/>


Note: In-situ groundwater use data and the proprietary PRISM 800 m data are provided by the USGS.
MAP ML project with the USGS, Colorado State University, and Desert Research Institute. This software has been successfully
tested on three systems&mdash; [Alienware M17R1 2020](https://www.dell.com/en-us/gaming/alienware) (Windows 10 Home), [Alpine HPC](https://curc.readthedocs.io/en/latest/clusters/alpine/quick-start.html) (RedHat Enterprise Linux version 8),
and the [Apple MacBook Pro 2023](https://www.apple.com/macbook-pro/) (macOS Ventura 13.4.1).

## Citations
**Software**: Majumdar, S., Smith, R., and White, V.E., 2023, Aquaculture and Irrigation Water Use Model 2.0 Repository: U.S. Geological Survey data release, https://doi.org…

**Data Release**: Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., Bristow, E.L., Rigby, J.R., Kress, W.H., Painter, J.A., and White, V.E., 2023, Aquaculture and Irrigation Water Use Model (AIWUM) 2.0 input and output datasets, https://doi.org/10.5066/P9CET25K.

## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If you have Anaconda or miniconda installed already, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. You may have to restart the current shell session to add conda to path.

You could update to the latest conda package manager by running ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help
you install the correct packages and Python version to run the codes.

### 2. Clone or download the repository

Download the repository from the compressed file link at the top right of the repository webpage, or clone the repository using Git.
Unzip all zipped files.  Several of the input datasets in this repository are zipped for efficient storage and must be unzipped before they can be used to run this project.

#### Repository disk space requirements
The uncompressed Input data size is around 55.6 GB. Without the PRISM 800 m precipitation (ppt) and maximum temperature (tmax) datasets, 
the Input data size is around 6.6 GB. If you do not have access to these proprietary PRISM datasets, pass 'PRISM' and 'TMAX' to 
data-list (instead of 'ppt' and 'tmax') for downloading the free version (4 km spatial resolution) from Google Earth Engine. 
These 4 km PRISM datasets are provided in the Inputs directory as well. See the last section of this document to understand 
the various input arguments.

The Output directory size (including intermediate files) is  84.7 GB. Therefore, make sure to have around 92 GB free space if you are not using the proprietary PRISM data.
If you are using the 800 m PRISM product, then around 141 GB free space is recommended.

### 3. Creating the conda environment and installing packages
Open Linux/Mac terminal or Windows PowerShell and run the following:
```
conda create -y -n aiwum2 python=3.10
conda activate aiwum2
conda install -y -c conda-forge rioxarray geopandas lightgbm earthengine-api rasterstats seaborn openpyxl
conda install -y -c conda-forge dask-ml dask-jobqueue swifter # may take a while to solve dependencies
```

Once the above steps are successfully executed, run the following to load the GDAL_DATA environment variable which is needed by 
rasterio.

```
conda deactivate aiwum2  
conda activate aiwum2
```

### 4. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk).

### 5. Running AIWUM 2.0
Make sure that the aiwum2 conda environment is active. If not, run ```conda activate aiwum2``` before running the following codes.
#### Linux/Mac Terminal:
```
cd aiwum2/ 
python map_ml.py \
--input-rt-shp <rt-shp-path> \
--input-rt-xls <rt-xls-path> \
--vmp-csv <vmp-csv-path> \
--field-shp-dir <shp-dir-path> \
--load-files <True/False> \
--load-data <True/False> \
--load-model <True/False> \
--data-list <data1 data2 ... datan> \
--prism-path <prism-path> \
--cdl-path <cdl-path> \
--lanid-path <lanid-path> \
--nhd-path <nhd-path> \
--swb-data-path <swb-path> \
--map-extent-file <extent-shp-path> \
--train-year-list <year1 year2 ... yearn> \
--pred-year-list <year1 year2 ... yearn> \
--load-map-csv <True/False> \
--load-pred-raster <True/False> \
--load-pred-file <True/False> \
--load-map-extent <True/False> \
--model-name <model-name> \
--compare-aiwums <True/False> \
--aiwum1-monthly-tot-dir <aiwum1-monthly-dir-path>
```
##### Example run format (Linux/Mac Terminal) using the proprietary 800 m data PRISM datasets
```
cd aiwum2/ 
python map_ml.py \
--input-rt-shp ../AIWUM2_Data/Inputs/Real-time/Realtime_WU_meter_site_2018–2021.shp \
--input-rt-xls ../AIWUM2_Data/Inputs/Real-time/2__Real-time_WU_daily_values_2018–2021.xlsx \
--vmp-csv ../AIWUM2_Data/Inputs/VMP_Readings_Latest_2014_2020.csv \
--field-shp-dir ../AIWUM2_Data/Inputs/Permitted_Boundaries/Shapefiles/ \
--load-files True \
--load-data True \
--load-model False \
--data-list SSEBop SM_IDAHO ppt tmax RO \
--prism-path ../AIWUM2_Data/Inputs/PRISM/PRISM_800m/AN81/ \
--cdl-path ../AIWUM2_Data/Inputs/CDL/ \
--lanid-path ../AIWUM2_Data/Inputs/LANID/ \
--nhd-path ../AIWUM2_Data/Inputs/NHD/NHD_MAP_merged_nofish.shp \
--swb-data-path ../AIWUM2_Data/Inputs/SWB/ \
--map-extent-file ../AIWUM2_Data/Inputs/Model_Extent/Model_Extent.shp \
--train-year-list 2014 2015 2016 2017 2018 2019 2020 2021 \
--pred-year-list 2014 2015 2016 2017 2018 2019 2020 2021 \
--load-map-csv False \
--load-pred-raster False \
--load-pred-file True \
--load-map-extent True \
--model-name LGBM \
--compare-aiwums False \
--aiwum1-monthly-tot-dir ../AIWUM2_Data/Inputs/AIWUM1_Monthly_Total/
```

#### Windows PowerShell:
```
cd aiwum2\ 
python map_ml.py `
--input-rt-shp <rt-shp-path> `
--input-rt-xls <rt-xls-path> `
--vmp-csv <vmp-csv-path> `
--field-shp-dir <shp-dir-path> `
--load-files <True/False> `
--load-data <True/False> `
--load-model <True/False> `
--data-list <data1 data2 ... datan> `
--prism-path <prism-path> `
--cdl-path <cdl-path> `
--lanid-path <lanid-path> `
--nhd-path <nhd-path> `
--swb-data-path <swb-path> `
--map-extent-file <extent-shp-path> `
--train-year-list <year1 year2 ... yearn> `
--pred-year-list <year1 year2 ... yearn> `
--load-map-csv <True/False> `
--load-pred-raster <True/False> `
--load-pred-file <True/False> `
--load-map-extent <True/False> `
--model-name <model-name> `
--compare-aiwums <True/False> `
--aiwum1-monthly-tot-dir <aiwum1-dir-path>
```

##### Example run format (Windows PowerShell) using the free 4 km PRISM datasets
```
cd aiwum2\
python map_ml.py `
--input-rt-shp ../AIWUM2_Data/Inputs/Real-time/Realtime_WU_meter_site_2018–2021.shp `
--input-rt-xls ../AIWUM2_Data/Inputs/Real-time/2__Real-time_WU_daily_values_2018–2021.xlsx `
--vmp-csv ../AIWUM2_Data/Inputs/VMP_Readings_Latest_2014_2020.csv `
--field-shp-dir ../AIWUM2_Data/Inputs/Permitted_Boundaries/Shapefiles/ `
--load-files False `
--load-data False `
--load-model False `
--data-list SSEBop SM_IDAHO PPT TMAX RO `
--cdl-path ../AIWUM2_Data/Inputs/CDL/ `
--lanid-path ../AIWUM2_Data/Inputs/LANID/ `
--nhd-path ../AIWUM2_Data/Inputs/NHD/NHD_MAP_merged_nofish.shp `
--swb-data-path ../AIWUM2_Data/Inputs/SWB/ `
--map-extent-file ../AIWUM2_Data/Inputs/Model_Extent/Model_Extent.shp `
--train-year-list 2014 2015 2016 2017 2018 2019 2020 2021 `
--pred-year-list 2014 2015 2016 2017 2018 2019 2020 2021 `
--load-map-csv False `
--load-pred-raster False `
--load-pred-file False `
--load-map-extent False `
--model-name LGBM `
--compare-aiwums False `
--aiwum1-monthly-tot-dir ../AIWUM2_Data/Inputs/AIWUM1_Monthly_Total/
```

#### Other usage notes
``python map_ml.py -h`` can be run to know about all the required and optional command line arguments as follows:
```
usage: map_ml.py [-h] --input-rt-shp INPUT_RT_SHP [--site-id-shp SITE_ID_SHP] [--year-shp YEAR_SHP] [--crop-shp CROP_SHP] [--state-shp STATE_SHP] [--acre-shp ACRE_SHP] [--lat-shp LAT_SHP] [--lon-shp LON_SHP] --input-rt-xls INPUT_RT_XLS [--site-id-xls SITE_ID_XLS] [--dt-xls DT_XLS] [--state-list STATE_LIST [STATE_LIST ...]] [--vmp-csv VMP_CSV] --field-shp-dir FIELD_SHP_DIR
                 [--load-files LOAD_FILES] [--load-data LOAD_DATA] [--load-model LOAD_MODEL] [--use-sub-cols USE_SUB_COLS] [--sub-cols SUB_COLS [SUB_COLS ...]] [--lat-pump LAT_PUMP] [--lon-pump LON_PUMP] [--field-permit-col FIELD_PERMIT_COL] [--test-size TEST_SIZE] [--random-state RANDOM_STATE] [--output-dir OUTPUT_DIR] [--model-dir MODEL_DIR] [--pred-attr PRED_ATTR]
                 [--pred-start-month PRED_START_MONTH] [--pred-end-month PRED_END_MONTH] --data-list DATA_LIST [DATA_LIST ...] [--gee-scale GEE_SCALE] [--prism-path PRISM_PATH] --cdl-path CDL_PATH --lanid-path LANID_PATH --nhd-path NHD_PATH [--openet-path OPENET_PATH] [--eemetric-path EEMETRIC_PATH] [--pt-jpl-path PT_JPL_PATH] [--sims-path SIMS_PATH] --map-extent-file
                 MAP_EXTENT_FILE [--stratified-kfold STRATIFIED_KFOLD] --train-year-list TRAIN_YEAR_LIST [TRAIN_YEAR_LIST ...] [--scaling SCALING] [--split-strategy SPLIT_STRATEGY] [--test-years TEST_YEARS [TEST_YEARS ...]] [--load-map-csv LOAD_MAP_CSV] [--model-name MODEL_NAME] [--randomized-search RANDOMIZED_SEARCH] [--fold-count FOLD_COUNT] [--repeats REPEATS]
                 [--drop-attr DROP_ATTR [DROP_ATTR ...]] [--outlier-op OUTLIER_OP] [--compare-aiwums COMPARE_AIWUMS] [--load-pred-raster LOAD_PRED_RASTER] [--load-pred-file LOAD_PRED_FILE] [--load-map-extent LOAD_MAP_EXTENT] [--aiwum1-monthly-tot-dir AIWUM1_MONTHLY_TOT_DIR] --pred-year-list PRED_YEAR_LIST [PRED_YEAR_LIST ...] [--use-dask USE_DASK] --swb-data-path SWB_DATA_PATH
                 [--hsg-to-inf HSG_TO_INF] [--volume-units VOLUME_UNITS] [--pdp-plot-features PDP_PLOT_FEATURES [PDP_PLOT_FEATURES ...]] [--calc-cc CALC_CC] [--calc-relative-et CALC_RELATIVE_ET] [--calc-eff-ppt CALC_EFF_PPT]

Flags to run AIWUM 2.0

options:
  -h, --help            show this help message and exit
  --input-rt-shp INPUT_RT_SHP
                        Input real-time flowmeter shapefile (default: None)
  --site-id-shp SITE_ID_SHP
                        Name of the Site ID field in the shapefile (default: Site_numbe)
  --year-shp YEAR_SHP   Name of the year field in the shapefile (default: Year)
  --crop-shp CROP_SHP   Name of the crop field in the shapefile (default: crop)
  --state-shp STATE_SHP
                        Name of the state field in the shapefile (default: State)
  --acre-shp ACRE_SHP   Name of the acre field in the shapefile (default: combined_a)
  --lat-shp LAT_SHP     Name of the latitude field in the shapefile (default: lat_dd)
  --lon-shp LON_SHP     Name of the longitude field in the shapefile (default: long_dd)
  --input-rt-xls INPUT_RT_XLS
                        Input real-time data XLSX file (default: None)
  --site-id-xls SITE_ID_XLS
                        Name of the Site ID field in the XLS file (default: site_no)
  --dt-xls DT_XLS       Name of the date column in the XLS file (default: datetime)
  --state-list STATE_LIST [STATE_LIST ...]
                        List of MAP states (abbreviated, e.g, LA, MS, AR, etc.) for model training (default: [])
  --vmp-csv VMP_CSV     VMP CSV file if both VMP and real-time data are to be merged after disaggregating the VMP data with real-time weights (default: )
  --field-shp-dir FIELD_SHP_DIR
                        Field polygon shapefile directory path (default: None)
  --load-files LOAD_FILES
                        Set True to load existing data sets (default: False)
  --load-data LOAD_DATA
                        Set to True to load already downloaded data (default: False)
  --load-model LOAD_MODEL
                        Set True to load pre-built model (default: False)
  --use-sub-cols USE_SUB_COLS
                        Set True to make subset data using sub-cols (default: True)
  --sub-cols SUB_COLS [SUB_COLS ...]
                        List of columns for creating subset data (default: ['Year', 'Month', 'AF_Acre', 'lat_dd', 'long_dd', 'crop', 'State', 'Data'])
  --lat-pump LAT_PUMP   Name of the latitude column of the VMP data (default: Latitude)
  --lon-pump LON_PUMP   Name of the longitude column of the VMP data (default: Longitude)
  --field-permit-col FIELD_PERMIT_COL
                        Field permit column name (default: PermitNumb)
  --test-size TEST_SIZE
                        Test data size (default: 0.2)
  --random-state RANDOM_STATE
                        PRNG seed (default: 1234)
  --output-dir OUTPUT_DIR
                        Output directory (default: ../Outputs/)
  --model-dir MODEL_DIR
                        Model directory (default: ../Models/)
  --pred-attr PRED_ATTR
                        Prediction/target attribute name (default: AF_Acre)
  --pred-start-month PRED_START_MONTH
                        Start month for predicting water use rasters (default: 1)
  --pred-end-month PRED_END_MONTH
                        End month for predicting water use rasters (default: 12)
  --data-list DATA_LIST [DATA_LIST ...]
                        List of data sets to use/download. Valid names include 'SSEBop', 'SM_IDAHO', 'MOD16', 'SMOS_SMAP', 'DROUGHT', 'PPT', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', 'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin', 'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', 'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF', 'SWB_RINF', 'SWB_RO', 'SWB_SS',
                        'SWB_MRD', 'SWB_SSM', 'SWB_AWC'Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for USDA-NASS cropland data, 'SWB*' for SWB products (default: None)
  --gee-scale GEE_SCALE
                        Google Earth Engine scale (m) for downloading (default: 1000)
  --prism-path PRISM_PATH
                        Path to the 800 m monthly PRISM products. Required if PRISM 800 m products are in data-list (default: )
  --cdl-path CDL_PATH   Path to the 30 m CDL products. Required for generating AIWUM 2 rasters and/or if CDL is in data-list (default: None)
  --lanid-path LANID_PATH
                        Path to the 30 m LANID TIFs. Required for generating AIWUM 2 rasters (default: None)
  --nhd-path NHD_PATH   Path to the MAP NHD shapefile (default: None)
  --openet-path OPENET_PATH
                        Path to the annual OpenET products. Required if OpenET is in data-list (default: )
  --eemetric-path EEMETRIC_PATH
                        Path to the monthly EEMETRIC products. Required if EEMETRIC is in data-list (default: )
  --pt-jpl-path PT_JPL_PATH
                        Path to the monthly PT-JPL products. Required if PT-JPL is in data-list (default: )
  --sims-path SIMS_PATH
                        Path to the monthly SIMS products. Required if SIMS is in data-list (default: )
  --map-extent-file MAP_EXTENT_FILE
                        Path to the MAP extent shapefile (default: )
  --stratified-kfold STRATIFIED_KFOLD
                        Set True to use repeated stratified k-fold to generate stratified splits based on the crop type (default: True)
  --train-year-list TRAIN_YEAR_LIST [TRAIN_YEAR_LIST ...]
                        Years to train the model (default: None)
  --scaling SCALING     Whether to perform feature scaling (default: True)
  --split-strategy SPLIT_STRATEGY
                        If 1, Split train test data based on a particular attribute like year_col or crop_col.If 2, then test_size amount of data from year_col or crop_col are kept for testing and rest for training; for this option, test-years should have some value other than None, else splitting is based on crop_col. For any other value of split-strategy, the data are randomly
                        split. (default: 2)
  --test-years TEST_YEARS [TEST_YEARS ...]
                        If split-strategy=3, test-size amount of data from each year is selected as test data.If split-strategy=2, then test-years are purely kept for testing. (default: [2020])
  --load-map-csv LOAD_MAP_CSV
                        Set True to use existing train and test data (default: False)
  --model-name MODEL_NAME
                        Set model name. Valid names include 'LGBM', 'DRF', 'RF', 'ETR', 'DT', 'BT', 'ABR', 'KNN', 'LR' (default: DRF)
  --randomized-search RANDOMIZED_SEARCH
                        Set False to use the exhaustive GridSearchCV (default: True)
  --fold-count FOLD_COUNT
                        Number of folds for kFold (default: 5)
  --repeats REPEATS     Number of repeats for kFold (default: 3)
  --drop-attr DROP_ATTR [DROP_ATTR ...]
                        Attributes to drop from the modeling process (default: ['Year', 'Month', 'PermitNumb', 'State', 'Data'])
  --outlier-op OUTLIER_OP
                        Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outlier by each crop, 3 for removing outliers by each year,4 for removing as per AIWUM 1 based on irrigation thresholds (default: 2)
  --compare-aiwums COMPARE_AIWUMS
                        Set True to compare AIWUM 1.1 and 2.0 monthly rasters (default: False)
  --load-pred-raster LOAD_PRED_RASTER
                        Load existing prediction rasters. (default: False)
  --load-pred-file LOAD_PRED_FILE
                        Load existing predictor parquet files. (default: False)
  --load-map-extent LOAD_MAP_EXTENT
                        Set True to load existing MAP extent rasters. (default: False)
  --aiwum1-monthly-tot-dir AIWUM1_MONTHLY_TOT_DIR
                        AIWUM 1.1 monthly prediction raster path, required if compare-aiwums is True (default: None)
  --pred-year-list PRED_YEAR_LIST [PRED_YEAR_LIST ...]
                        Years to predict (default: None)
  --use-dask USE_DASK   Set False to disable Dask (default: False)
  --swb-data-path SWB_DATA_PATH
                        Path to SWB data (default: None)
  --hsg-to-inf HSG_TO_INF
                        Set False to disable creating the infiltration rate column based on the HSGs. Only works when SWB_HSG is present in --data-list (default: True)
  --volume-units VOLUME_UNITS
                        Set False to use mm as water use units instead of m3. Only applies to AIWUM 2 predicted water use rasters. (default: True)
  --pdp-plot-features PDP_PLOT_FEATURES [PDP_PLOT_FEATURES ...]
                        Features to use for generating PDP plots. Set 'All' to use all the features used for model training. (default: [])
  --calc-cc CALC_CC     Set True to add crop coefficients as a predictor. (default: False)
  --calc-relative-et CALC_RELATIVE_ET
                        Set True to add relative ETs of all ET predictors as predictors. (default: False)
  --calc-eff-ppt CALC_EFF_PPT
                        Set True to add effective precipitation using as a predictor. Only works if SWB_INF and SWB_PPT are in data-list (default: False)
```