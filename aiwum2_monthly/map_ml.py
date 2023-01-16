# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu
# In-situ groundwater use data and the proprietary PRISM 800 m data are provided by the USGS
# MAP ML project with the USGS and Colorado State University
#
# --------------------------------------- Running HydroMAP_ML -------------------------------------------------------
#
# Copy source files and saved CSV data to the HPC on Linux or Mac
# scp -r ./aiwum2/ "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/"
# scp -r ./Data/main/Annual_Subset.csv ./Data/main/VMP_readings_2014_2020.csv \
# "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/Data/main/"
#
# Copy source files and saved CSV data to the HPC on Windows powershell
# scp -r .\aiwum2\ "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/"
# scp -r .\Data\main\Annual_Subset.csv .\Data\main\VMP_readings_2014_2020.csv `
# "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/Data/main/"
#
# Execute map_ml.py (change the paths and flags accordingly) on Linux or Mac
# python map_ml.py \
# --input-rt-shp '../Data/RT/Realtime_WU_meter_site_2018–2021.shp' \
# --input-rt-xls '../Data/RT/2__Real-time_WU_daily_values_2018–2021.xlsx' \
# --field-shp-dir ../../HydroMAP_ML/USGS_MAP/permitted_boundaries/Shapefiles/ \
# --load-files False \
# --load-data False \
# --load-model False \
# --use-sub-cols True \
# --sub-cols Year Month AF_Acre lat_dd long_dd crop \
# --test-size 0.2 \
# --random-state 1234 \
# --output-dir ../Outputs/ \
# --model-dir ../Models/ \
# --pred-attr AF_Acre \
# --data-list SSEBop RO ppt tmax SWB_HSG SWB_IRR \
# --gee-scale 1000 \
# --prism-path ../../HydroMAP_ML/USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/PRISM/PRISM_800m/AN81/ \
# --cdl-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/CDL/ \
# --lanid-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/LANID/TIFs/ \
# --nhd-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/NHD/NHD_MAP_merged_nofish.shp \
# --map-extent-file '../Data/RT/Realtime_WU_meter_site_2018–2021.shp' \
# --train-year-list 2018 2019 2020 \
# --scaling True \
# --split-strategy 2 \
# --test-years 2020 \
# --load-map-csv False \
# --model-name DRF \
# --randomized-search False \
# --fold-count 5 \
# --repeats 3 \
# --outlier-op 2 \
# --gdal-path C:/OSGeo4W64/ \
# --gee-files RO \
# --prism-files ppt tmax \
# --swb-files SWB_IRR SWB_HSG \
# --load-pred-raster True \
# --load-pred-csv True \
# --load-map-extent True \
# --pred-year-list 2018 2019 2020 \
# --use-dask False \
# --swb-data-path ../../HydroMAP_ML/USGS_MAP/SWB_Data/ \
# --hsg-to-inf True \
# --pdp-plot-features All
#
# Execute map_ml.py (change the paths and flags accordingly) on Windows powershell
# python map_ml.py `
# --input-rt-shp '../Data/RT/Realtime_WU_meter_site_2018–2021.shp' `
# --input-rt-xls '../Data/RT/2__Real-time_WU_daily_values_2018–2021.xlsx' `
# --field-shp-dir ../../HydroMAP_ML/USGS_MAP/permitted_boundaries/Shapefiles/ `
# --load-files False `
# --load-data True `
# --load-model False `
# --use-sub-cols True `
# --sub-cols Year Month AF_Acre lat_dd long_dd crop `
# --test-size 0.2 `
# --random-state 1234 `
# --output-dir ../Outputs/ `
# --model-dir ../Models/ `
# --pred-attr AF_Acre `
# --data-list SSEBop RO ppt tmax SWB_HSG SWB_IRR `
# --gee-scale 1000 `
# --prism-path ../../HydroMAP_ML/USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/PRISM/PRISM_800m/AN81/ `
# --cdl-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/CDL/ `
# --lanid-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/LANID/TIFs/ `
# --nhd-path ../../HydroMAP_ML/AIWUM2_Data/Inputs/NHD/NHD_MAP_merged_nofish.shp `
# --map-extent-file '../Data/RT/Realtime_WU_meter_site_2018–2021.shp' `
# --train-year-list 2018 2019 2020 `
# --scaling True `
# --split-strategy 2 `
# --test-years 2020 `
# --load-map-csv False `
# --model-name DRF `
# --randomized-search False `
# --fold-count 5 `
# --repeats 3 `
# --outlier-op 2 `
# --gdal-path C:/OSGeo4W64/ `
# --gee-files RO SM_IDAHO `
# --prism-files ppt tmax `
# --swb-files SWB_IRR SWB_HSG `
# --load-pred-raster True `
# --load-pred-csv True `
# --load-map-extent True `
# --pred-year-list 2018 2019 2020 `
# --use-dask False `
# --swb-data-path ../../HydroMAP_ML/USGS_MAP/SWB_Data/ `
# --hsg-to-inf True `
# --pdp-plot-features All

# ------------------------------------------------- Main code begins --------------------------------------------------


import argparse

from maplibs.dataops import create_monthly_wu_csv, prepare_data, create_prediction_map, clean_file_dirs
from maplibs.dataops import create_train_test_data
from maplibs.mlops import build_ml_model, get_prediction_results, calc_train_test_metrics
from maplibs.mlops import create_pdplots
from maplibs.sysops import boolean_string


def run_map_ml(args):
    """
    Driver function to prepare data and run ML models.
    :param args: Namespace containing the following keys:

    Keys: Description
    ___________________________________________________________________________________________________________________
    input_rt_shp: Input MAP CSV file obtained from the USGS
    site_id_shp: Name of the Site ID field in the shapefile
    year_shp: Name of the year field in the shapefile
    crop_shp: Name of the crop field in the shapefile
    state_shp: Name of the state field in the shapefile
    acre_shp: Name of the acre field in the shapefile
    input_rt_xls: Input real-time data XLSX file
    site_id_xls: Name of the Site ID field in the XLS file
    dt_xls: Name of the date column in the XLS file
    state_list: List of states for model training
    field_shp_dir: Field polygon shapefile directory path (required for generating monthly rasters)
    load_files: Set True to load existing data sets
    load_data: Set to True to load already downloaded data
    load_model: Set True to load pre-built model
    use_sub_cols: Set True to make subset data using sub_cols
    sub_cols: List of columns for creating subset data
    lat_pump: Name of the latitude column of the VMP data
    lon_pump: Name of the longitude column of the VMP data
    test_size: Test data size
    random_state: PRNG seed
    output_dir: Output directory
    model_dir: Model directory
    pred_attr: Prediction/target attribute name
    data_list: List of data sets to use/download. Valid names include 'SSEBop', 'SM_IDAHO', 'MOD16', 'SMOS_SMAP',
    'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', 'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin',
    'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', 'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF',
    'SWB_RINF', 'SWB_RO', 'SWB_SS', 'SWB_MRD', 'SWB_SSM', 'SWB_AWC'
    Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for USDA-NASS cropland data,
    'SWB*' for SWB products.
    gee_scale: Google Earth Engine scale (m) for downloading
    prism_path: Path to the 800 m monthly PRISM products. Required if PRISM products are in data-list
    cdl_path: Path to the 30 m CDL products. Required if CDL is in data-list
    lanid_path: Path to the annual LANID TIFs
    nhd_path: Path to the MAP NHD shapefile
    openet_path: Path to the annual OpenET products. Required if OpenET is in data-list
    eemetric_path: Path to the monthly EEMETRIC products. Required if EEMETRIC is in data-list
    pt_jpl_path: Path to the monthly PT-JPL products. Required if PT-JPL is in data-list
    sims_path: Path to the monthly SIMS products. Required if SIMS is in data-list
    map_extent_file: Path to the MAP extent shapefile for handling OpenET, EEMETRIC, PT-JPL, and SIMS.
    Required if any or all of OpenET, EEMETRIC, PT-JPL, and SIMS are in data-list
    stratified_kfold: Set True to use repeated stratified k-fold to generate stratified splits based on the crop type
    train_year_list: Years to use in the model
    scaling: Whether to perform feature scaling (automatically False if crop_models is True)
    split_strategy: If 1, Split train test data based on year_col.
    If 2, then test_size amount of data from year_col or crop_col are kept for testing and rest for training;
    for this option, test-years should have some value other than None, else splitting is based on crop_col.
    For any other value of split-strategy, the data are randomly split.
    test_years: Years to purely keep as test data. split-attribute must be true to use this feature
    load_map_csv: Set True to use existing train and test data
    model_name: Set model name. Valid names include 'LGBM', 'DRF', 'RF', 'ETR', 'DT', 'BT', 'ABR', 'KNN', 'LR'
    randomized_search: Set False to use the exhaustive GridSearchCV
    fold_count: Number of folds for kFold
    repeats: Number of repeats for kFold
    drop_attr: Attributes to drop from the modeling process
    outlier_op: Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
    by each crop, 3 for removing outliers by each year, or 4 for removing as per AIWUM 1 based on irrigation thresholds.
    Note: for this project we only process outliers above the boxplot upper limit for 1-3.
    gdal_path: GDAL path. This is required if AttributeError occurs
    gee_files: GEE data set names such as MOD16, RO, etc. This is required if compare-aiwums is True
    prism_files: PRISM data set names such as ppt, tmax, tmin, tmean. This is required if compare-aiwums is True
    swb_files: SWB data set names such as SWB_IRR, SWB_HSG, etc. This is required if compare-aiwums is True
    load_pred_raster: Load existing prediction rasters, required if compare-aiwums is True
    load_pred_csv: Load existing prediction CSVs, required if compare-aiwums is True
    load_map_extent: Set True to load existing MAP extent rasters, required if compare-aiwums is True
    pred_year_list: Years to predict
    use_dask: Flag for using dask
    swb_data_path: Path to the SWB data sets
    hsg_to_inf: Set False to disable creating the infiltration rate column based on the HSGs. Only works when
    SWB_HSG is present in data_list and not in drop_attr
    pdp_plot_features: List of features to use for generating PDP plots. These should match items in data_list.
    Set 'All' to use all the features used for  model training.
    ___________________________________________________________________________________________________________________
    :return: None
    """

    if not args.use_sub_cols:
        args.sub_cols = []
    monthly_df = create_monthly_wu_csv(
        args.input_rt_shp,
        args.input_rt_xls,
        args.output_dir,
        site_id_shp=args.site_id_shp,
        site_id_xls=args.site_id_xls,
        year_shp=args.year_shp,
        state_shp=args.state_shp,
        acre_shp=args.acre_shp,
        crop_shp=args.crop_shp,
        dt_xls=args.dt_xls,
        year_list=args.train_year_list,
        state_list=args.state_list,
        load_csv=True
    )
    monthly_df, file_dirs = prepare_data(
        monthly_df,
        args.output_dir,
        args.sub_cols,
        data_list=args.data_list,
        already_prepared=args.load_files,
        skip_download=args.load_data,
        gee_scale=args.gee_scale,
        prism_data_path=args.prism_path,
        cdl_data_path=args.cdl_path,
        openet_data_path=args.openet_path,
        eemetric_data_path=args.eemetric_path,
        pt_jpl_data_path=args.pt_jpl_path,
        sims_data_path=args.sims_path,
        swb_data_path=args.swb_data_path,
        map_extent_file=args.map_extent_file,
        year_col=args.year_shp,
        lat_pump=args.lat_shp,
        lon_pump=args.lon_shp
    )
    ret_vals = create_train_test_data(
        monthly_df,
        args.output_dir,
        pred_attr=args.pred_attr,
        drop_attr=args.drop_attr,
        test_size=args.test_size,
        random_state=args.random_state,
        scaling=args.scaling,
        already_created=args.load_map_csv,
        year_col=args.year_shp,
        crop_col=args.crop_shp,
        year_list=args.train_year_list,
        split_strategy=args.split_strategy,
        test_year=args.test_years,
        outlier_op=args.outlier_op,
        hsg_to_inf=args.hsg_to_inf
    )
    x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test, crop_train, crop_test = ret_vals
    stratify_labels = crop_train
    if args.test_years:
        stratify_labels = year_train
    model = build_ml_model(
        x_train, y_train, args.model_dir,
        args.model_name, args.random_state,
        args.load_model, args.fold_count,
        args.repeats, y_scaler,
        args.randomized_search,
        args.stratified_kfold, args.use_dask,
        stratify_labels=stratify_labels
    )
    pred_df = get_prediction_results(
        model, x_train, x_test,
        y_train, y_test, x_scaler,
        y_scaler, year_train,
        year_test, args.model_dir,
        args.model_name, args.year_col,
        args.crop_col, crop_train, crop_test
    )
    calc_train_test_metrics(pred_df, args.crop_col, args.year_col)
    if args.pdp_plot_features:
        create_pdplots(
            x_train, model,
            args.pdp_plot_features,
            args.output_dir,
            args.scaling,
            args.random_state
        )
    # pred_data_list = [dl for dl in args.data_list if dl not in args.drop_attr]
    # file_dirs = clean_file_dirs(
    #     file_dirs, args.drop_attr,
    #     cdl_data_path=args.cdl_path,
    #     openet_data_path=args.openet_path,
    #     eemetric_data_path=args.eemetric_path,
    #     pt_jpl_data_path=args.pt_jpl_path,
    #     sims_data_path=args.sims_path
    # )
    # _, pred_wu_dir, map_extent_raster_dir = create_prediction_map(
    #     model, args.map_extent_file, file_dirs,
    #     args.output_dir, args.pred_year_list,
    #     pred_data_list, args.cdl_path, args.nhd_path,
    #     args.lanid_path, args.field_shp_dir,
    #     args.start_month, args.end_month,
    #     args.crop_col, args.load_pred_csv,
    #     args.load_map_extent, args.load_pred_raster,
    #     args.gee_files, args.prism_files, args.swb_files,
    #     x_scaler, y_scaler, args.gdal_path
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flags to run HydroMAP_ML')
    parser.add_argument('--input-rt-shp', type=str, required=True, help='Input real-time flowmeter shapefile')
    parser.add_argument('--site-id-shp', type=str, default='Site_numbe', help='Name of the Site ID field in the shapefile')
    parser.add_argument('--year-shp', type=str, default='Year', help='Name of the year field in the shapefile')
    parser.add_argument('--crop-shp', type=str, default='crop', help='Name of the crop field in the shapefile')
    parser.add_argument('--state-shp', type=str, default='State', help='Name of the state field in the shapefile')
    parser.add_argument('--acre-shp', type=str, default='combined_a', help='Name of the acre field in the shapefile')
    parser.add_argument('--lat-shp', type=str, default='lat_dd', help='Name of the latitude field in the shapefile')
    parser.add_argument('--lon-shp', type=str, default='long_dd',
                        help='Name of the longitude field in the shapefile')
    parser.add_argument('--input-rt-xls', type=str, required=True, help='Input real-time data XLSX file')
    parser.add_argument('--site-id-xls', type=str, default='site_no', help='Name of the Site ID field in the XLS file')
    parser.add_argument('--dt-xls', type=str, default='datetime', help='Name of the date column in the XLS file')
    parser.add_argument('--state-list', type=str, nargs='+', default=[], help='List of states for model training')
    parser.add_argument('--field-shp-dir', type=str, required=True,
                        help='Field polygon shapefile directory path (required for generating monthly rasters)')
    parser.add_argument('--load-files', type=boolean_string, default=True,
                        help='Set True to load existing data sets')
    parser.add_argument('--load-data', type=boolean_string, default=True,
                        help='Set to True to load already downloaded data')
    parser.add_argument('--load-model', type=boolean_string, default=False,
                        help='Set True to load pre-built model')
    parser.add_argument('--use-sub-cols', type=boolean_string, default=False,
                        help='Set True to make subset data using sub-cols')
    parser.add_argument('--sub-cols', type=str, nargs='+', default=[],
                        help='List of columns for creating subset data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test data size')
    parser.add_argument('--random-state', type=int, default=43, help='PRNG seed')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model-dir', type=str, required=True, help='Model directory')
    parser.add_argument('--pred-attr', type=str, required=True, help='Prediction/target attribute name')
    parser.add_argument('--data-list', type=str, nargs='+', required=True,
                        help="List of data sets to use/download. Valid names include 'SSEBop', 'SM_IDAHO', 'MOD16',"
                             " 'SMOS_SMAP', 'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', "
                             "'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin', 'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', "
                             "'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF', 'SWB_RINF', 'SWB_RO', "
                             "'SWB_SS', 'SWB_MRD', 'SWB_SSM', 'SWB_AWC'"
                             "Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, "
                             "'CDL' for USDA-NASS cropland data, 'SWB*' for SWB products")
    parser.add_argument('--gee-scale', type=int, default=1000, help='Google Earth Engine scale (m) for downloading')
    parser.add_argument('--prism-path', type=str, default='',
                        help='Path to the 800 m monthly PRISM products. Required if PRISM products are in data-list')
    parser.add_argument('--cdl-path', type=str, default='',
                        help='Path to the 30 m CDL products. Required for generating AIWUM 2 rasters and/or '
                             'if CDL is in data-list')
    parser.add_argument('--lanid-path', type=str, default='',
                        help='Path to the 30 m LANID TIFs. Required for generating AIWUM 2 rasters')
    parser.add_argument('--nhd-path', type=str, default='', help='Path to the MAP NHD shapefile')
    parser.add_argument('--openet-path', type=str, default='',
                        help='Path to the annual OpenET products. Required if OpenET is in data-list')
    parser.add_argument('--eemetric-path', type=str, default='',
                        help='Path to the monthly EEMETRIC products. Required if EEMETRIC is in data-list')
    parser.add_argument('--pt-jpl-path', type=str, default='',
                        help='Path to the monthly PT-JPL products. Required if PT-JPL is in data-list')
    parser.add_argument('--sims-path', type=str, default='',
                        help='Path to the monthly SIMS products. Required if SIMS is in data-list')
    parser.add_argument('--map-extent-file', type=str, default='',
                        help='Path to the MAP extent shapefile for handling OpenET, EEMETRIC, PT-JPL, and SIMS. '
                             'Required if any or all of OpenET, EEMETRIC, PT-JPL, and SIMS are in data-list')
    parser.add_argument('--stratified-kfold', type=boolean_string, default=False,
                        help='Set True to use repeated stratified k-fold to generate stratified splits based on the '
                             'crop type')
    parser.add_argument('--train-year-list', type=int, nargs='+', required=True, help='Years to train the model')
    parser.add_argument('--scaling', type=boolean_string, default=True, help='Whether to perform feature scaling')
    parser.add_argument('--split-strategy', type=int, default=3,
                        help='If 1, Split train test data based on a particular attribute like year_col or crop_col.'
                             'If 2, then test_size amount of data from year_col or crop_col are kept for testing and '
                             'rest for training; for this option, test-years should have some value '
                             'other than None, else splitting is based on crop_col. '
                             'For any other value of split-strategy, the data are randomly split.')
    parser.add_argument('--test-years', type=int, nargs='+', default=None,
                        help='Years to purely keep as test data. split-attribute must be true to use this feature')
    parser.add_argument('--load-map-csv', type=boolean_string, default=False,
                        help='Set True to use existing train and test data')
    parser.add_argument('--model-name', type=str, required=True,
                        help="Set model name. Valid names include 'LGBM', 'DRF', 'RF', 'ETR', 'DT', 'BT', 'ABR', "
                             "'KNN', 'LR'")
    parser.add_argument('--randomized-search', type=boolean_string, default=True,
                        help='Set False to use the exhaustive GridSearchCV')
    parser.add_argument('--fold-count', type=int, default=5, help='Number of folds for kFold')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats for kFold')
    parser.add_argument('--drop-attr', type=str, nargs='+', default=[],
                        help='Attributes to drop from the modeling process')
    parser.add_argument('--outlier-op', type=int, default=None,
                        help='Outlier operation to perform. Set to 1 for removing outlier directly, '
                             '2 for removing outlier by each crop, 3 for removing outliers by each year,'
                             '4 for removing as per AIWUM 1 based on irrigation thresholds')
    parser.add_argument('--gdal-path', type=str, help='GDAL path. This is required in case an AttributeError occurs')
    parser.add_argument('--gee-files', type=str, nargs='+', default=[],
                        help='GEE data set names such as MOD16, RO, etc. This is required if compare-aiwums is True')
    parser.add_argument('--prism-files', type=str, nargs='+', default=[],
                        help='PRISM data set names such as ppt, tmax, tmin, tmean. '
                             'This is required if compare-aiwums is True')
    parser.add_argument('--swb-files', type=str, nargs='+', default=[],
                        help='SWB data set names such as SWB_IRR, SWB_HSG, etc. '
                             'This is required if compare-aiwums is True')
    parser.add_argument('--load-pred-raster', type=boolean_string, default=False,
                        help='Load existing prediction rasters, required if compare-aiwums is True')
    parser.add_argument('--load-pred-csv', type=boolean_string, default=False,
                        help='Load existing prediction CSVs, required if compare-aiwums is True')
    parser.add_argument('--load-map-extent', type=boolean_string, default=False,
                        help='Set True to load existing MAP extent rasters, required if compare-aiwums is True')
    parser.add_argument('--pred-year-list', type=int, nargs='+', required=True, help='Years to predict')
    parser.add_argument('--use-dask', type=boolean_string, default=True, help='Set False to disable Dask')
    parser.add_argument('--swb-data-path', type=str, required=True, help='Path to SWB data')
    parser.add_argument('--hsg-to-inf', type=boolean_string, default=True,
                        help='Set False to disable creating the infiltration rate column based on the HSGs. '
                             'Only works when SWB_HSG is present in --data-list')
    parser.add_argument('--pdp-plot-features', type=str, nargs='+', default=[],
                        help="Features to use for generating PDP plots. Set 'All' to use all the features "
                             "used for model training.")
    map_args = parser.parse_args()
    run_map_ml(map_args)
