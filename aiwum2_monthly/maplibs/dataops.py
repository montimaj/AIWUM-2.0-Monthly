# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu


import pandas as pd
import geopandas as gpd
import ee
import requests
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn.utils as sk
from pyproj import Transformer
from glob import glob
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from collections import defaultdict
from .sysops import makedirs, make_proper_dir_name, copy_files
from .mlops import get_prediction_stats
from .rasterops import read_raster_as_arr, reproject_coords, write_raster, create_long_lat_grid
from .rasterops import get_monthly_raster_file_names, resample_raster, crop_raster, crop_rasters
from .rasterops import get_raster_extent, map_nodata, create_raster_file_dict
from .rasterops import generate_predictor_raster_values, get_ensemble_avg
from .rasterops import reproject_raster_gdal, reproject_raster_gdal_syscall, correct_cdl_rasters


def get_gee_dict(get_key_list=False):
    """
    Get the available GEE data dictionary.
    :param get_key_list: Set True to get only the key list
    :return: GEE data dictionary if get_key_list if False. Otherwise, only list of keys
    """

    gee_data_dict = {
        'MOD16': 'MODIS/006/MOD16A2',
        'SM_IDAHO': 'IDAHO_EPSCOR/TERRACLIMATE',
        'SMOS_SMAP': ['NASA_USDA/HSL/soil_moisture', 'NASA_USDA/HSL/SMAP10KM_soil_moisture'],
        'DROUGHT': 'GRIDMET/DROUGHT',
        'PRISM': 'OREGONSTATE/PRISM/AN81m',
        'TMIN': 'OREGONSTATE/PRISM/AN81m',
        'TMAX': 'OREGONSTATE/PRISM/AN81m',
        'WS': 'IDAHO_EPSCOR/TERRACLIMATE',
        'RO': 'IDAHO_EPSCOR/TERRACLIMATE',
        'DEF_SMAP': ['NASA_USDA/HSL/soil_moisture', 'NASA_USDA/HSL/SMAP10KM_soil_moisture'],
        'DEF': 'IDAHO_EPSCOR/TERRACLIMATE',
        'VPD': 'IDAHO_EPSCOR/GRIDMET',
        'SPH': 'IDAHO_EPSCOR/GRIDMET',
        'NDWI': 'LANDSAT/LC08/C01/T1_8DAY_NDWI',
        'NDVI': 'LANDSAT/LC08/C01/T1_8DAY_NDVI'
    }
    if not get_key_list:
        return gee_data_dict
    return list(gee_data_dict.keys())


def download_gee_data(year_list, start_month, end_month, outdir, data_extent, data='MOD16', gee_scale=1000):
    """
    Download GEE data. (Note: MOD16 has to be divided by 10 (line 38) as its original scale is 0.1 mm/8 days.)
    :param year_list: List of years in %Y format
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param outdir: Download directory
    :param data_extent: Data extent as a list in [minx, miny, maxx, maxy] format
    :param data: Name of the data set, MOD16 for MOD16 ET, SM_IDAHO for IDAHO_EPSCOR TERRACLIMATE soil moisture,
    SMOS_SMAP for SMOS/SMAP soil moisture, 'DROUGHT' for GRIDMET Palmer Drought Severity Index,
    'PRISM' for PRISM precipitation, 'TMIN' and 'TMAX' for PRISM min and max temperatures,
    'WS' for IDAHO_EPSCOR TERRACLIMATE wind speed, 'SPH' for GRIDMET specific humidity, 'RO' for
    IDAHO_EPSCOR TERRACLIMATE runoff, 'DEF', and 'DEF_SMAP' for IDAHO_EPSCOR TERRACLIMATE and
    SMOS/SMAP + MOD16 derived climate water deficit, respectively, 'VPD' for PRISM average
    water vapour pressure deficit
    :param gee_scale: GEE Data Scale in m
    :return: None
    """

    ee.Initialize()
    gee_data_dict = get_gee_dict()
    if data != 'SMOS_SMAP':
        data_collection = ee.ImageCollection(gee_data_dict[data])
    else:
        data_collection = ee.ImageCollection(gee_data_dict[data][0])
    gee_aoi = ee.Geometry.Rectangle(data_extent)
    for year in year_list:
        start_date = ee.Date.fromYMD(year, start_month, 1)
        sm_start_date_end = ee.Date.fromYMD(year, start_month + 1, 1)
        sm_end_date = ee.Date.fromYMD(year, end_month, 1)
        if end_month < 12:
            sm_end_date_end = ee.Date.fromYMD(year, end_month + 1, 1)
        else:
            sm_end_date_end = ee.Date.fromYMD(year + 1, 1, 1)
        if end_month == 12:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)
        else:
            end_date = ee.Date.fromYMD(year, end_month + 1, 1)
        if end_month <= start_month:
            start_date = ee.Date.fromYMD(year - 1, start_month, 1)
        gee_data = None
        if data == 'MOD16':
            gee_data = data_collection.select('ET').filterDate(start_date, end_date).sum().divide(10).toDouble()
        elif data == 'SM_IDAHO':
            sm_start_data = data_collection.select('soil').filterDate(start_date,
                                                                      sm_start_date_end).first().divide(10).toDouble()
            sm_end_data = data_collection.select('soil').filterDate(sm_end_date,
                                                                    sm_end_date_end).first().divide(10).toDouble()
            gee_data = sm_end_data.subtract(sm_start_data)
        elif data == 'SMOS_SMAP':
            if year >= 2015 and start_month >= 4:
                data_collection = ee.ImageCollection(gee_data_dict[data][1])
            else:
                data_collection = ee.ImageCollection(gee_data_dict[data][0])
            sm_start_data = data_collection.select('ssm').filterDate(start_date, sm_start_date_end).first().toDouble()
            sm_end_data = data_collection.select('ssm').filterDate(sm_end_date, sm_end_date_end).first().toDouble()
            gee_data = sm_end_data.subtract(sm_start_data)
        elif data == 'DROUGHT':
            gee_data = data_collection.select('pdsi').filterDate(start_date, end_date).mean().toDouble()
        elif data == 'PRISM':
            gee_data = data_collection.select('ppt').filterDate(start_date, end_date).sum().toDouble()
        elif data == 'TMIN' or data == 'TMAX':
            gee_data = data_collection.select(data.lower()).filterDate(start_date, end_date).median().toDouble()
        elif data == 'WS':
            gee_data = data_collection.select('vs').filterDate(start_date, end_date).mean().divide(100).toDouble()
        elif data == 'RO':
            gee_data = data_collection.select('ro').filterDate(start_date, end_date).sum().toDouble()
        elif data == 'SPH':
            gee_data = data_collection.select('sph').filterDate(start_date, end_date).mean().toDouble()
        elif data in ['NDWI', 'NDVI']:
            gee_data = data_collection.select(data).filterDate(start_date, end_date).mean().toDouble()
        elif data == 'DEF':
            gee_data = data_collection.select('def').filterDate(start_date, end_date).sum().divide(10).toDouble()
        elif data == 'DEF_SMAP':
            if year >= 2015 and start_month >= 4:
                data_collection_sm = ee.ImageCollection(gee_data_dict[data][1])
            else:
                data_collection_sm = ee.ImageCollection(gee_data_dict[data][0])
            data_collection_mod16 = ee.ImageCollection(gee_data_dict['MOD16'])
            sm_data = data_collection_sm.select('ssm').filterDate(start_date, end_date).sum().toDouble()
            et_data = data_collection_mod16.select('ET').filterDate(start_date, end_date).sum().divide(10).toDouble()
            gee_data = et_data.subtract(sm_data)
        elif data == 'VPD':
            gee_data = data_collection.select('vpd').filterDate(start_date, end_date).sum().toDouble()
        gee_url = gee_data.getDownloadUrl({
            'scale': gee_scale,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        local_file_name = outdir + data + '_' + str(year) + '.zip'
        print('Dowloading', local_file_name, '...')
        r = requests.get(gee_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)


def download_ssebop_data(year_list, start_month, end_month, outdir):
    """
    Download SSEBop Data
    :param year_list: List of years
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param outdir: Download directory
    :return: None
    """

    sse_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/' \
               'downloads/'
    month_flag = False
    month_list = []
    actual_start_year = year_list[0]
    if end_month <= start_month:
        year_list = [actual_start_year - 1] + list(year_list)
        month_flag = True
    else:
        month_list = range(start_month, end_month + 1)
    for year in year_list:
        print('Downloading SSEBop for', year, '...')
        if month_flag:
            month_list = list(range(start_month, 13))
            if actual_start_year <= year < year_list[-1]:
                month_list = list(range(1, end_month + 1)) + month_list
            elif year == year_list[-1]:
                month_list = list(range(1, end_month + 1))
        for month in month_list:
            month_str = str(month)
            if 1 <= month <= 9:
                month_str = '0' + month_str
            url = sse_link + 'm' + str(year) + month_str + '.zip'
            local_file_name = outdir + 'SSEBop_' + str(year) + month_str + '.zip'
            r = requests.get(url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)


def extract_data(zip_dir, out_dir, rename_extracted_files=False):
    """
    Extract data from zip file
    :param zip_dir: Input zip directory
    :param out_dir: Output directory to write extracted files
    :param rename_extracted_files: Set True to rename extracted files according the original zip file name
    :return: None
    """

    print('Extracting zip files...')
    for zip_file in glob(zip_dir + '*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_extracted_files:
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_file[zip_file.rfind(os.sep) + 1: zip_file.rfind('.')] + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)


def get_crop_type_cc_dict():
    """
    Get CDL crop type dictionary and Crop coefficient dictionary value based on crop name
    :return: CDL crop type dictionary and Crop Coefficient Dictionary
    """

    crop_type_dict = {
        'Corn': 1,
        'Cotton': 2,
        'Rice': 3,
        'Soybeans': 5,
        'Fish Culture': 92,
        'Peanuts': np.nan,
        'Other': np.nan
    }
    cc_dict = {
        'Corn': 1.2,
        'Cotton': 1.2,
        'Rice': 1.2,
        'Soybeans': 1.15,
        'Fish Culture': 0,
        'Peanuts': 1.15,
        'Other': 1
    }

    return crop_type_dict, cc_dict


def calculate_relative_et(input_df, crop_col):
    """
    Calculate relative ET based on crop types
    :param input_df: Input dataframe containing
    :param crop_col: Name of the crop column
    :return: Modified data frame with the new 'Relative_ET' column for each ET product
    """

    et_products = ['SSEBop', 'MOD16', 'PT-JPL', 'SIMS', 'OpenET', 'EEMETRIC']
    for et_product in et_products:
        if et_product in input_df.columns:
            new_col = 'Relative_{}'.format(et_product)
            input_df[new_col] = input_df[et_product]
            for crop in input_df[crop_col].unique():
                crop_check = input_df[crop_col] == crop
                mean_et = input_df[crop_check][et_product].to_numpy().mean()
                input_df.loc[crop_check, new_col] /= mean_et
    return input_df


def create_monthly_wu_csv(rt_shp_file, rt_xls_file, output_dir, site_id_shp='Site_numbe', site_id_xls='site_no',
                          year_shp='Year', state_shp='State', acre_shp='combined_a', crop_shp='crop',
                          dt_xls='datetime', year_list=(), state_list=(), load_csv=False):
    """
    Create total monthly water use CSV file from the real-time flowmeter shapefile and XLSX file.
    :param rt_shp_file: Real-time sites shapefile
    :param rt_xls_file: Real-time XLS file containing site ids and daily water use
    :param output_dir: Output directory
    :param site_id_shp: Name of the Site ID field in the shapefile
    :param site_id_xls: Name of the Site ID field in the XLS file
    :param year_shp: Name of the year column in the shapefile
    :param state_shp: Name of the state column in the shapefile
    :param acre_shp: Name of the acre column in the shapefile
    :param crop_shp: Name of the crop column in the shapefile
    :param dt_xls: Name of the date column in the XLS file
    :param year_list: Year list
    :param state_list: State list
    :param load_csv: Load existing CSV
    :return List containing average monthly water use values
    """

    output_dir = make_proper_dir_name(output_dir)
    map_monthly_csv = f'{output_dir}MAP_Monthly.csv'
    if not load_csv:
        rt_gdf = gpd.read_file(rt_shp_file)
        rt_df = pd.read_excel(rt_xls_file)
        rt_df[site_id_xls] = rt_df[site_id_xls].astype(str)
        rt_df[dt_xls] = pd.to_datetime(rt_df[dt_xls])
        rt_gdf[site_id_shp] = rt_gdf[site_id_shp].astype(str)
        rt_gdf = rt_gdf.rename({site_id_shp: site_id_xls}, axis=1)
        makedirs([output_dir])
        if state_list:
            state_list = [s.lower() for s in state_list]
            rt_gdf = rt_gdf[rt_gdf[state_shp].str.tolower().isin(state_list)]
        if year_list:
            rt_gdf = rt_gdf[rt_gdf[year_shp].isin(year_list)]
            rt_df = rt_df[rt_df[dt_xls].dt.year.isin(year_list)]
        rt_df = rt_df[~rt_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        rt_df = rt_df[rt_df[rt_df.columns[2]] >= 0]
        rt_df = rt_df.groupby([site_id_xls, pd.Grouper(key=dt_xls, freq="M")])[rt_df.columns[2]].sum().reset_index()
        rt_df[year_shp] = rt_df[dt_xls].dt.year
        cdl_dict = {
            'fish': 'Fish Culture',
            'duck': 'Fish Culture',
            'flood': 'Fish Culture',
            'wild': 'Fish Culture',
            'rice': 'Rice',
            'corn': 'Corn',
            'cotton': 'Cotton',
            'soybeans': 'Soybeans'
        }
        for crop in cdl_dict.keys():
            crop_rows = rt_gdf[crop_shp].str.contains(crop, case=False)
            rt_gdf.loc[crop_rows, crop_shp] = cdl_dict[crop]
        rt_df_new = rt_df.merge(rt_gdf, on=[site_id_xls, year_shp])
        rt_df_new['Month'] = rt_df_new[dt_xls].dt.month
        rt_df_new = rt_df_new[rt_df_new[acre_shp] > 0]
        rt_df_new = rt_df_new.sort_values(by=[site_id_xls, year_shp, 'Month'])
        rt_df_new['AF_Acre'] = rt_df_new[rt_df_new.columns[2]] / rt_df_new[acre_shp]
        rt_df_new.to_csv(map_monthly_csv, index=False)
    else:
        rt_df_new = pd.read_csv(map_monthly_csv)
    return rt_df_new


def prepare_data(monthly_df, output_dir, sub_cols=None, lat_pump='Latitude', lon_pump='Longitude',
                 year_col='ReportYear', data_list=('MOD16', 'SSEBop'), gee_scale=1000,
                 already_prepared=False, skip_download=False, map_extent_file=None, **kwargs):
    """
    Prepare data from existing CSV file and download additional data
    :param monthly_df: Input MAP monthly data frame
    :param output_dir: Output directory
    :param sub_cols: Columns to select from input_csv. Set None to use all columns
    :param lat_pump: Latitude column in input_csv.
    :param lon_pump: Longitude column in input_csv.
    :param year_col: Name of the Year column
    :param data_list: List of data sets to download, valid names include 'SSEBop', 'SM_IDAHO', 'MOD16', 'SMOS_SMAP',
    'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', 'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin',
    'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS'.
    Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for USDA-NASS cropland data
    :param gee_scale: GEE data scale in metres
    :param already_prepared: Set True if subset csv already exists
    :param skip_download: Set True to load already downloaded files
    :param map_extent_file: Set MAP shapefile or raster path to get extents from this shapefile.
    Otherwise, extent will be calculated  from the latitude and longitude of the input_csv
    :param kwargs: Pass additional data paths such as prism_data_path, cdl_data_path, openet_data_path,
    eemetric_data_path, pt_jpl_data_path, sims_data_path
    :return: Subset of input_csv as a Pandas dataframe, tuple of data directories in the order of SSEBop, CDL, PRISM,
    OpenET, EEMETRIC, PT-JPL, SIMS, GEE Files.
    """

    print('Preparing data sets...')
    output_csv = output_dir + 'Monthly_Predictor.csv'
    data_dir = make_proper_dir_name(output_dir + 'Downloaded_Data')
    makedirs([data_dir])
    if not already_prepared:
        sub_df = monthly_df.copy(deep=True)
        if sub_cols:
            sub_df = sub_df[sub_cols].copy()
        year_list = sorted(sub_df[year_col].unique().tolist())
        sub_df, file_dirs = download_data(
            sub_df, data_dir, year_list,
            lat_pump, lon_pump, year_col,
            data_list, gee_scale, skip_download,
            map_extent_file, **kwargs
        )
        sub_df.to_csv(output_csv, index=False)
    else:
        if sub_cols:
            sub_df = pd.read_csv(output_csv)
        else:
            sub_df = monthly_df
        file_dirs = download_data(sub_df, data_dir, get_dirs_only=True, **kwargs)
    print('CSV file prepared...')
    return sub_df, file_dirs


def import_new_data(original_data_csv, new_data_csv, location_shp_file, output_dir, pumping_year_attr='ReportYear',
                    pumping_id_attr='id', pumping_area_attr='area_ac', pumping_attr='pumping_acft',
                    pumping_crop_attr='Crops', loc_shp_id_attr='PermitNumb',
                    loc_shp_long_lat=('Longitude', 'Latitude'),
                    report_year_list=(2019,), data_list=('ppt', 'tmin', 'tmax'), data_path=None, start_month=4,
                    end_month=9, gee_scale=1000, already_prepared=False, skip_download=False):
    """
    Add new data to existing CSV
    :param original_data_csv: Original CSV
    :param new_data_csv: New CSV file to import data from (should contain USGS pumping readings and permit numbers
    :param pumping_year_attr: Pumping report year in CSV file
    :param pumping_id_attr: Pumping ID column name in CSV file
    :param pumping_area_attr: Pumping area attribute name
    :param pumping_attr: Pumping attribute name
    :param pumping_crop_attr: Pumping crop attribute name
    :param loc_shp_id_attr: Well permit ID in shape file
    :param location_shp_file: Location shapefile containing polygons
    :param output_dir: Output directory
    :param loc_shp_long_lat: Longitude and latitude column names of location shapefile
    :param report_year_list: List of report years to use
    :param data_list: Extra data sets other than AF/Acre to add to original_data_csv.
    Note These data are not readily downloadable.
    :param data_path: Path to data folder. Set None if data_sets is None
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param gee_scale: GEE data scale in metres
    :param already_prepared: Set True if subset csv already exists
    :param skip_download: Set True to load already downloaded files
    :return: Updated Pandas data frame
    """

    if not already_prepared:
        print(original_data_csv)
        original_df = pd.read_csv(original_data_csv)
        new_df = pd.read_csv(new_data_csv)
        loc_shp = gpd.read_file(location_shp_file)
        crop_type_dict, _ = get_crop_type_cc_dict()
        update_df = pd.DataFrame()
        for report_year in report_year_list:
            new_data = new_df.loc[new_df[pumping_year_attr] == report_year]
            crops = [crop_type_dict[crop_n] for crop_n in new_data[pumping_crop_attr]]
            latitude, longitude = [], []
            for pump_id in new_data[pumping_id_attr].values.tolist():
                loc_gdf = loc_shp.loc[loc_shp[loc_shp_id_attr] == pump_id]
                lat_list = loc_gdf[loc_shp_long_lat[1]].tolist()
                long_list = loc_gdf[loc_shp_long_lat[0]].tolist()
                latitude.append(lat_list[0])
                longitude.append(long_list[0])
            af_ac = new_data[pumping_attr].to_numpy().ravel() / new_data[pumping_area_attr].to_numpy().ravel()
            year = new_data[pumping_year_attr].to_list()
            update_dict = {'Year': year, 'Latitude': latitude, 'Longitude': longitude,
                           'Crop_n': crops, 'AF/Acre': af_ac}
            update_df = update_df.append(pd.DataFrame(data=update_dict))
        update_df = reindex_df(update_df, column_names=None)
        update_df.to_csv(output_dir + 'test.csv', index=False)
        data_dir = make_proper_dir_name(output_dir + 'Downloaded_Data')
        makedirs([data_dir])
        update_df, _ = download_data(update_df, data_dir, report_year_list, data_list=data_list,
                                     data_start_month=start_month, data_end_month=end_month, gee_scale=gee_scale,
                                     skip_download=skip_download, prism_data_path=data_path)
        column_dict = {
            'ppt': 'ppt_',
            'tmin': 'tmin_median_',
            'tmax': 'tmax_median_'
        }
        for column in column_dict.keys():
            col_name = column_dict[column] + str(start_month) + 'to' + str(end_month)
            update_df.rename(columns={column: col_name}, inplace=True)
        original_df = pd.concat([original_df, update_df])
        original_df = reindex_df(original_df, column_names=None)
        original_df.to_csv(output_dir + 'Annual_New_Data.csv', index=False)
        return original_df
    else:
        return pd.read_csv(output_dir + 'Annual_New_Data.csv')


def copy_attributes(original_csv, sub_df, original_cols):
    """
    Copy columns from original data frame to subset data frame
    :param original_csv: Original CSV file
    :param sub_df: Subset data frame created using #prepare_data()
    :param original_cols: Column names in the original data frame
    :return: Sub set data frame with additional columns added from original_df
    """

    file_sep = original_csv.rfind(os.sep)
    if file_sep == -1:
        file_sep = original_csv.rfind('/')
    output_dir = original_csv[: file_sep + 1]
    output_csv = output_dir + 'Annual_Subset_Modified.csv'
    original_df = pd.read_csv(original_csv)
    for col in original_cols:
        sub_df[col] = original_df[col]
    if 'Crop_n' in original_cols:
        sub_df['CDL_Corrected'] = sub_df['CDL']
        fix_pos = ~sub_df['CDL_Corrected'].isin([1, 2, 3, 5, 92])
        sub_df.loc[fix_pos, 'CDL_Corrected'] = sub_df['Crop_n'][fix_pos]
    sub_df = reindex_df(sub_df, column_names=None)
    sub_df.to_csv(output_csv, index=False)
    return sub_df


def reindex_df(df, column_names, ordering=False):
    """
    Reindex dataframe columns
    :param df: Input dataframe
    :param column_names: Dataframe column names, these must be df headers
    :param ordering: Set True to apply ordering
    :return: Reindexed dataframe
    """
    if not column_names:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)


def download_data(input_df, data_dir, year_list=None, lat_col='Latitude', lon_col='Longitude',
                  year_col='Year', data_list=('MOD16', 'SSEBop'), gee_scale=1000, skip_download=False,
                  map_extent_file=None, get_dirs_only=False, **kwargs):
    """
    Download GEE and SSEBop data
    :param input_df: Input data frame
    :param data_dir: Data directory
    :param year_list: List of years for which data will be downloaded
    :param lat_col: Name of the latitude column used for downloading data
    :param lon_col: Name of the longitude column used for downloading data
    :param year_col: Name of the Year column
    :param data_list: List of data sets to download, valid names include 'SSEBop', 'SM_IDAHO', 'MOD16', 'SMOS_SMAP',
    'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', 'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin',
    'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', 'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF',
    'SWB_RINF', 'SWB_RO', 'SWB_SS', 'SWB_MRD', 'SWB_SSM'
    Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for USDA-NASS cropland data,
    'SWB*' for SWB products.
    :param gee_scale: GEE data scale in metres
    :param skip_download: Set True to load already downloaded files
    :param map_extent_file: Set MAP shapefile or raster path to get extents from this shapefile.
    Otherwise, extent will be calculated  from the latitude and longitude of the input_csv
    :param get_dirs_only: Set True to get directories for pre-processed files
    :param kwargs: Pass additional data paths such as prism_data_path, cdl_data_path, openet_data_path,
    eemetric_data_path, pt_jpl_data_path, sims_data_path
    :return: Modified input_df with downloaded data, tuple of data directories in the order of SSEBop, CDL, PRISM,
    OpenET, EEMETRIC, PT-JPL, SIMS, GEE Files. If get_dirs_only is True then only the data paths are returned.
    """

    gee_zip_dir = make_proper_dir_name(data_dir + 'GEE_Data')
    ssebop_zip_dir = make_proper_dir_name(data_dir + 'SSEBop_Data')
    gee_file_dir = make_proper_dir_name(gee_zip_dir + 'GEE_Files')
    ssebop_file_dir = make_proper_dir_name(ssebop_zip_dir + 'SSEBop_Files')
    makedirs([gee_zip_dir, ssebop_zip_dir, gee_file_dir, ssebop_file_dir])
    cdl_data_path = kwargs.get('cdl_data_path', '')
    prism_data_path = kwargs.get('prism_data_path', '')
    openet_data_path = kwargs.get('openet_data_path', '')
    eemetric_data_path = kwargs.get('eemetric_data_path', '')
    pt_jpl_data_path = kwargs.get('pt_jpl_data_path', '')
    sims_data_path = kwargs.get('sims_data_path', '')
    swb_data_path = kwargs.get('swb_data_path', '')
    data_paths = (
        ssebop_file_dir, cdl_data_path, prism_data_path, openet_data_path,
        eemetric_data_path, pt_jpl_data_path, sims_data_path, gee_file_dir,
        swb_data_path
    )
    if get_dirs_only:
        return data_paths
    src_crs = 'EPSG:4326'
    local_et_data_path_dict = {
        'OpenET': openet_data_path,
        'EEMETRIC': eemetric_data_path,
        'PT-JPL': pt_jpl_data_path,
        'SIMS': sims_data_path,
    }
    local_et_products = list(local_et_data_path_dict.keys())
    gee_products = get_gee_dict(get_key_list=True)
    for data in data_list:
        if not skip_download:
            if data == 'SSEBop':
                download_ssebop_data(year_list, 1, 12, ssebop_zip_dir)
                extract_data(ssebop_zip_dir, ssebop_file_dir)
            elif data in gee_products:
                if map_extent_file:
                    if '.shp' in map_extent_file:
                        data_extent = gpd.read_file(map_extent_file).to_crs('EPSG:4326').total_bounds.tolist()
                    else:
                        data_extent = get_raster_extent(map_extent_file, src_crs)
                else:
                    data_extent = [
                        np.min(input_df[lon_col]), np.min(input_df[lat_col]),
                        np.max(input_df[lon_col]), np.max(input_df[lat_col])
                    ]
                download_gee_data(
                    year_list, 1, 12,
                    gee_zip_dir, data_extent,
                    data, gee_scale
                )
                extract_data(gee_zip_dir, out_dir=gee_file_dir, rename_extracted_files=True)
    for idx, row in input_df.iterrows():
        year = int(row[year_col])
        month = int(row['Month'])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        for data in data_list:
            raster_dir = ssebop_file_dir
            if data in local_et_products:
                if year >= 2016:
                    raster_dir = local_et_data_path_dict[data]
                else:
                    raster_dir = ssebop_file_dir
            elif data in ['ppt', 'tmax', 'tmin', 'tmean']:
                raster_dir = prism_data_path
            elif data in gee_products:
                raster_dir = gee_file_dir
            elif data == 'CDL':
                raster_dir = cdl_data_path
            elif data.startswith('SWB'):
                raster_dir = swb_data_path
            raster_file_name = get_monthly_raster_file_names(raster_dir, year, month, data)
            raster_arr, raster_file = read_raster_as_arr(raster_file_name, change_dtype=data.startswith('SWB'))
            if raster_file_name.endswith('.asc'):
                raster_crs = 'EPSG:5070'
            else:
                raster_crs = raster_file.crs.to_string()
            lon_reproj, lat_reproj = lon, lat
            if data == 'CDL' or data.startswith('SWB'):
                new_coords = reproject_coords(src_crs, raster_crs, [[lon, lat]])
                lon_reproj, lat_reproj = new_coords[0]
            py, px = raster_file.index(lon_reproj, lat_reproj)
            raster_val = np.nan
            try:
                raster_val = raster_arr[py, px]
                if np.isnan(raster_val):
                    categorical = data == 'SWB_HSG'
                    raster_val = get_ensemble_avg(raster_arr, (py, px), categorical=categorical)
            except IndexError:
                print('Index error occured for', raster_file_name)
            input_df.loc[idx, data] = raster_val
    input_df = reindex_df(input_df, column_names=None)
    return input_df, data_paths


def create_map_extent_rasters(input_extent_file, output_dir, year_list, cdl_data_dir, nhd_shp_file, lanid_dir,
                              field_shp_dir, load_files=False):
    """
    Create MAP extent rasters based on AIWUM 1.1 and CDL rasters
    :param input_extent_file: Input MAP extent raster file path
    :param output_dir: Output directory
    :param year_list: List of years to be predicted
    :param cdl_data_dir: CDL data directory
    :param nhd_shp_file: MAP NHD shapefile
    :param lanid_dir: LANID directory
    :param field_shp_dir: Field shapefile directory for permitted boundaries
    :param load_files: Set True to load existing files
    :return: MAP extent raster directory and MAP extent raster dictionary where values contain CDL array and
    rasterio object, as tuples
    """

    map_extent_raster_dir = make_proper_dir_name(output_dir + 'MAP_Extent_Rasters')
    if not load_files:
        makedirs([map_extent_raster_dir])
        cdl_output_dir = make_proper_dir_name(output_dir + 'CDL_100m')
        makedirs([cdl_output_dir])
        print('Creating yearly corrected CDL 100 m rasters using NHD and LANID...')
        cdl_data_dict = correct_cdl_rasters(
            cdl_data_dir, nhd_shp_file,
            lanid_dir, field_shp_dir,
            cdl_output_dir, year_list
        )
        for year in year_list:
            cdl_file = cdl_data_dict[year]
            map_extent_raster = map_extent_raster_dir + f'MAP_Extent_{year}.tif'
            print('Creating MAP Extent raster from CDL', year, '...')
            crop_raster(cdl_file, input_extent_file, output_raster_file=map_extent_raster)
    map_extent_raster_dict = {}
    for year in year_list:
        map_extent_raster = map_extent_raster_dir + f'MAP_Extent_{year}.tif'
        map_extent_raster_arr, map_extent_raster_file = read_raster_as_arr(map_extent_raster, change_dtype=False)
        map_extent_raster_dict[year] = (map_extent_raster_arr, map_extent_raster_file)
    return map_extent_raster_dir, map_extent_raster_dict


def create_map_prediction_rasters(ml_model, pred_csv_dir, output_dir, x_scaler, y_scaler, map_extent_raster_dict,
                                  crop_col, load_files=False, gdal_path=None):
    """
    Create prediction raster (in acre-feet) from prediction CSVs
    :param ml_model: Pre-fitted ML model object
    :param pred_csv_dir: Directory containing the prediction CSVs
    :param output_dir: Output directory
    :param x_scaler: X scaler object
    :param y_scaler: Y scaler object
    :param map_extent_raster_dict: MAP extent raster dictionary where values contain CDL array and rasterio object
    :param crop_col: Name of the crop column in the original VMP data
    :param load_files: Set True to load existing model for prediction
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory. Required in case the correct GDAL
    backend is not present with the 'sum' resampling function
    :return: Predicted raster directory
    """

    pred_raster_dir = make_proper_dir_name(output_dir + 'Pred_Raster')
    if not load_files:
        pred_df_list = glob(pred_csv_dir + '*.csv')
        makedirs([pred_raster_dir])
        other_crop_col = crop_col + '_Other'
        for pred_df_file in pred_df_list:
            print('Predicting', pred_df_file)
            pred_df = pd.read_csv(pred_df_file)
            pred_df_other_col_idx = pred_df.index[pred_df[other_crop_col] == 1]
            pred_df = pred_df.drop(columns=[other_crop_col])
            pred_arr = pred_df.to_numpy().copy()
            pred_arr_copy = pred_arr.copy()
            nan_pos = np.isnan(pred_arr)
            inf_pos = np.isinf(pred_arr)
            if nan_pos.size > 0:
                pred_arr[nan_pos] = 0
            if inf_pos.size > 0:
                pred_arr[inf_pos] = 0
            pred_df = pd.DataFrame(pred_arr, columns=pred_df.columns)
            pred_df_copy = pred_df.copy(deep=True)
            if x_scaler:
                pred_df_copy = pd.DataFrame(x_scaler.transform(pred_df), columns=pred_df_copy.columns)
            pred_df_corn = pred_df_copy.copy(deep=True)
            pred_df_cotton = pred_df_copy.copy(deep=True)
            pred_df_soybeans = pred_df_copy.copy(deep=True)
            pred_df_corn.loc[pred_df_other_col_idx, crop_col + '_Corn'] = 1
            pred_df_cotton.loc[pred_df_other_col_idx, crop_col + '_Cotton'] = 1
            pred_df_soybeans.loc[pred_df_other_col_idx, crop_col + '_Soybeans'] = 1
            pred_wu = np.abs(ml_model.predict(pred_df_copy))
            pred_wu_corn = np.abs(ml_model.predict(pred_df_corn.loc[pred_df_other_col_idx]))
            pred_wu_cotton = np.abs(ml_model.predict(pred_df_cotton.loc[pred_df_other_col_idx]))
            pred_wu_soybeans = np.abs(ml_model.predict(pred_df_soybeans.loc[pred_df_other_col_idx]))
            pred_other_avg = np.mean(np.array([pred_wu_corn, pred_wu_cotton, pred_wu_soybeans]), axis=0)
            if y_scaler:
                pred_wu = y_scaler.inverse_transform(pred_wu.reshape(-1, 1))
                pred_other_avg = y_scaler.inverse_transform(pred_other_avg.reshape(-1, 1))
            pred_df['Pred_WU'] = pred_wu
            pred_df.loc[pred_df_other_col_idx, 'Pred_WU'] = pred_other_avg
            num_features = pred_arr.shape[1]
            year = pred_df_file[pred_df_file.rfind('_') + 1: pred_df_file.rfind('.')]
            map_extent_raster_arr, map_extent_raster_file = map_extent_raster_dict[int(year)]
            for feature in range(num_features):
                nan_pos = np.isnan(pred_arr_copy[:, feature])
                inf_pos = np.isinf(pred_arr_copy[:, feature])
                if nan_pos.size > 0:
                    pred_wu[nan_pos] = map_extent_raster_file.nodata
                if inf_pos.size > 0:
                    pred_wu[inf_pos] = map_extent_raster_file.nodata
            pred_wu = pred_wu.reshape(map_extent_raster_arr.shape) * 2.471  # 6.4 if in (ft/acre/0.1 mi * 0.1 mi)
            pred_wu_suffix = pred_raster_dir + 'Pred_WU_{}'.format(year)
            pred_wu_csv = pred_wu_suffix + '.csv'
            pred_df.to_csv(pred_wu_csv, index=False)
            pred_wu_raster = pred_wu_suffix + '.tif'
            write_raster(pred_wu, map_extent_raster_file, transform_=map_extent_raster_file.transform,
                         outfile_path=pred_wu_raster, no_data_value=map_extent_raster_file.nodata)
            pred_wu_resampled_raster = pred_raster_dir + 'Pred_WU_Resampled_{}.tif'.format(year)
            try:
                reproject_raster_gdal(
                    pred_wu_raster,
                    pred_wu_resampled_raster,
                    resampling_factor=10,
                    downsampling=True,
                    resampling_func='sum'
                )
            except AttributeError:
                print('Using GDAL system call because of gdal.GRA_Sum attribute error...')
                reproject_raster_gdal_syscall(
                    pred_wu_raster,
                    pred_wu_resampled_raster,
                    resampling_factor=10,
                    downsampling=True,
                    resampling_func='sum',
                    verbose=False, gdal_path=gdal_path
                )

    return pred_raster_dir


def create_map_prediction_csv(input_extent_file, file_dirs, gee_files, prism_files, swb_files, data_list,
                              month, year_list, map_extent_raster_dict, output_dir, crop_col,
                              src_crs='EPSG:4326', load_files=False, verbose=False):
    """
    Create prediction CSVs from predictor data sets
    :param input_extent_file: Input MAP extent raster file path
    :param file_dirs: Input file directories in the order of SSEBop, CDL, PRISM, GEE Files
    :param gee_files: List of GEE files
    :param prism_files: List of PRISM files
    :param swb_files: List of SWB files
    :param data_list: List of data sets
    :param month: Month in %m format
    :param year_list: List of years to be predicted
    :param map_extent_raster_dict: MAP extent raster dictionary where values contain CDL array and rasterio object
    :param output_dir: Output directory
    :param crop_col: Name of the crop column in the original VMP data
    :param src_crs: Source CRS for creating lat/long values
    :param load_files: Set True to load existing files
    :param verbose: Set True to get additional details about intermediate steps
    :return: Directory containing yearly Prediction CSVs
    """

    pred_csv_dir = make_proper_dir_name(output_dir + 'Pred_CSV')
    cdl_dict = defaultdict(
        lambda: 'Other', {
        0: 'NaN',
        1: 'Corn',
        2: 'Cotton',
        3: 'Rice',
        5: 'Soybeans',
        92: 'Fish Culture',
    })
    if not load_files:
        makedirs([pred_csv_dir])
        year = year_list[0]
        cdl_file = map_extent_raster_dict[year][1]
        grid_dir = make_proper_dir_name(output_dir + 'Grids')
        makedirs([grid_dir])
        if verbose:
            print('Creating lat/long grids...')
        long_vals, lat_vals = create_long_lat_grid(cdl_file, grid_dir, target_crs=src_crs, is_rio_object=True)
        for year in year_list:
            pred_df_file = pred_csv_dir + 'Pred_' + str(year) + '.csv'
            print('Creating', pred_df_file, '...')
            pred_df = pd.DataFrame()
            cdl_arr, cdl_file = map_extent_raster_dict[year]
            if verbose:
                print('Retrieving CDL lat/long', year, 'values...')
            pred_df[crop_col] = cdl_arr.ravel()
            pred_df['Latitude'] = lat_vals
            pred_df['Longitude'] = long_vals
            raster_file_dict = create_raster_file_dict(file_dirs, gee_files, prism_files, swb_files, month, year)
            for data in data_list:
                raster_file_list = raster_file_dict[data]
                if not isinstance(raster_file_list, list):
                    raster_file_list = [raster_file_list]
                raster_vals = generate_predictor_raster_values(raster_file_list, cdl_file, output_dir, year, data,
                                                               input_extent_file, verbose)
                pred_df[data] = raster_vals
                if verbose:
                    print('\nCurrent DF', pred_df)
            pred_df = convert_hsg_to_inf(pred_df)
            pred_df[crop_col] = pred_df[crop_col].apply(
                lambda x: cdl_dict[int(x)]
            ).astype(str)
            pred_df = pd.get_dummies(pred_df, columns=[crop_col])
            nan_crop_column = crop_col + '_NaN'
            pred_df.loc[pred_df[nan_crop_column] == 1, data_list[0]] = np.nan
            pred_df = pred_df.drop(columns=[nan_crop_column])
            pred_df = reindex_df(pred_df, column_names=None)
            pred_df.to_csv(pred_df_file, index=False)
    return pred_csv_dir


def create_prediction_map(ml_model, input_extent_file, file_dirs, output_dir, year_list, data_list,
                          input_cdl_dir, nhd_shp_file, lanid_dir, field_shp_dir, data_start_month, data_end_month,
                          crop_col, load_pred_csv=False, load_map_extent=False, load_pred_raster=False, verbose=False,
                          gee_files=('SM_IDAHO', 'MOD16', 'SMOS_SMAP', 'RO', 'DEF'),
                          prism_files=('ppt', 'tmax', 'tmin'), swb_files=('SWB_HSG', 'SWB_IRR'), x_scaler=None,
                          y_scaler=None, gdal_path=None, src_crs='EPSG:4326'):
    """
    Create MAP prediction raster
    :param ml_model: Pre-fitted ML model object
    :param input_extent_file: Input MAP extent raster file path
    :param file_dirs: Input file directories in the order of SSEBop, CDL, PRISM, GEE Files
    :param output_dir: Output directory
    :param year_list: List of years to be predicted
    :param data_list: List of data sets
    :param input_cdl_dir: CDL directory containing CDL rasters at 30 m resolution
    :param nhd_shp_file: MAP NHD shapefile
    :param lanid_dir: LANID directory
    :param field_shp_dir: Field shapefile directory for permitted boundaries
    :param data_start_month: Data start month in %m format
    :param data_end_month: Data end month in %m format
    :param crop_col: Name of the crop column in the original VMP data
    :param load_map_extent: Set True to load existing MAP extent rasters
    :param load_pred_csv: Set True to load existing Prediction CSV files
    :param load_pred_raster: Set True to load existing Prediction rasters
    :param verbose: Set True to get additional details about intermediate steps
    :param gee_files: List of GEE files
    :param prism_files: List of PRISM files
    :param swb_files: List of SWB files
    :param x_scaler: X scaler object
    :param y_scaler: Y scaler object
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param src_crs: Source CRS for creating lat/long values
    :return: Predicted DF directory, raster directory, and MAP extent raster as tuples
    """

    output_dir = make_proper_dir_name(output_dir)
    map_extent_raster_dir, map_extent_raster_dict = create_map_extent_rasters(
        input_extent_file, output_dir, year_list,
        input_cdl_dir, nhd_shp_file, lanid_dir,
        field_shp_dir, load_map_extent
    )
    pred_csv_dir = create_map_prediction_csv(
        input_extent_file, file_dirs, gee_files,
        prism_files, swb_files, data_list,
        data_start_month, data_end_month, year_list,
        map_extent_raster_dict, output_dir, crop_col,
        src_crs, load_pred_csv, verbose
    )
    pred_raster_dir = create_map_prediction_rasters(
        ml_model, pred_csv_dir, output_dir,
        x_scaler, y_scaler, map_extent_raster_dict,
        crop_col, load_pred_raster, gdal_path
    )
    return pred_csv_dir, pred_raster_dir, map_extent_raster_dir


def compare_aiwums_map(aiwum1_tot_dir, aiwum2_dir, input_extent_file, output_dir):
    """
    Compare AIWUM 1.1 and 2.0, create AIWUM 2 maps with volumetric units (m3), and
    generate bar plot for entire MAP region
    :param aiwum1_tot_dir: AIWUM1.1 total crop water use raster directory. Note: AIWUM1.1 predictions are in m^3
    :param aiwum2_dir: Directory containing AIWUM 2.0 predicted rasters. Note: AIWUM2.0 predictions are in acremm
    :param input_extent_file: Input MAP extent raster file path from AIWUM 1.1 or a shapefile
    :param output_dir: Output directory to store file
    :return: None
    """

    aiwum_compare_dir = make_proper_dir_name(output_dir + 'AIWUM_Comparison')
    aiwum1_crop_dir = make_proper_dir_name(aiwum1_tot_dir + 'Cropped')
    makedirs([aiwum_compare_dir, aiwum1_crop_dir])
    if input_extent_file.endswith('.shp'):
        crop_rasters(aiwum1_tot_dir, input_extent_file, aiwum1_crop_dir, prefix='AIWUM1')
    else:
        aiwum1_crop_dir = aiwum_compare_dir
        copy_files(aiwum1_tot_dir, aiwum1_crop_dir, prefix='AIWUM1', pattern='y*.tif')
    aiwum1_rasters = sorted(glob(aiwum1_crop_dir + 'AIWUM1*.tif'))
    aiwum2_rasters = sorted(glob(aiwum2_dir + '*Resampled*.tif'))
    no_data = map_nodata()
    aiwum_tot_pred_df = pd.DataFrame()
    aiwum1_arr_list = []
    aiwum2_arr_list = []
    diff_arr_list = []
    aiwum2_ref_file = None
    for idx, aiwum2_raster in enumerate(aiwum2_rasters):
        aiwum2_arr, aiwum2_ref_file = read_raster_as_arr(aiwum2_raster)
        aiwum2_arr *= 4.047  # acremm to mm^3 to m^3
        year = aiwum2_raster[aiwum2_raster.rfind('_') + 1: aiwum2_raster.rfind('.')]
        if idx < len(aiwum1_rasters):
            aiwum2_arr_list.append(aiwum2_arr)
            aiwum1_raster = aiwum1_rasters[idx]
            aiwum1_arr = resample_raster(
                aiwum1_raster, ref_raster=aiwum2_ref_file,
                is_ref_rio=True, resampling_func='near'
            )
            aiwum1_arr[np.isnan(aiwum2_arr)] = np.nan
            aiwum1_arr_list.append(aiwum1_arr)
            df = {'Year': [int(year)], 'AIWUM1.1': [np.nansum(aiwum1_arr)], 'AIWUM2.0': [np.nansum(aiwum2_arr)]}
            aiwum_tot_pred_df = pd.concat([aiwum_tot_pred_df, pd.DataFrame(data=df)])
            aiwum1_arr_copy = deepcopy(aiwum1_arr)
            aiwum1_arr_copy[np.isnan(aiwum1_arr_copy)] = no_data
            aiwum1_out = f'{aiwum_compare_dir}AIWUM1_Final_{year}.tif'
            write_raster(
                aiwum1_arr_copy,
                aiwum2_ref_file,
                transform_=aiwum2_ref_file.transform,
                outfile_path=aiwum1_out,
                no_data_value=no_data
            )
            diff_arr = aiwum2_arr - aiwum1_arr
            diff_arr_list.append(diff_arr)
            diff_out = f'{aiwum_compare_dir}Diff_{year}.tif'
            diff_arr[np.isnan(diff_arr)] = no_data
            write_raster(
                diff_arr,
                aiwum2_ref_file,
                transform_=aiwum2_ref_file.transform,
                outfile_path=diff_out,
                no_data_value=no_data
            )
        aiwum2_out = f'{aiwum_compare_dir}AIWUM2_Final_{year}.tif'
        aiwum2_arr_copy = deepcopy(aiwum2_arr)
        aiwum2_arr_copy[np.isnan(aiwum2_arr_copy)] = no_data
        write_raster(
            aiwum2_arr_copy,
            aiwum2_ref_file,
            transform_=aiwum2_ref_file.transform,
            outfile_path=aiwum2_out,
            no_data_value=no_data
        )
    aiwum_tot_pred_df.to_csv(aiwum_compare_dir + 'Annual_Tot_AIWUM.csv', index=False)
    aiwum_tot_pred_df.set_index('Year').plot.bar(rot=0)
    plt.ylabel(r'Total Water Use ($m^3$)')
    fig_name = aiwum_compare_dir + 'AIWUM_Total_Comparison.png'
    plt.savefig(fig_name, dpi=600)
    mean_data_list = [aiwum1_arr_list, aiwum2_arr_list, diff_arr_list]
    output_mean_files = ['AIWUM1_Mean.tif', 'AIWUM2_Mean.tif', 'Diff_Mean.tif']
    for data_list, output_file in zip(mean_data_list, output_mean_files):
        mean_data = np.stack(data_list).mean(axis=0)
        mean_data[np.isnan(mean_data)] = no_data
        write_raster(
            mean_data,
            aiwum2_ref_file,
            transform_=aiwum2_ref_file.transform,
            outfile_path=aiwum_compare_dir + output_file,
            no_data_value=no_data
        )


def compare_aiwums_md(aiwum1_tot_dir, aiwum2_pred_md_df, output_dir, src_crs='EPSG:4326', load_files=False,
                      verbose=False):
    """
    Compare AIWUM 1.1 and 2.0 for the Mississippi Delta only. Note: AIWUM1.1 predictions are in m^3/mile^2
    :param aiwum1_tot_dir: AIWUM1.1 total crop water use raster directory
    :param aiwum2_pred_md_df: AIWUM2 pandas data frame
    :param output_dir: Output directory to store file
    :param src_crs: Source CRS of lat/long values in aiwum2_pred_md_df
    :param load_files: Set True to load existing files
    :param verbose: Set True to get extra details
    :return: None
    """

    aiwum_compare_csv = output_dir + 'AIWUM_MD_Comparison.csv'
    if not load_files:
        year_list = sorted(set(aiwum2_pred_md_df.Year))
        aiwum_compare_df = pd.DataFrame()
        for year in year_list:
            aiwum1_raster = glob(aiwum1_tot_dir + '*' + str(year) + '*.tif')[0]
            aiwum1_arr, aiwum1_rf = read_raster_as_arr(aiwum1_raster, change_dtype=False)
            aiwum2_sub_df = aiwum2_pred_md_df.loc[(aiwum2_pred_md_df.Year == year) & (aiwum2_pred_md_df.DATA == 'TEST')]
            lat_vals, long_vals = aiwum2_sub_df['Latitude'].tolist(), aiwum2_sub_df['Longitude'].tolist()
            aiwum1_wu = 0
            if verbose:
                print('Extracting pixel values from', aiwum1_raster)
            raster_crs = aiwum1_rf.crs.to_string()
            for lat, long in zip(lat_vals, long_vals):
                if src_crs != raster_crs:
                    new_coords = reproject_coords(src_crs, raster_crs, [[long, lat]])
                    long, lat = new_coords[0]
                py, px = aiwum1_rf.index(long, lat)
                aiwum1_wu += aiwum1_arr[py, px] * 0.000810714 / 640
            aiwum2_wu = np.sum(aiwum2_sub_df['Pred_AF'].to_numpy())
            actual_wu = np.sum(aiwum2_sub_df['Actual_AF'].to_numpy())
            compare_data_dict = {
                'Year': [year],
                'AIWUM1_WU': [aiwum1_wu],
                'AIWUM2_WU': [aiwum2_wu],
                'Actual_WU': [actual_wu]
            }
            aiwum_compare_df = pd.concat([aiwum_compare_df, pd.DataFrame(data=compare_data_dict)])
        aiwum_compare_df.to_csv(aiwum_compare_csv, index=False)
    else:
        aiwum_compare_df = pd.read_csv(aiwum_compare_csv)
    aiwum1_wu = aiwum_compare_df[['AIWUM1_WU']]
    aiwum2_wu = aiwum_compare_df[['AIWUM2_WU']]
    actual_wu = aiwum_compare_df[['Actual_WU']]
    print('\nAIWUM1 stats...')
    r2, mae, rmse = get_prediction_stats(actual_wu, aiwum1_wu)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('\nAIWUM2 stats...')
    r2, mae, rmse = get_prediction_stats(actual_wu, aiwum2_wu)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    aiwum_compare_df.set_index('Year').plot.bar(rot=0)
    plt.ylabel('Total Water Use (AF/Acre)')
    fig_name = output_dir + 'AIWUM_MD_Comparison' + '.png'
    plt.savefig(fig_name, dpi=600)


def split_data_train_test(input_df, pred_attr='AF_Acre', shuffle=True, random_state=0, test_size=0.2,
                          test_year=True, year_col='ReportYear', crop_col=None, split_strategy=2):
    """
    Split data yearly, randomly, or based on train-test percentage based on year or crop. For the last option,
    by default test_size amount of data is kept from each year for testing.
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param test_year: If True, build test data from the year_col. Otherwise, use crop_col
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column. By default it's None
    :param split_strategy: If 1, Split train test data based on year_col. If 2, then test_size amount of data from
    year_col or crop_col are kept for testing and rest for training;
    for this option, test-year should have some value other than None, else splitting is based on crop_col.
    For any other value of split-strategy, the data are randomly split.
    :return: X_train, X_test, y_train, y_test data frames
    """

    if split_strategy == 1:
        x_train, x_test, y_train, y_test = split_data_yearly(
            input_df, pred_attr=pred_attr, year_col=year_col,
            test_years=test_year, shuffle=shuffle,
            random_state=random_state
        )
    elif split_strategy == 2:
        x_train, x_test, y_train, y_test = split_data_train_test_ratio(
            input_df, pred_attr=pred_attr,
            test_size=test_size, random_state=random_state,
            shuffle=shuffle, test_year=test_year,
            crop_col=crop_col, year_col=year_col
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            input_df, input_df[pred_attr].to_frame(),
            shuffle=shuffle, random_state=random_state,
            test_size=test_size
        )
    return x_train, x_test, y_train, y_test


def split_data_train_test_ratio(input_df, pred_attr='AF_Acre', shuffle=True, random_state=0, test_size=0.2,
                                test_year=True, year_col='ReportYear', crop_col=None):
    """
    Split data based on train-test percentage based on year or crop. By default test_size amount of data is kept from
    each year for testing.
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param test_year: If True, build test data from the year_col. Otherwise, use crop_col
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column. By default it's None
    :return: X_train, X_test, y_train, y_test data frames
    """

    selection_var = input_df[year_col].unique()
    selection_label = year_col
    if not test_year:
        selection_var = input_df[crop_col].unique()
        selection_label = crop_col
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    for svar in selection_var:
        selected_data = input_df.loc[input_df[selection_label] == svar]
        y = selected_data[pred_attr].to_frame()
        x_train, x_test, y_train, y_test = train_test_split(selected_data, y, shuffle=shuffle,
                                                            random_state=random_state, test_size=test_size)
        x_train_df = pd.concat([x_train_df, x_train])
        x_test_df = pd.concat([x_test_df, x_test])
        y_train_df = pd.concat([y_train_df, y_train])
        y_test_df = pd.concat([y_test_df, y_test])
    return x_train_df, x_test_df, y_train_df, y_test_df


def split_data_yearly(input_df, pred_attr='AF_Acre', test_years=(2016,), year_col='ReportYear', shuffle=True,
                      random_state=0):
    """
    Split data based on a particular year
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param test_years: Build test data from only these years
    :param year_col: Name of the year column
    :param shuffle: Set False to stop data shuffling
    :param random_state: Seed for PRNG
    :return: X_train, X_test, y_train, y_test data frames
    """

    years = input_df[year_col].unique()
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    for year in years:
        selected_data = input_df.loc[input_df[year_col] == year]
        x_t = selected_data
        if year not in test_years:
            x_train_df = pd.concat([x_train_df, x_t])
        else:
            x_test_df = pd.concat([x_test_df, x_t])
    y_train_df = x_train_df[pred_attr].to_frame()
    y_test_df = x_test_df[pred_attr].to_frame()
    if shuffle:
        x_train_df = sk.shuffle(x_train_df, random_state=random_state)
        y_train_df = sk.shuffle(y_train_df, random_state=random_state)
        x_test_df = sk.shuffle(x_test_df, random_state=random_state)
        y_test_df = sk.shuffle(y_test_df, random_state=random_state)
    return x_train_df, x_test_df, y_train_df, y_test_df


def process_outliers(input_df, target_attr, crop_col, year_col, operation=1):
    """
    Remove outliers from a dataframe based on target_attr
    :param input_df: Input data frame
    :param target_attr: Target attribute based on which outlier removal will occur
    :param crop_col: Name of the crop column
    :param year_col: Name of the year column
    :param operation: Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
    by each crop, 3 for removing outliers by each year, or 4 for removing as per AIWUM 1 based on irrigation thresholds
    Note: for this project we only process outliers above the boxplot upper limit for 1-3
    :return: Modified data frame
    """

    input_df = input_df.copy(deep=True)
    init_rows = input_df.shape[0]
    num_outliers = 0
    max_irrigation_dict = {
        'Rice': 8,
        'Fish Culture': 10,
        'Soybeans': 5,
        'Corn': 5,
        'Cotton': 5
    }
    if operation == 1:
        target_vals = input_df[target_attr].to_numpy().ravel()
        q3, q1 = np.percentile(target_vals, [75, 25])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        invalid_idx = input_df[target_attr] > upper_limit
        num_outliers = invalid_idx.sum()
        input_df.loc[invalid_idx, target_attr] = np.nan
    elif operation >= 2:
        selection_vals = input_df[crop_col].unique()
        selection_col = crop_col
        if operation == 3:
            selection_vals = input_df[year_col].unique()
            selection_col = year_col
        for val in selection_vals:
            selection = input_df[selection_col] == val
            selected_data = input_df[selection]
            if operation != 4:
                target_vals = selected_data[target_attr].to_numpy().ravel()
                q3, q1 = np.percentile(target_vals, [75, 25])
                iqr = q3 - q1
                upper_limit = q3 + 1.5 * iqr
            else:
                upper_limit = max_irrigation_dict[val]
            invalid_idx = selected_data[target_attr] > upper_limit
            outliers = invalid_idx.sum()
            print('{} {} outliers: {}'.format(selection_col, val, outliers))
            num_outliers += outliers
            input_df.loc[selection, 'Outlier'] = invalid_idx
        input_df = input_df[input_df['Outlier'] == False]
        input_df = input_df.drop(columns='Outlier')
    input_df = input_df.dropna()
    print('Old DF rows = {}, New DF rows = {}'.format(init_rows, input_df.shape[0]))
    print("{} outliers removed...".format(num_outliers))
    return input_df


def convert_to_mm(input_df, pred_attr='AF_Acre'):
    """
    Convert feature units to mm for SWB data and Groundwater Pumpings
    :param input_df: Cleaned input data frame
    :param pred_attr: Name of the target attribute (originally in ft)
    """

    input_df[pred_attr] *= 304.8
    for column in input_df.columns:
        if column != 'SWB_HSG' and (column == 'EFF_PPT' or column.startswith('SWB')):
            input_df[column] *= 25.4
    return input_df


def convert_hsg_to_inf(input_df, drop_hsg=True, hsg_col='SWB_HSG'):
    """
    Reclassify HSG into 4 groups and convert SWB HSG to Infiltration Rates
    :param input_df: Cleaned input data frame containing hsg_col column
    :param drop_hsg: Set False to disable dropping HSG column. If False HSG_INF column won't be created
    :param hsg_col: Name of the HSG column
    """

    if hsg_col in input_df.columns:
        input_df[hsg_col] = input_df[hsg_col].apply(lambda x: x if x <= 4 else x - 4)
        if drop_hsg:
            inf_dict = {
                1: 7.62,
                2: 5.715,
                3: 2.54,
                4: 0.635
            }
            input_df['HSG_INF'] = input_df[hsg_col].apply(lambda x: np.nan if np.isnan(x) else inf_dict[x])
            input_df = input_df.drop(columns=[hsg_col])
    return input_df


def create_train_test_data(input_df, output_dir, pred_attr='AF_Acre', drop_attr=('ReportYear',), test_size=0.2,
                           test_year=(), year_col='ReportYear', random_state=42, already_created=False,
                           scaling=False, year_list=(), crop_col='Crop(s)', split_strategy=3, outlier_op=1,
                           shuffle=True, crop_models=False, hsg_to_inf=True):
    """
    Create train and test data
    :param input_df: Input dataframe
    :param output_dir: Output directory
    :param pred_attr: Attribute to be predicted
    :param drop_attr: List of attributes to drop from model training
    :param test_size: Test size between (0, 1)
    :param test_year: Build test data from only this year. Use tuple of years to split train and test data using
    #split_data_attribute
    :param year_col: Name of the year column
    :param random_state: PRNG seed
    :param already_created: Set True to load existing train and test data
    :param scaling: Set True to perform minmax scaling
    :param year_list: List of years to build the data set
    :param crop_col: Name of the crop column to create dummy variables. Set None to disable dummy creation
    :param split_strategy: If 1, Split train test data based on year_col. If 2, then test_size amount of data from
    year_col or crop_col are kept for testing and rest for training;
    for this option, test-year should have some value other than None, else splitting is based on crop_col.
    For any other value of split-strategy, the data are randomly split.
    :param outlier_op: Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
    by each crop, 3 for removing outliers by each year, or 4 for removing as per AIWUM 1 based on irrigation thresholds.
    Note: for this project we only process outliers above the boxplot upper limit for 1-3.
    Set None to disable outlier processing.
    :param shuffle: Set False to stop data shuffling
    :param crop_models: Set True for individual crop models for each crop type. If True, dummies are not created
    :param hsg_to_inf: Set False to disable creating the infiltration rate column based on the HSGs. Only works when
    SWB_HSG is present in input_df and not in drop_attr
    :return: X_train, X_test as pandas data frames, y_train, y_test as numpy arrays. If scaling=True, then
    x_scaler and y_scaler are also returned. Year_train, Year_test, Crop_train, and Crop_test are returned as well for
    AIWUM analysis later on
    """

    makedirs([make_proper_dir_name(output_dir)])
    x_train_file = output_dir + 'X_train.csv'
    x_test_file = output_dir + 'X_test.csv'
    y_train_file = output_dir + 'y_train.csv'
    y_test_file = output_dir + 'y_test.csv'
    year_train_file = output_dir + 'Year_train.csv'
    year_test_file = output_dir + 'Year_test.csv'
    crop_train_file = output_dir + 'Crop_train.csv'
    crop_test_file = output_dir + 'Crop_test.csv'
    x_scaler_file, x_scaler, y_scaler_file, y_scaler = [None] * 4
    if scaling:
        x_scaler_file = output_dir + 'x_scaler'
        y_scaler_file = output_dir + 'y_scaler'
    if not already_created:
        drop_attr = [attr for attr in drop_attr]
        crop_flag = False
        if crop_col in drop_attr:
            drop_attr.remove(crop_col)
            crop_flag = True
        if year_col in drop_attr:
            drop_attr.remove(year_col)
        input_df = input_df.drop(columns=drop_attr)
        input_df = input_df[input_df[crop_col].isin([
            'Rice', 'Corn', 'Cotton', 'Fish Culture', 'Soybeans'
        ])]
        input_df = input_df[~input_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        if year_list and year_col in input_df.columns:
            input_df = input_df[input_df[year_col].isin(year_list)]
        if outlier_op is not None:
            input_df = process_outliers(input_df, pred_attr, crop_col, year_col, outlier_op)
        input_df = convert_to_mm(input_df, pred_attr)
        input_df = convert_hsg_to_inf(input_df, drop_hsg=hsg_to_inf)
        input_df.to_csv(output_dir + 'Cleaned_MAP_GW_Data.csv', index=False)
        x_train, x_test, y_train, y_test = split_data_train_test(
            input_df, pred_attr=pred_attr,
            test_size=test_size,
            random_state=random_state, shuffle=shuffle,
            test_year=test_year, crop_col=crop_col,
            split_strategy=split_strategy, year_col=year_col
        )
        year_train = x_train[year_col].copy().to_frame()
        year_test = x_test[year_col].copy().to_frame()
        x_train = x_train.drop(columns=[year_col, pred_attr])
        x_test = x_test.drop(columns=[year_col, pred_attr])
        crop_train = x_train[crop_col].copy().to_frame()
        crop_test = x_test[crop_col].copy().to_frame()
        if crop_flag:
            x_train = x_train.drop(columns=[crop_col])
            x_test = x_test.drop(columns=[crop_col])
        if (not crop_models) and (not crop_flag):
            x_train = pd.get_dummies(x_train, columns=[crop_col])
            x_test = pd.get_dummies(x_test, columns=[crop_col])
        if not hsg_to_inf:
            x_train = pd.get_dummies(x_train, columns=['SWB_HSG'])
            x_test = pd.get_dummies(x_test, columns=['SWB_HSG'])
        x_train = reindex_df(x_train, column_names=None)
        x_test = reindex_df(x_test, column_names=None)
        if scaling:
            x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
            x_train = pd.DataFrame(x_scaler.fit_transform(x_train), columns=x_train.columns)
            x_test = pd.DataFrame(x_scaler.transform(x_test), columns=x_test.columns)
            y_train = pd.DataFrame(y_scaler.fit_transform(y_train), columns=y_train.columns)
            y_test = pd.DataFrame(y_scaler.transform(y_test), columns=y_test.columns)
        x_train.to_csv(x_train_file, index=False)
        x_test.to_csv(x_test_file, index=False)
        y_train.to_csv(y_train_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        year_train.to_csv(year_train_file, index=False)
        year_test.to_csv(year_test_file, index=False)
        crop_train.to_csv(crop_train_file, index=False)
        crop_test.to_csv(crop_test_file, index=False)
        if scaling:
            pickle.dump(x_scaler, open(x_scaler_file, mode='wb'))
            pickle.dump(y_scaler, open(y_scaler_file, mode='wb'))
    else:
        x_train = pd.read_csv(x_train_file)
        x_test = pd.read_csv(x_test_file)
        y_train = pd.read_csv(y_train_file)
        y_test = pd.read_csv(y_test_file)
        year_train = pd.read_csv(year_train_file)
        year_test = pd.read_csv(year_test_file)
        crop_train = pd.read_csv(crop_train_file)
        crop_test = pd.read_csv(crop_test_file)
        if scaling:
            x_scaler = pickle.load(open(x_scaler_file, mode='rb'))
            y_scaler = pickle.load(open(y_scaler_file, mode='rb'))
    ret_vals = (
        x_train, x_test, y_train.to_numpy().ravel(), y_test.to_numpy().ravel(),
        x_scaler, y_scaler, year_train, year_test, crop_train, crop_test
    )

    return ret_vals


def well_loc_to_field_centroids(pump_df, field_shp_dir, lat_pump='Latitude', lon_pump='Longitude',
                                year_col='ReportYear', field_permit_col='PermitNumb', reproject=True):
    """
    Replace pump coordinates with field centroids
    Original author: Md Fahim Hasan, Modifier: Sayantan Majumdar
    :param pump_df: Input pandas dataframe containing VMP and predictor data.
    :param field_shp_dir: Path to the field polygon shapefile directory.
    :param lat_pump: Latitude column in pump_csv.
    :param lon_pump: Longitude column in pump_csv.
    :param year_col: Name of the year column in pump_df
    :param field_permit_col: Field permit column name
    :param reproject: Set to False if coordinates are already in projected system and conversion from
    geographic to projected is not necessary.
    :return: A joined dataframe of pumps and nearest fields.
    """

    lat_field, lon_field = 'Lat_Field', 'Lon_Field'
    output_pump_df = pd.DataFrame()
    for year in sorted(pump_df[year_col].unique()):
        pump_year_df = pump_df[pump_df[year_col] == year].copy().reset_index(drop=True)
        field_shp = glob(field_shp_dir + '*{}.shp'.format(year))[0]
        print('Replacing pump coordinates with field centroids using', field_shp, '...')
        field_gdf = gpd.read_file(field_shp)
        field_gdf[lat_field] = field_gdf['geometry'].apply(lambda x: x.centroid.y)
        field_gdf[lon_field] = field_gdf['geometry'].apply(lambda x: x.centroid.x)
        pumps_coords = pump_year_df[[lat_pump, lon_pump]]
        fields_coords = field_gdf[[lat_field, lon_field]]
        if reproject:
            latitude_pump = pumps_coords[lat_pump].tolist()
            longitude_pump = pumps_coords[lon_pump].tolist()
            transformer = Transformer.from_crs('EPSG:4326', field_gdf.crs.to_string(), always_xy=True)
            lon_pump_tr, lat_pump_tr = transformer.transform(longitude_pump, latitude_pump)
            pumps_coords = pd.DataFrame()
            pumps_coords[lat_pump] = pd.Series(lat_pump_tr)
            pumps_coords[lon_pump] = pd.Series(lon_pump_tr)
        kdtree_classifier = KDTree(fields_coords.values, metric='euclidean')
        indices = kdtree_classifier.query(pumps_coords, k=1, return_distance=False)
        indices_list = indices.flatten().tolist()
        nearest_fields = field_gdf.iloc[indices_list].reset_index()
        nearest_fields = nearest_fields[[lat_field, lon_field, field_permit_col]]
        pump_year_df = pump_year_df.join(nearest_fields, how='outer').reset_index(drop=True)
        field_lat = pump_year_df[lat_field].tolist()
        field_lon = pump_year_df[lon_field].tolist()
        transformer = Transformer.from_crs(field_gdf.crs.to_string(), 'EPSG:4326', always_xy=True)
        field_lon, field_lat = transformer.transform(field_lon, field_lat)
        pump_year_df = pump_year_df.drop(columns=[lat_field, lon_field])
        pump_year_df[lat_pump] = pd.Series(field_lat)
        pump_year_df[lon_pump] = pd.Series(field_lon)
        output_pump_df = pd.concat([output_pump_df, pump_year_df], ignore_index=True)
    return output_pump_df


def clean_file_dirs(file_dirs, drop_attrs, **kwargs):
    """
    Remove data paths which are dropped from modeling
    :param file_dirs: Tuple of file dirs
    :param drop_attrs: List of attributes which are dropped
    :param kwargs: Pass additional data paths such as openet_data_path, eemetric_data_path, pt_jpl_data_path,
    sims_data_path, etc.
    :return Cleaned file_dirs
    """

    cdl_data_path = kwargs.get('cdl_data_path', '')
    openet_data_path = kwargs.get('openet_data_path', '')
    eemetric_data_path = kwargs.get('eemetric_data_path', '')
    pt_jpl_data_path = kwargs.get('pt_jpl_data_path', '')
    sims_data_path = kwargs.get('sims_data_path', '')
    drop_dict = {
        'CDL': cdl_data_path,
        'Crop(s)': cdl_data_path,
        'OpenET': openet_data_path,
        'EEMETRIC': eemetric_data_path,
        'PT-JPL': pt_jpl_data_path,
        'SIMS': sims_data_path
    }
    file_dirs = list(file_dirs)
    for attr in drop_attrs:
        if attr in drop_dict.keys():
            file_dirs.remove(drop_dict[attr])
    file_dirs = list(filter(lambda val: val != '', file_dirs))
    return file_dirs
