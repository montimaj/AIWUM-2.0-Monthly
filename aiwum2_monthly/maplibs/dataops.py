"""
Provides methods for different data operations required for the MAP project.
"""
import calendar

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu


import pandas as pd
import geopandas as gpd
import ee
import rasterio
import requests
import zipfile
import os
import swifter
import calendar
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
from typing import Any
from sysops import makedirs, make_proper_dir_name, copy_files
from rasterops import read_raster_as_arr, reproject_coords, write_raster, create_long_lat_grid
from rasterops import get_monthly_raster_file_names, resample_raster, crop_raster, crop_rasters
from rasterops import get_raster_extent, map_nodata, create_raster_file_dict
from rasterops import generate_predictor_raster_values
from rasterops import reproject_raster_gdal, correct_cdl_rasters


def get_gee_dict(
        get_key_list: bool = False
) -> dict[str, str | list[str]] | list[str]:
    """Get the available GEE data dictionary.

    Args:
        get_key_list (bool): Set True to get only the key list.

    Returns:
        dict (str, str or list (str)): GEE data dictionary if get_key_list is False.
        list (str): List of keys if get_key_list is True.
    """
    gee_data_dict = {
        'MOD16': 'MODIS/006/MOD16A2',
        'SM_IDAHO': 'IDAHO_EPSCOR/TERRACLIMATE',
        'SMOS_SMAP': ['NASA_USDA/HSL/soil_moisture', 'NASA_USDA/HSL/SMAP10KM_soil_moisture'],
        'DROUGHT': 'GRIDMET/DROUGHT',
        'PPT': 'OREGONSTATE/PRISM/AN81m',
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


def download_gee_data(
        year_list: tuple[int, ...],
        start_month: int,
        end_month: int,
        outdir: str,
        data_extent: tuple[float, float, float, float],
        data: str = 'SSEBop',
        gee_scale: int = 1000
) -> None:
    """Download GEE data.

    Notes: MOD16 has to be divided by 10 (line 38) as its original scale is 0.1 mm/8 days. SMOS_SMAP 2021 may fail to
    download from GEE.

    Args:
        year_list (tuple (int, ...)): Tuple of years in YYYY format, i.e., (2014, 2015,).
        start_month (int): Start month in integer format, i.e, 4, 5, etc.
        end_month (int): End month in integer format, i.e., 4, 5, etc.
        outdir (str): Download directory.
        data_extent (tuple (float, float, float, float)): Data extent as a tuple in (minx, miny, maxx, maxy) format.
        data (str): Name of the data set, MOD16 for MOD16 ET, SM_IDAHO for IDAHO_EPSCOR TERRACLIMATE soil moisture,
                    SMOS_SMAP for SMOS/SMAP soil moisture, 'DROUGHT' for GRIDMET Palmer Drought Severity Index,
                    'PPT' for PRISM precipitation, 'TMIN' and 'TMAX' for PRISM min and max temperatures,
                    'WS' for IDAHO_EPSCOR TERRACLIMATE wind speed, 'SPH' for GRIDMET specific humidity, 'RO' for
                    IDAHO_EPSCOR TERRACLIMATE runoff, 'DEF', and 'DEF_SMAP' for IDAHO_EPSCOR TERRACLIMATE and
                    SMOS/SMAP + MOD16 derived climate water deficit, respectively, 'VPD' for PRISM average
                    water vapour pressure deficit.
        gee_scale (int): GEE Data Scale in meters.

    Returns:
        None
    """
    ee.Initialize()
    gee_data_dict = get_gee_dict()
    if data != 'SMOS_SMAP':
        data_collection = ee.ImageCollection(gee_data_dict[data])
    else:
        data_collection = ee.ImageCollection(gee_data_dict[data][0])
    gee_aoi = ee.Geometry.Rectangle(data_extent)
    for year in year_list:
        for month in range(start_month, end_month + 1):
            start_date = ee.Date.fromYMD(year, month, 1)
            if end_month == 12:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)
            else:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
            gee_data = None
            if data == 'MOD16':
                gee_data = data_collection.select('ET').filterDate(start_date, end_date).sum().divide(10).toDouble()
            elif data == 'SM_IDAHO':
                gee_data = data_collection.select('soil').filterDate(start_date, end_date).first().divide(10).toDouble()
            elif data == 'SMOS_SMAP':
                if year >= 2015 and start_month >= 4:
                    data_collection = ee.ImageCollection(gee_data_dict[data][1])
                else:
                    data_collection = ee.ImageCollection(gee_data_dict[data][0])
                sm_data = data_collection.select('ssm').filterDate(start_date, end_date)
                sm_start_data = sm_data.first()
                sm_dl = sm_data.toList(sm_data.size())
                sm_end_data = ee.Image(sm_dl.get(-1))
                gee_data = sm_end_data.subtract(sm_start_data).toDouble()
            elif data == 'DROUGHT':
                gee_data = data_collection.select('pdsi').filterDate(start_date, end_date).mean().toDouble()
            elif data == 'PPT':
                gee_data = data_collection.select(data.lower()).filterDate(start_date, end_date).sum().toDouble()
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
            if month < 10:
                month = f'0{month}'
            local_file_name = f'{outdir}{data}_{year}_{month}.zip'
            print('Dowloading', local_file_name, '...')
            r = requests.get(gee_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)


def download_ssebop_data(
        year_list: tuple[int, ...],
        start_month: int,
        end_month: int,
        outdir: str
) -> None:
    """Download SSEBop Data

    Args:
        :year_list (tuple (int, ...)): Tuple of years in YYYY format, i.e., (2014, 2015,)
        :start_month (int): Start month in integer format, i.e., 4, 5, etc.
        :end_month (int): End month in integer format, i.e., 4, 5, etc.
        :outdir (str): Download directory.

    Returns:
        None
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
                month_str = f'0{month_str}'
            url = f'{sse_link}m{year}{month_str}.zip'
            local_file_name = f'{outdir}SSEBop_{year}{month_str}.zip'
            r = requests.get(url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)


def extract_data(
        zip_dir: str,
        out_dir: str,
        rename_extracted_files: bool = False
) -> None:
    """Extract data from zip file.

    Args:
        zip_dir (str): Input zip directory.
        out_dir (str): Output directory to write extracted files.
        rename_extracted_files (bool): Set True to rename extracted files according the original zip file name.

    Returns:
        None
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


def get_crop_type_cc_dict() -> tuple[dict[str, int | float], dict[str, float]]:
    """Get CDL crop type dictionary and Crop coefficient dictionary value based on crop name.

    Returns:
        dict (str: int or float):  CDL crop type dictionary.
        dict (str: float): Crop Coefficient Dictionary.
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


def calculate_relative_et(input_df: pd.DataFrame, crop_col: str) -> pd.DataFrame:
    """Calculate relative ET based on crop types.

    Args:
        input_df (pd.DataFrame): Input dataframe containing the crop column.
        crop_col (str): Name of the crop column.

    Returns:
        pd.DataFrame: Modified data frame with the new 'Relative_ET' column for each ET product.
    """
    et_products = ['SSEBop', 'MOD16', 'PT-JPL', 'SIMS', 'OpenET', 'EEMETRIC']
    for et_product in et_products:
        if et_product in input_df.columns:
            new_col = f'Relative_{et_product}'
            input_df[new_col] = input_df[et_product]
            for crop in input_df[crop_col].unique():
                crop_check = input_df[crop_col] == crop
                mean_et = input_df[crop_check][et_product].to_numpy().mean()
                input_df.loc[crop_check, new_col] /= mean_et
    return input_df


def prepare_data(
        monthly_df: pd.DataFrame,
        field_shp_dir: str,
        output_dir: str,
        sub_cols: tuple[str, ...] | None = None,
        lat_pump: str = 'lat_dd',
        lon_pump: str = 'long_dd',
        year_col: str = 'Year',
        crop_col: str = 'crop',
        field_permit_col: str = 'PermitNumb',
        data_list: tuple[str, ...] = ('SSEBop', 'ppt'),
        gee_scale: int = 1000,
        already_prepared: bool = False,
        skip_download: bool = False,
        map_extent_file: str | None = None,
        calc_cc: bool = False,
        calc_relative_et: bool = False,
        calc_eff_ppt: bool = False,
        **kwargs: dict[str, str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Prepare data from existing CSV file and download additional data.

    Args:
        monthly_df (pd.DataFrame): Input monthly dataframe.
        field_shp_dir (str): Field polygon shapefile directory path.
        output_dir (str): Output directory.
        sub_cols (tuple (str, ...) or None): Columns to select from input_csv. Set None to use all columns.
        lat_pump (str): Latitude column in input_csv.
        lon_pump (str): Longitude column in input_csv.
        year_col (str): Name of the Year column.
        crop_col (str): Name of the Crop column.
        field_permit_col (str): Field permit column name.
        data_list (tuple (str, ...)): Tuple of data sets to ingest, valid names include 'SSEBop', 'SM_IDAHO', 'MOD16',
                                      'SMOS_SMAP', 'DROUGHT', 'PPT', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF',
                                      'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin', 'tmean', 'CDL', 'EEMETRIC', 'PT-JPL',
                                      'SIMS','SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF',
                                      'SWB_RINF', 'SWB_RO', 'SWB_SS', 'SWB_MRD', and 'SWB_SSM'.
                                      Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for
                                      USDA-NASS cropland data, and 'SWB*' for SWB products.
        gee_scale (int): GEE data scale in meters.
        already_prepared (bool): Set True if subset csv already exists.
        skip_download (bool): Set True to load already downloaded files.
        map_extent_file (str or None): Set MAP shapefile or raster path to get extents from this shapefile.
                                       Otherwise, extent will be calculated  from the latitude and longitude
                                       of the input_csv.
        calc_cc (bool): Set True to calculate crop coefficients and add 'CC' as a column to the output data frame.
        calc_relative_et (bool): Set True to calculate relative ET and add relative ET columns for each ET product to
                                 the output data frame.
        calc_eff_ppt (bool): Set True to calculate effective precipitation and add EFF_PPT. Only works if SWB_PPT and
                             SWB_INF are in data_list.
        kwargs (dict (str, str)): Pass additional data paths such as prism_data_path, cdl_data_path, openet_data_path,
                                  eemetric_data_path, pt_jpl_data_path, and sims_data_path.

    Returns:
        A tuple containing:
        pd.DataFrame: Subset of input_csv as a Pandas dataframe.
        tuple (str, ...): Tuple of data directories in the order of SSEBop, CDL, PRISM, OpenET, EEMETRIC, PT-JPL,
                          SIMS, and GEE Files.
    """
    print('Preparing data sets...')
    output_csv = output_dir + 'Monthly_Predictor.csv'
    data_dir = make_proper_dir_name(output_dir + 'Downloaded_Data')
    makedirs(data_dir)
    if not already_prepared:
        sub_df = monthly_df.copy(deep=True)
        if sub_cols:
            sub_df = sub_df[sub_cols].copy()
        year_list = tuple(sorted(sub_df[year_col].unique().tolist()))
        if year_col in sub_df.columns:
            sub_df.sort_values(by=year_col, inplace=True)
        sub_df = well_loc_to_field_centroids(sub_df, field_shp_dir, lat_pump, lon_pump, year_col, field_permit_col)
        sub_df, file_dirs = ingest_data(sub_df, data_dir, year_list, lat_pump, lon_pump, year_col, data_list,
                                        gee_scale, skip_download, map_extent_file, **kwargs)
        crop_type_dict, cc_dict = get_crop_type_cc_dict()
        crop_col_check = crop_col in sub_df.columns
        if crop_col_check:
            if calc_cc:
                sub_df['CC'] = sub_df[crop_col].apply(lambda x: cc_dict[x])
            if calc_relative_et:
                sub_df = calculate_relative_et(sub_df, crop_col)
        if calc_eff_ppt and 'SWB_PPT' in data_list and 'SWB_INF' in data_list:
            sub_df['EFF_PPT'] = sub_df.SWB_PPT - sub_df.SWB_INF
        sub_df.to_csv(output_csv, index=False)
    else:
        if sub_cols:
            sub_df = pd.read_csv(output_csv)
        else:
            sub_df = monthly_df
        file_dirs = ingest_data(sub_df, data_dir, get_dirs_only=True, **kwargs)
    print('CSV file prepared...')
    return sub_df, file_dirs


def reindex_df(
        df: pd.DataFrame,
        column_names: tuple[str, ...] | None,
        ordering: bool = False
) -> pd.DataFrame:
    """Reindex dataframe columns.

    Args:
        df (pd.DataFrame): Input pandas DataFrame object.
        column_names (tuple (str, ...)): Data frame column names, these must be df headers.
        ordering (bool): Set True to sort df by column_names.

    Returns:
        pd.DataFrame: Reindexed pandas DataFrame object.
    """
    if column_names is None:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)


def get_monthly_weight_dict(rt_shp_file: str, rt_xls_file: str) -> dict[str, list[float]]:
    """Calculate monthly weights for each crop.

    Args:
        rt_shp_file (str): Real-time sites shapefile path.
        rt_xls_file (str): Real-time XLS file path containing site ids and daily water use.

    Returns:
        dict (str, list (float)): Monthly weight dict where keys are crop types and values include the normalized
                                  weights for each month from January to December.
    """
    avg_wu = {
        'Fish Culture': calculate_aquaculture_monthly_avg_wu(
            rt_shp_file, rt_xls_file
        ), 'Corn': [
            np.nan,
            np.nan,
            np.nan,
            1e-10,
            0.480943768,
            2.935120885,
            2.959187198,
            1.81353687,
            0.285195862,
            np.nan,
            np.nan,
            np.nan
        ], 'Cotton': [
            np.nan,
            np.nan,
            np.nan,
            1e-10,
            0.147144869,
            0.106776469,
            2.048001153,
            1.889042786,
            1.1928,
            np.nan,
            np.nan,
            np.nan
        ], 'Rice': [
            np.nan,
            np.nan,
            np.nan,
            1e-10,
            2.634713034,
            6.896929037,
            7.547371072,
            8.337695059,
            3.684132039,
            np.nan,
            np.nan,
            np.nan
        ], 'Soybeans': [
            np.nan,
            np.nan,
            np.nan,
            1.81364E-06,
            9.30566E-07,
            1.136553566,
            3.515970684,
            3.381447538,
            2.018117186,
            np.nan,
            np.nan,
            np.nan
        ]}
    monthly_weight_dict = {}
    for crop in avg_wu.keys():
        wu_arr = np.array(avg_wu[crop])
        wu_arr = wu_arr / np.nansum(wu_arr)
        monthly_weight_dict[crop] = wu_arr.tolist()
    return monthly_weight_dict


def calculate_aquaculture_monthly_avg_wu(
        rt_shp_file: str,
        rt_xls_file: str,
        site_id_shp: str = 'Site_numbe',
        site_id_xls: str = 'site_no',
        crop_shp: str = 'crop',
        year_shp: str = 'Year',
        state_shp: str = 'State',
        dt_xls: str = 'datetime',
        year_list: tuple[int, ...] = (2018, 2019, 2020)
) -> list[float]:
    """Calculate average monthly water use for aquaculture throughout the year instead of just the growing season.

    Args:
        rt_shp_file (str): Real-time sites shapefile path.
        rt_xls_file (str): Real-time XLS file path containing site ids and daily water use.
        site_id_shp (str): Name of the Site ID field in the shapefile.
        site_id_xls (str): Name of the Site ID field in the XLS file.
        crop_shp (str): Name of the crop column in the shapefile.
        year_shp (str): Name of the year column in the shapefile.
        state_shp (str): Name of the state column in the shapefile.
        dt_xls (str): Name of the date column in the XLS file.
        year_list (tuple (int, ...)): Tuple of years in YYYY format.

    Returns:
        list (float): List containing average monthly water use values.
    """

    rt_gdf = gpd.read_file(rt_shp_file)
    rt_df = pd.read_excel(rt_xls_file)
    rt_df[site_id_xls] = rt_df[site_id_xls].astype(str)
    rt_gdf = rt_gdf[rt_gdf[state_shp] == 'MS']
    rt_gdf = rt_gdf[((rt_gdf[year_shp].isin(year_list)) & (rt_gdf[crop_shp].str.contains('fish', case=False)))]
    rt_gdf[site_id_shp] = rt_gdf[site_id_shp].astype(str)
    rt_gdf = rt_gdf.rename({site_id_shp: site_id_xls}, axis=1)
    rt_df[dt_xls] = pd.to_datetime(rt_df[dt_xls])
    rt_df = rt_df[rt_df[dt_xls].dt.year.isin(year_list)]
    rt_df = rt_df.replace([np.inf, -np.inf], np.nan).dropna()
    rt_df = rt_df[rt_df[rt_df.columns[2]] >= 0]
    rt_df = rt_df.groupby([site_id_xls, pd.Grouper(key=dt_xls, freq="M")])[rt_df.columns[2]].mean().reset_index()
    rt_df[year_shp] = rt_df[dt_xls].dt.year
    rt_df_new = rt_df.merge(rt_gdf, on=[site_id_xls, year_shp])
    rt_df_new['Month'] = rt_df_new[dt_xls].dt.month
    rt_df_new = rt_df_new.sort_values(by=[site_id_xls, year_shp, 'Month'])
    rt_df_new = rt_df_new[['Month', rt_df.columns[2]]]
    monthly_avg_wu = rt_df_new.groupby('Month')[rt_df.columns[2]].mean().reset_index()
    return monthly_avg_wu[rt_df.columns[2]].tolist()


def get_monthly_estimates(vmp_csv, rt_shp_file, rt_xls_file, year_shp, crop_shp, target_attr='AF_Acre'):
    """
    Calculate monthly estimates from the annual VMP data
    :param vmp_csv: VMP CSV file
    :param rt_shp_file: Real-time sites shapefile
    :param rt_xls_file: Real-time XLS file containing site ids and daily water use
    :param year_shp: Name of the year column in rt_shp_file
    :param crop_shp: Name of the crop column in rt_shp_file
    :param target_attr: Attribute name of the water use column to disaggregate
    :return: Monthly disaggregated VMP WU dataframe
    """

    vmp_df = pd.read_csv(vmp_csv)
    crop_col = 'Crop(s)'
    vmp_df[crop_col] = vmp_df[crop_col].apply(
        lambda x: x.strip() if 'Soybean' not in x else 'Soybeans'
    )
    monthly_wt = get_monthly_weight_dict(
        rt_shp_file, rt_xls_file,
    )
    vmp_df = vmp_df[vmp_df[crop_col].isin(monthly_wt.keys())]
    output_df = pd.DataFrame()
    for _, row in vmp_df.iterrows():
        vmp_monthly_df = pd.DataFrame()
        crop = row[crop_col]
        wu = row[target_attr]
        vmp_monthly_df[target_attr] = [wu * wt for wt in monthly_wt[crop]]
        vmp_monthly_df['Month'] = list(range(1, 13))
        vmp_monthly_df[crop_shp] = crop
        vmp_monthly_df['lat_dd'] = row['Latitude']
        vmp_monthly_df['long_dd'] = row['Longitude']
        vmp_monthly_df[year_shp] = row['ReportYear']
        output_df = pd.concat([output_df, vmp_monthly_df])
    output_df = output_df.dropna()
    return output_df


def create_monthly_wu_csv(
        rt_shp_file: str,
        rt_xls_file: str,
        output_dir: str,
        site_id_shp: str = 'Site_numbe',
        site_id_xls: str = 'site_no',
        year_shp: str = 'Year',
        state_shp: str = 'State',
        acre_shp: str = 'combined_a',
        crop_shp: str = 'crop',
        dt_xls: str = 'datetime',
        year_list: tuple[int, ...] = (),
        state_list: tuple[str, ...] = (),
        vmp_csv: str | None = None,
        load_csv: bool = False
) -> pd.DataFrame:
    """Create total monthly water use CSV file from the real-time flowmeter shapefile and XLSX file.

    Args:
        rt_shp_file (str): Real-time sites shapefile path.
        rt_xls_file (str): Real-time XLS file path containing site ids and daily water use.
        output_dir (str): Output directory.
        site_id_shp (str): Name of the Site ID field in the shapefile.
        site_id_xls (str): Name of the Site ID field in the XLS file.
        year_shp (str): Name of the year column in the shapefile.
        state_shp (str): Name of the state column in the shapefile.
        acre_shp (str): Name of the acre column in the shapefile.
        crop_shp (str): Name of the crop column in the shapefile.
        dt_xls (str): Name of the date column in the XLS file.
        year_list (tuple (int, ...)): Tuple of years in YYYY format, i.e., (2014, 2015), etc.
        state_list (tuple (str, ...)): Tuple of abbreviated state list in the MAP, i.e., (MS, AR, LA, etc.)
        vmp_csv (str or None): VMP CSV file if both VMP and real-time data are to be merged after disaggregating the
                               VMP data with real-time weights.
        load_csv (bool): Set True to load existing CSV

    Returns:
        pd.DataFrame: Pandas DataFrame object containing total monthly water use values
    """

    output_dir = make_proper_dir_name(output_dir)
    map_monthly_csv = f'{output_dir}MAP_Monthly.csv'
    if not load_csv:
        print('Merging real-time and VMP data...')
        rt_gdf = gpd.read_file(rt_shp_file)
        rt_df = pd.read_excel(rt_xls_file)
        rt_df[site_id_xls] = rt_df[site_id_xls].astype(str)
        rt_df[dt_xls] = pd.to_datetime(rt_df[dt_xls])
        rt_gdf[site_id_shp] = rt_gdf[site_id_shp].astype(str)
        rt_gdf = rt_gdf.rename({site_id_shp: site_id_xls}, axis=1)
        makedirs(output_dir)
        if state_list:
            state_list = [s.lower() for s in state_list]
            rt_gdf = rt_gdf[rt_gdf[state_shp].str.tolower().isin(state_list)]
        if year_list:
            rt_gdf = rt_gdf[rt_gdf[year_shp].isin(year_list)]
            rt_df = rt_df[rt_df[dt_xls].dt.year.isin(year_list)]
        rt_df = rt_df.replace([np.inf, -np.inf], np.nan).dropna()
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
        rt_df_new['Data'] = 'RT'
        if vmp_csv:
            vmp_monthly = get_monthly_estimates(
                vmp_csv,
                rt_shp_file,
                rt_xls_file,
                year_shp,
                crop_shp
            )
            vmp_monthly['Data'] = 'VMP'
            rt_df_new = pd.concat([rt_df_new, vmp_monthly])
        rt_df_new = rt_df_new.reset_index(drop=True)
        rt_df_new.to_csv(map_monthly_csv, index=False)
    else:
        rt_df_new = pd.read_csv(map_monthly_csv)
    return rt_df_new


def extract_monthly_raster_val(raster_dir, year, month, data, lon, lat, src_crs='EPSG:4326'):
    """
    Extract pixel value from a monthly raster
    :param raster_dir: Predictor data directory
    :param year: Year
    :param month: Month
    :param data: Name of the predictor data
    :param lon: Longitude in WGS84
    :param lat: Latitude in WGS84
    :param src_crs: Source raster CRS
    :return: Extracted pixel value
    """

    raster_file_name = get_monthly_raster_file_names(raster_dir, year, month, data)
    raster_file = rasterio.open(raster_file_name)
    if raster_file_name.endswith('.asc'):
        raster_crs = 'EPSG:5070'
    else:
        raster_crs = raster_file.crs.to_string()
    lon_reproj, lat_reproj = lon, lat
    if data == 'CDL':
        new_coords = reproject_coords(src_crs, raster_crs, ((lon, lat),))
        lon_reproj, lat_reproj = new_coords[0]
    raster_val = raster_file.sample([(lon_reproj, lat_reproj)])
    return next(raster_val)[0]


def ingest_data(
        input_df: pd.DataFrame,
        data_dir: str,
        year_list: tuple[int, ...] or None = None,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude',
        year_col: str = 'Year',
        data_list: tuple[str, ...] = ('SSEBop', 'PPT'),
        gee_scale: int = 1000,
        skip_download: bool = False,
        map_extent_file: str or None = None,
        get_dirs_only: bool = False,
        **kwargs: Any
) -> tuple[pd.DataFrame, tuple[str, ...]] | tuple[str, ...]:
    """Ingest predictor data sets and add them as columns to a Pandas DataFrame object.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        data_dir (str): Data directory.
        year_list (tuple (int)): Tuple of years (YYYY format, i.e., 2014, 2015, etc.) for which data will be ingested.
        lat_col (str): Name of the latitude column used for downloading data.
        lon_col (str): Name of the longitude column used for downloading data.
        year_col (str): Name of the Year column.
        data_list (tuple (str, ...)): Tuple of data sets to ingest, valid names include 'SSEBop', 'SM_IDAHO', 'MOD16',
                                      'SMOS_SMAP', 'DROUGHT', 'PPT', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF',
                                      'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin', 'tmean', 'CDL', 'EEMETRIC', 'PT-JPL',
                                      'SIMS','SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF',
                                      'SWB_RINF', 'SWB_RO', 'SWB_SS', 'SWB_MRD', and 'SWB_SSM'.
                                      Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for
                                      USDA-NASS cropland data, and 'SWB*' for SWB products.
        gee_scale (int): GEE data scale in meters.
        skip_download (bool): Set True to load already downloaded files.
        map_extent_file (str or None): Set MAP shapefile or raster path to get extents from this shapefile.
                                       Otherwise, extent will be calculated  from the latitude and longitude
                                       of input_csv.
        get_dirs_only (bool): Set True to get directories for pre-processed files.
        kwargs (dict (str, str)): Pass additional data paths such as prism_data_path, cdl_data_path, openet_data_path,
                                  eemetric_data_path, pt_jpl_data_path, and sims_data_path.

    Returns:
        If get_dirs_only is True then only the data paths are returned. Else,a tuple containing the following
        is returned:
        pd.DataFrame: Modified input_df with downloaded data.
        tuple (str): tuple of data directories in the order of SSEBop, CDL, PRISM, OpenET, EEMETRIC, PT-JPL, SIMS,
                     and GEE Files.
    """

    gee_zip_dir = make_proper_dir_name(data_dir + 'GEE_Data')
    ssebop_zip_dir = make_proper_dir_name(data_dir + 'SSEBop_Data')
    gee_file_dir = make_proper_dir_name(gee_zip_dir + 'GEE_Files')
    ssebop_file_dir = make_proper_dir_name(ssebop_zip_dir + 'SSEBop_Files')
    ml_csv_dir = make_proper_dir_name(data_dir + 'ML_CSV')
    makedirs((gee_zip_dir, ssebop_zip_dir, gee_file_dir, ssebop_file_dir, ml_csv_dir))
    cdl_data_path = kwargs.get('cdl_data_path', '')
    prism_data_path = kwargs.get('prism_data_path', '')
    swb_data_path = kwargs.get('swb_data_path', '')
    data_paths = (
        ssebop_file_dir, cdl_data_path,
        prism_data_path, gee_file_dir,
        swb_data_path
    )
    if get_dirs_only:
        return data_paths
    gee_products = get_gee_dict(get_key_list=True)
    src_crs = 'EPSG:4326'
    for data in data_list:
        if not skip_download:
            if data == 'SSEBop':
                download_ssebop_data(year_list, 1, 12, ssebop_zip_dir)
                extract_data(ssebop_zip_dir, ssebop_file_dir)
            elif data in gee_products:
                if map_extent_file:
                    if '.shp' in map_extent_file:
                        data_extent = gpd.read_file(map_extent_file).to_crs(src_crs).total_bounds.tolist()
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
        raster_dir = ssebop_file_dir
        if data in ['ppt', 'tmax', 'tmin', 'tmean']:
            raster_dir = prism_data_path
        elif data in gee_products:
            raster_dir = gee_file_dir
        elif data == 'CDL':
            raster_dir = cdl_data_path
        elif data.startswith('SWB'):
            raster_dir = swb_data_path
        input_df[data] = input_df.swifter.apply(
            lambda row: extract_monthly_raster_val(
                raster_dir,
                row[year_col],
                row.Month,
                data,
                row[lon_col],
                row[lat_col],
                src_crs
            ), axis=1
        )
        input_df.to_csv(
            f'{ml_csv_dir}ML_CSV_{data}.csv',
            index=False
        )
    input_df = reindex_df(input_df, column_names=None)
    return input_df, data_paths


def create_map_extent_rasters(
        input_extent_file: str,
        output_dir: str,
        year_list: tuple[int, ...],
        cdl_data_dir: str,
        nhd_shp_file: str,
        lanid_dir: str,
        field_shp_dir: str,
        load_files: bool = False
) -> tuple[str, dict[int, tuple[np.ndarray, rasterio.DatasetReader]]]:
    """Create MAP extent rasters based on CDL rasters.

    Args:
        input_extent_file (str): Input MAP extent shapefile or raster path.
        output_dir (str): Output directory.
        year_list (tuple (int)): Tuple of years in YYYY format, i.e., (2014, 2015,) to be predicted.
        cdl_data_dir (str): CDL 30 m data directory.
        nhd_shp_file (str): MAP NHD shapefile path.
        lanid_dir (str): LANID directory.
        field_shp_dir (str): Field shapefile directory for permitted boundaries.
        load_files (bool): Set True to load existing files.

    Returns:
        A tuple containing:
        str: MAP extent raster directory and
        dict (int, tuple(np.ndarray, rasterio.DatasetReader)): MAP extent raster dictionary where values contain CDL
                                                               array and rasterio DataSetReader object, as tuples.
    """
    map_extent_raster_dir = make_proper_dir_name(output_dir + 'MAP_Extent_Rasters')
    if not load_files:
        makedirs(map_extent_raster_dir)
        cdl_output_dir = make_proper_dir_name(output_dir + 'CDL_100m')
        makedirs(cdl_output_dir)
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


def create_map_prediction_rasters(
        ml_model: Any,
        pred_file_dir: str,
        output_dir: str,
        x_scaler: MinMaxScaler,
        y_scaler: MinMaxScaler,
        map_extent_raster_dict: dict[int, tuple[np.ndarray, rasterio.DatasetReader]],
        crop_col: str,
        load_files: bool = False,
        volume_units: bool = True
) -> str:
    """Create prediction raster (in acre-feet) from prediction CSVs.

    Args:
        ml_model (Any): Pre-fitted ML model object. This can be any sklearn or LightGBM regressor objects.
        pred_file_dir (str): Directory containing the predictor parquet files.
        output_dir (str): Output directory.
        x_scaler (MinMaxScaler): Predictor (X) scaler object.
        y_scaler (MinMaxScaler): Response (y) scaler object.
        map_extent_raster_dict (dict (int, tuple(np.ndarray, rasterio.DatasetReader)): MAP extent raster dictionary
                                                                                       where values are tuples of
                                                                                       CDL array and rasterio object.
        crop_col (str): Name of the crop column in the original VMP data.
        load_files (bool): Set True to load existing model for prediction.
        volume_units (bool): Set False to use mm as water use units instead of m3.

    Returns:
        str: Predicted raster directory
    """
    pred_raster_dir = make_proper_dir_name(output_dir + 'Pred_Raster')
    if not load_files:
        pred_df_list = glob(pred_file_dir + '*.parquet')
        makedirs(pred_raster_dir)
        other_crop_col = crop_col + '_Other'
        unit_dict = {True: 'm3', False: 'mm'}
        for pred_df_file in pred_df_list:
            print('Predicting', pred_df_file)
            pred_df = pd.read_parquet(pred_df_file).astype(float)
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
            month_pos = pred_df_file.rfind('_')
            month = pred_df_file[month_pos + 1: pred_df_file.rfind('.')]
            month = calendar.month_abbr[int(month)]
            year = pred_df_file[month_pos - 4: month_pos]
            map_extent_raster_arr, map_extent_raster_file = map_extent_raster_dict[int(year)]
            for feature in range(num_features):
                nan_pos = np.isnan(pred_arr_copy[:, feature])
                inf_pos = np.isinf(pred_arr_copy[:, feature])
                if nan_pos.size > 0:
                    pred_wu[nan_pos] = map_extent_raster_file.nodata
                if inf_pos.size > 0:
                    pred_wu[inf_pos] = map_extent_raster_file.nodata
            pred_wu = pred_wu.reshape(map_extent_raster_arr.shape)
            if volume_units:
                pred_wu *= 2.471 * 4.047  # acremm/acre to acremm/(1e-2 km2) to m3/(1e-2 km2)
            pred_wu_suffix = f'{pred_raster_dir}AIWUM2_100m_{unit_dict[volume_units]}_{year}_{month}'
            pred_wu_file = pred_wu_suffix + '.parquet'
            pred_df.to_parquet(pred_wu_file, index=False)
            pred_wu_raster = pred_wu_suffix + '.tif'
            write_raster(
                pred_wu, map_extent_raster_file,
                transform_=map_extent_raster_file.transform,
                outfile_path=pred_wu_raster,
                no_data_value=map_extent_raster_file.nodata
            )
            pred_wu_resampled_raster = f'{pred_raster_dir}AIWUM2_1km_{unit_dict[volume_units]}_{year}_{month}.tif'  # m3/km2
            reproject_raster_gdal(
                pred_wu_raster,
                pred_wu_resampled_raster,
                resampling_factor=10,
                downsampling=True,
                resampling_func='sum'
            )
    return pred_raster_dir


def create_map_prediction_files(
        file_dirs: tuple[str, ...],
        data_list: tuple[str, ...],
        data_start_month: int,
        data_end_month: int,
        year_list: tuple[int, ...],
        map_extent_raster_dict: dict[int, tuple[np.ndarray, rasterio.DatasetReader]],
        output_dir: str,
        crop_col: str,
        lat_col: str = 'lat_dd',
        lon_col: str = 'long_dd',
        src_crs: str = 'EPSG:4326',
        load_files: bool = False,
        verbose: bool = False
) -> str:
    """Create predictor parquet files from predictor data sets.

    Args:
        file_dirs (tuple (str, ...)): Input file directories as a tuple in the order of SSEBop, CDL, PRISM, GEE Files.
        data_list (tuple (str, ...)): Tuple of data names, i.e., (SSEBop, RO).
        data_start_month (int): Data start month in integer format, i.e., 4, 5, etc.
        data_end_month (int): Data end month in integer format, i.e., 4, 5, etc.
        year_list (tuple (int)): Tuple of years in YYYY format to be predicted, i.e., (2014, 2015).
        map_extent_raster_dict (dict (int, tuple(np.ndarray, rasterio.DatasetReader)): MAP extent raster dictionary
                                                                                       where values are tuples of
                                                                                       CDL array and rasterio object.
        output_dir (str): Output directory.
        crop_col (str): Name of the crop column in the real-time file.
        lat_col (str): Name of the latitude column in the real-time file.
        lon_col (str): Name of the longitude column in the real-time file.
        src_crs (str): Source CRS for creating lat/long values.
        load_files (bool): Set True to load existing files.
        verbose (bool): Set True to get additional details about intermediate steps.

    Returns:
        str: Directory containing yearly Prediction files (parquet).
    """
    pred_file_dir = make_proper_dir_name(output_dir + 'Predictor_Files')
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
        print('Creating Predictor ...')
        makedirs(pred_file_dir)
        year = year_list[0]
        cdl_file = map_extent_raster_dict[year][1]
        grid_dir = make_proper_dir_name(output_dir + 'Grids')
        makedirs(grid_dir)
        gee_files = []
        prism_files = []
        swb_files = []
        gee_keys = get_gee_dict(True)
        for data_name in data_list:
            if data_name in gee_keys:
                gee_files.append(data_name)
            elif data_name in ['ppt', 'tmax', 'tmin', 'tmean']:
                prism_files.append(data_name)
            elif data_name.startswith('SWB'):
                swb_files.append(data_name)
        if verbose:
            print('Creating lat/long grids...')
        long_vals, lat_vals = create_long_lat_grid(
            cdl_file, grid_dir,
            target_crs=src_crs
        )
        for year in year_list:
            cdl_arr, cdl_file = map_extent_raster_dict[year]
            for month in range(data_start_month, data_end_month + 1):
                pred_df_file = f'{pred_file_dir}Pred_{year}_{calendar.month_abbr[month]}.parquet'
                print('Creating', pred_df_file, '...')
                pred_df = pd.DataFrame()
                if verbose:
                    print('Retrieving CDL lat/long', year, 'values...')
                pred_df[crop_col] = cdl_arr.ravel()
                pred_df[lat_col] = lat_vals
                pred_df[lon_col] = long_vals
                raster_file_dict = create_raster_file_dict(
                    file_dirs, tuple(gee_files), tuple(prism_files),
                    tuple(swb_files), month, year
                )
                for data in data_list:
                    raster_file_path = raster_file_dict[data]
                    raster_vals = generate_predictor_raster_values(raster_file_path, cdl_file, output_dir)
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
                pred_df.to_parquet(pred_df_file, index=False)
    return pred_file_dir


def create_prediction_map(
        ml_model: Any,
        input_extent_file: str,
        file_dirs: tuple[str, ...],
        output_dir: str,
        year_list: tuple[int, ...],
        data_list: tuple[str, ...],
        input_cdl_dir: str,
        nhd_shp_file: str,
        lanid_dir: str,
        field_shp_dir: str,
        data_start_month: int,
        data_end_month: int,
        crop_col: str,
        lat_col: str,
        lon_col: str,
        load_pred_file: bool = False,
        load_map_extent: bool = False,
        load_pred_raster: bool = False,
        x_scaler: MinMaxScaler | None = None,
        y_scaler: MinMaxScaler | None = None,
        volume_units: bool = True,
        src_crs: str = 'EPSG:4326'
) -> tuple[str, str, str]:
    """Create MAP prediction raster.

    Args:
        ml_model (Any): Pre-fitted ML model object. This can be any sklearn or LightGBM regressor objects.
        input_extent_file (str): Input MAP extent shapefle or raster file path.
        file_dirs (tuple (str, ...)): Input file directories as a tuple in the order of SSEBop, CDL, PRISM, GEE Files.
        output_dir (str): Output directory.
        year_list (tuple (int, ...)): List of years to be predicted, i.e., (2014, 2015).
        data_list (tuple (str, ...)): List of data names, i.e., (SSEBop, RO).
        input_cdl_dir (str): CDL directory containing CDL rasters at 30 m resolution.
        nhd_shp_file (str): MAP NHD shapefile path.
        lanid_dir (str): LANID directory.
        field_shp_dir (str): Field shapefile directory for permitted boundaries.
        data_start_month (int): Data start month in integer format, i.e., 4, 5, etc.
        data_end_month (int): Data end month in integer format, i.e., 4, 5, etc.
        crop_col (str): Name of the crop column in the real-time file.
        lat_col (str): Name of the latitude column in the real-time file.
        lon_col (str): Name of the longitude column in the real-time file.
        load_map_extent (bool): Set True to load existing MAP extent rasters.
        load_pred_file (bool): Set True to load existing Prediction parquet files.
        load_pred_raster (bool): Set True to load existing Prediction rasters.
        x_scaler (MinMaxScaler): Predictor (X) scaler object.
        y_scaler (MinMaxScaler): Response (y) scaler object.
        volume_units (bool): Set False to use mm as water use units instead of m3.
        src_crs (str): Source CRS for creating lat/long values.

    Returns:
        tuple(str): Predicted CSV directory, raster directory, and MAP extent raster as a tuple.
    """

    output_dir = make_proper_dir_name(output_dir)
    map_extent_raster_dir, map_extent_raster_dict = create_map_extent_rasters(
        input_extent_file, output_dir, year_list,
        input_cdl_dir, nhd_shp_file, lanid_dir,
        field_shp_dir, load_map_extent
    )
    pred_file_dir = create_map_prediction_files(
        file_dirs, data_list,
        data_start_month, data_end_month,
        year_list, map_extent_raster_dict,
        output_dir, crop_col, lat_col, lon_col,
        src_crs, load_pred_file
    )
    pred_raster_dir = create_map_prediction_rasters(
        ml_model, pred_file_dir, output_dir,
        x_scaler, y_scaler, map_extent_raster_dict,
        crop_col, load_pred_raster, volume_units
    )
    return pred_file_dir, pred_raster_dir, map_extent_raster_dir


def compare_aiwums_map(
        aiwum1_tot_dir: str,
        aiwum2_dir: str,
        input_extent_file: str,
        output_dir: str,
        volume_units: bool = True
) -> None:
    """Compare AIWUM 1.1 and 2.0, create AIWUM 2 maps with volumetric units (m3), and generate bar plot for entire
    MAP region.

    Args:
        aiwum1_tot_dir (str): AIWUM1.1 total crop water use raster directory. Note: AIWUM1.1 predictions are in m^3.
        aiwum2_dir (str): Directory containing AIWUM 2.0 predicted rasters. Note: AIWUM2.0 predictions are in acremm.
        input_extent_file (str): Input MAP extent raster file path from AIWUM 1.1 or a shapefile.
        output_dir (str): Output directory to store file.
        volume_units (bool): Set False to use mm as water use units instead of m3.

    Returns:
        None
    """
    aiwum_compare_dir = make_proper_dir_name(output_dir + 'AIWUM_Comparison')
    aiwum1_crop_dir = make_proper_dir_name(aiwum1_tot_dir + 'Cropped')
    makedirs((aiwum_compare_dir, aiwum1_crop_dir))
    if input_extent_file.endswith('.shp'):
        crop_rasters(aiwum1_tot_dir, input_extent_file, aiwum1_crop_dir, prefix='AIWUM1')
    else:
        aiwum1_crop_dir = aiwum_compare_dir
        copy_files(aiwum1_tot_dir, aiwum1_crop_dir, prefix='AIWUM1', pattern='y*.tif')
    aiwum1_rasters = sorted(glob(aiwum1_crop_dir + 'AIWUM1*.tif'))
    aiwum2_rasters = sorted(glob(aiwum2_dir + '*1km*.tif'))
    no_data = map_nodata()
    aiwum_tot_pred_df = pd.DataFrame()
    aiwum1_arr_list = []
    aiwum2_arr_list = []
    diff_arr_list = []
    aiwum2_ref_file = None
    for idx, aiwum2_raster in enumerate(aiwum2_rasters):
        aiwum2_arr, aiwum2_ref_file = read_raster_as_arr(aiwum2_raster)
        if not volume_units:
            aiwum2_arr *= 2.471 * 4.047  # needs to converted to volume units to compare with AIWUM 1.1
        year = aiwum2_raster[aiwum2_raster.rfind('_') + 1: aiwum2_raster.rfind('.')]
        if idx < len(aiwum1_rasters):
            aiwum2_arr_list.append(aiwum2_arr)
            aiwum1_raster = aiwum1_rasters[idx]
            aiwum1_arr = resample_raster(aiwum1_raster, ref_raster=aiwum2_ref_file)
            aiwum1_arr[np.isnan(aiwum2_arr)] = np.nan
            aiwum1_arr_list.append(aiwum1_arr)
            df = {'Year': [int(year)], 'AIWUM1.1': [np.nansum(aiwum1_arr)], 'AIWUM2.0': [np.nansum(aiwum2_arr)]}
            aiwum_tot_pred_df = pd.concat([aiwum_tot_pred_df, pd.DataFrame(data=df)])
            aiwum1_arr_copy = deepcopy(aiwum1_arr)
            aiwum1_arr_copy[np.isnan(aiwum1_arr_copy)] = no_data
            aiwum1_out = f'{aiwum_compare_dir}AIWUM1_1km_m3_{year}.tif'
            write_raster(
                aiwum1_arr_copy,
                aiwum2_ref_file,
                transform_=aiwum2_ref_file.transform,
                outfile_path=aiwum1_out,
                no_data_value=no_data
            )
            diff_arr = aiwum2_arr - aiwum1_arr
            diff_arr_list.append(diff_arr)
            diff_out = f'{aiwum_compare_dir}Diff_1km_m3_{year}.tif'
            diff_arr[np.isnan(diff_arr)] = no_data
            write_raster(
                diff_arr,
                aiwum2_ref_file,
                transform_=aiwum2_ref_file.transform,
                outfile_path=diff_out,
                no_data_value=no_data
            )
        aiwum2_out = f'{aiwum_compare_dir}AIWUM2_1km_m3_{year}.tif'
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


def split_data_train_test(
        input_df: pd.DataFrame,
        pred_attr: str = 'AF_Acre',
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = 0.2,
        test_year: tuple[int, ...] | bool = True,
        year_col: str = 'ReportYear',
        crop_col: str | None = None,
        split_strategy: int = 2
) -> tuple[pd.DataFrame, ...]:
    """Split data yearly, randomly, or based on train-test percentage based on year or crop. For the last option,
    by default test_size amount of data is kept from each year for testing.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        shuffle (bool): Default True for shuffling.
        random_state (int): Random state used during train test split.
        test_size (float): Test data size (0<=test_size<=1).
        test_year (tuple(int, ...) or bool): If split_strategy = 1, then this needs to be a tuple of years in YYYY
                                             format, i.e., (2014, 2015), and the test data is created from these years.
                                             For split_strategy=2, set this to True if the test data needs to be created
                                             based on year_col. Otherwise, if split_strategy=2 and test_year=False then
                                             the test data is created using crop_col.
        year_col (str): Name of the year column.
        crop_col (str): Name of the crop column. By default, it's None.
        split_strategy (int): If 1, Split train test data based on year_col. If 2, then test_size amount of data from
                              year_col or crop_col are kept for testing and rest for training;
                              for this option, test-year should have a tuple of integers or a True value, else splitting
                              is based on crop_col. For any other value of split-strategy, the data are randomly split.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
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


def split_data_train_test_ratio(
        input_df: pd.DataFrame,
        pred_attr: str = 'AF_Acre',
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = 0.2,
        test_year: bool = True,
        year_col: str = 'ReportYear',
        crop_col: str | None = None
) -> tuple[pd.DataFrame, ...]:
    """Split data based on train-test percentage based on year or crop. By default test_size amount of data is kept from
    each year for testing.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        shuffle (bool): Default True for shuffling.
        random_state (int): Random state used during train test split.
        test_size (float): Test data size (0<=test_size<=1).
        test_year (bool): If True, build test data from the year_col. Otherwise, use crop_col.
        year_col (str): Name of the year column.
        crop_col (str or None): Name of the crop column. By default it's None.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
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


def split_data_yearly(
        input_df: pd.DataFrame,
        pred_attr: str = 'AF_Acre',
        test_years: tuple[int, ...] = (2016,),
        year_col: str = 'ReportYear',
        shuffle: bool = True,
        random_state: int = 0
) -> tuple[pd.DataFrame, ...]:
    """Split data based on a particular year.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        test_years (tuple (int, ...)): Build test data from only these tuple of years, i.e., (2014, 2015).
        year_col (str): Name of the year column.
        shuffle (bool): Set False to stop data shuffling.
        random_state (int): Random state used during train test split.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
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


def process_outliers(
        input_df: pd.DataFrame,
        target_attr: str,
        crop_col: str,
        year_col: str,
        operation: int = 2
) -> pd.DataFrame:
    """Remove outliers from a dataframe based on target_attr.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        target_attr (str): Target attribute based on which outlier removal will occur.
        crop_col (str): Name of the crop column.
        year_col (str): Name of the year column.
        operation (int): Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
                         by each crop, 3 for removing outliers by each year, or 4 for removing as per AIWUM 1 based on
                         irrigation thresholds. Note: for this project we only process outliers above the boxplot upper
                         limit for 1-3.

    Returns:
        pd.DataFrame: Outlier removed input_df.
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
    input_df = input_df[input_df[crop_col].isin(max_irrigation_dict.keys())]
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
            print(f'{selection_col} {val} outliers: {outliers}')
            num_outliers += outliers
            input_df.loc[selection, 'Outlier'] = invalid_idx
        input_df = input_df[input_df['Outlier'] == False]
        input_df = input_df.drop(columns='Outlier')
    input_df = input_df.dropna()
    print('Old DF rows = {}, New DF rows = {}'.format(init_rows, input_df.shape[0]))
    print(f'{num_outliers} outliers removed...')
    return input_df


def convert_to_mm(
        input_df: pd.DataFrame,
        pred_attr: str = 'AF_Acre'
) -> pd.DataFrame:
    """Convert feature units to mm for SWB data and Groundwater Pumpings.

    Args:
        input_df (pd.DataFrame): Cleaned pandas input DataFame object.
        pred_attr (str): Name of the target attribute (originally in ft).

    Returns:
        pd.DataFrame: Updated input_df with pred_attr in mm instead of ft. The pred_attr column name is unchanged.
    """
    input_df[pred_attr] *= 304.8
    for column in input_df.columns:
        if column != 'SWB_HSG' and (column == 'EFF_PPT' or column.startswith('SWB')):
            input_df[column] *= 25.4
    return input_df


def convert_hsg_to_inf(
        input_df: pd.DataFrame,
        drop_hsg: bool = True,
        hsg_col: str = 'SWB_HSG'
) -> pd.DataFrame:
    """Reclassify HSG into 4 groups and convert SWB HSG to Infiltration Rates.

    Args:
        input_df (pd.DataFrame): Cleaned input data frame containing hsg_col column.
        drop_hsg (bool): Set False to disable dropping HSG column. If False HSG_INF column won't be created.
        hsg_col (str): Name of the HSG column.

    Returns:
        pd.DataFrame: Updated pandas DataFame with the HSG_INF column if hsg_col is present in input_df. Otherwise,
        returns the original input_df.
    """

    if hsg_col in input_df.columns:
        input_df[hsg_col] = input_df[hsg_col].apply(lambda x: x if x <= 4 else x - 4)
        if drop_hsg:
            inf_dict = {
                1: 7.62,
                2: 5.715,
                3: 2.54,
                4: 0.635,
                -9999: 4.1275
            }
            input_df['HSG_INF'] = input_df[hsg_col].apply(lambda x: np.nan if np.isnan(x) else inf_dict[x])
            input_df = input_df.drop(columns=[hsg_col])
    return input_df


def create_train_test_data(
        input_df: pd.DataFrame,
        output_dir: str,
        pred_attr: str = 'AF_Acre',
        drop_attr: tuple[str, ...] = ('Year',),
        test_size: float = 0.2,
        test_year: tuple[int, ...] | bool = True,
        year_col: str = 'Year',
        random_state: int = 42,
        already_created: bool = False,
        scaling: bool = False,
        year_list: tuple[int, ...] = (2014,),
        crop_col: str = 'crop',
        split_strategy: int = 2,
        outlier_op: int | None = 2,
        shuffle: bool = True,
        crop_models: bool = False,
        hsg_to_inf: bool = True
) -> tuple:
    """Create train and test data.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame.
        output_dir (str): Output directory.
        pred_attr (str): Attribute to be predicted.
        drop_attr (tuple (str, ...)): Tuple of attributes to drop from model training.
        test_size (float): Test size between (0, 1).
        test_year (tuple (int, ...) or bool): If split_strategy = 1, then this needs to be a tuple of years in YYYY
                                             format, i.e., (2014, 2015), and the test data is created from these years.
                                             For split_strategy=2, set this to True if the test data needs to be created
                                             based on year_col. Otherwise, if split_strategy=2 and test_year=False then
                                             the test data is created using crop_col.
        year_col (str): Name of the year column.
        random_state (int): Random state used during train test split.
        already_created (bool): Set True to load existing train and test data.
        scaling (bool): Set True to perform minmax scaling.
        year_list (tuple (int,...)): Tuple of years in YYYY format, i.e., (2014, 2015) to build the data set.
        crop_col (str): Name of the crop column to create dummy variables.
        split_strategy (int): If 1, Split train test data based on year_col. If 2, then test_size amount of data from
                              year_col or crop_col are kept for testing and rest for training;
                              for this option, test-year should have some value other than None, else splitting is
                              based on crop_col. For any other value of split-strategy, the data are randomly split.
        outlier_op (int): Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
                          by each crop, 3 for removing outliers by each year, or 4 for removing as per AIWUM 1 based on
                          irrigation thresholds. Note: for this project we only process outliers above the boxplot
                          upper limit for 1-3. Set None to disable outlier processing.
        shuffle (bool): Set False to stop data shuffling.
        crop_models (bool): Set True for individual crop models for each crop type. If True, dummies are not created.
        hsg_to_inf (bool): Set False to disable creating the infiltration rate column based on the HSGs. Only works when
                           SWB_HSG is present in input_df and not in drop_attr.

    Returns:
        tuple: A tuple containing X_train, X_test as pandas data frames, y_train, y_test as numpy arrays.
        If scaling=True, then x_scaler and y_scaler are also returned. Year_train, Year_test, Crop_train, and Crop_test
        are returned as well for AIWUM analysis later on.
    """
    makedirs(make_proper_dir_name(output_dir))
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
        input_df = input_df.replace([np.inf, -np.inf], np.nan).dropna()
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


def well_loc_to_field_centroids(
        pump_df: pd.DataFrame,
        field_shp_dir: str,
        lat_pump: str = 'Latitude',
        lon_pump: str = 'Longitude',
        year_col: str = 'Year',
        field_permit_col: str = 'PermitNumb',
        reproject: bool = True
) -> pd.DataFrame:
    """Replace pump coordinates with field centroids.

    Original author: Md Fahim Hasan, Modifier: Sayantan Majumdar

    Args:
        pump_df (pd.DataFrame): Input pandas dataframe containing VMP and predictor data.
        field_shp_dir (str): Path to the field polygon shapefile directory.
        lat_pump (str): Latitude column in pump_csv.
        lon_pump (str): Longitude column in pump_csv.
        year_col (str): Name of the year column in pump_df.
        field_permit_col (str): Field permit column name.
        reproject (bool): Set to False if coordinates are already in projected system and conversion from
                          geographic to projected is not necessary.

    Returns:
        pd.DataFrame: A joined dataframe of pumps and nearest fields.
    """
    lat_field, lon_field = 'Lat_Field', 'Lon_Field'
    output_pump_df = pd.DataFrame()
    pump_cond = (pump_df.State == 'MS') | (pump_df.Data == 'VMP')
    pump_df_nofield = pump_df[~pump_cond].copy(deep=True).reset_index(drop=True)
    pump_df = pump_df[pump_cond]
    for year in sorted(pump_df[year_col].unique()):
        pump_year_df = pump_df[pump_df[year_col] == year].copy().reset_index(drop=True)
        field_shp = glob(f'{field_shp_dir}*{year}.shp')[0]
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
    output_pump_df = pd.concat([output_pump_df, pump_df_nofield]).reset_index(drop=True)
    return output_pump_df


def clean_file_dirs(
        file_dirs: tuple[str, ...],
        drop_attrs: tuple[str, ...],
        **kwargs: dict[str, str]
) -> list[str]:
    """Remove data paths which are dropped from modeling.

    Args:
        file_dirs (tuple (str, ...)): Tuple of file dirs.
        drop_attrs (tuple (str, ...)): Tuple of attributes which are dropped.
        kwargs (dict (str, str)): Pass additional data paths such as openet_data_path, eemetric_data_path,
                                  pt_jpl_data_path, and sims_data_path, etc.

    Returns:
        list (str): Cleaned file_dirs.
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
