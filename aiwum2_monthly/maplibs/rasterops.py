# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu

import rasterio as rio
import numpy as np
import json
import os
import subprocess
import geopandas as gpd
import rioxarray
import pandas as pd
from osgeo import gdal
from rasterio.mask import mask
from rasterio.warp import transform as rio_transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from fiona import transform
from shapely.geometry import box
from glob import glob
from .sysops import make_proper_dir_name, makedirs, make_gdal_sys_call_str


def read_raster_as_arr(raster_file, band=1, get_file=True, rasterio_obj=False, change_dtype=True):
    """
    Get raster array
    :param raster_file: Input raster file path
    :param band: Selected band to read (Default 1)
    :param get_file: Get rasterio object file if set to True
    :param rasterio_obj: Set true if raster_file is a rasterio object
    :param change_dtype: Change raster data type to float if true
    :return: Raster numpy array and rasterio object file (get_file=True and rasterio_obj=False)
    """

    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata is not None:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
    if get_file:
        return raster_arr, raster_file
    return raster_arr


def write_raster(raster_data, raster_file, transform_, outfile_path, no_data_value, ref_file=None, out_crs=None):
    """
    Write raster file in GeoTIFF format
    :param raster_data: Raster data to be written
    :param raster_file: Original rasterio raster file containing geo-coordinates
    :param transform_: Affine transformation matrix
    :param outfile_path: Outfile file path
    :param no_data_value: No data value for raster (default float32 type is considered)
    :param ref_file: Write output raster considering parameters from reference raster file
    :param out_crs: Output crs
    :return: None
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform_ = raster_file.transform
    crs = raster_file.crs
    if out_crs:
        crs = out_crs
    with rio.open(
            outfile_path,
            'w',
            driver='GTiff',
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            dtype=raster_data.dtype,
            crs=crs,
            transform=transform_,
            count=raster_file.count,
            nodata=no_data_value
    ) as dst:
        dst.write(raster_data, raster_file.count)


def shp2raster(input_shp_file, outfile_path, value_field=None, value_field_pos=0, xres=1000., yres=1000.,
               add_value=True, ref_raster=None, burn_value=None):
    """
    Convert Shapefile to Raster TIFF file using GDAL rasterize
    :param input_shp_file: Input Shapefile path
    :param outfile_path: Output TIFF file path
    :param value_field: Name of the value attribute. Set None to use value_field_pos
    :param value_field_pos: Value field position (zero indexing)
    :param xres: Pixel width in geographic units
    :param yres: Pixel height in geographic units
    :param add_value: Set False to disable adding value to existing raster cell
    :param ref_raster: Set to reference raster file path for creating the new raster as per this reference raster
    SRS and resolution
    :param burn_value: Set burn value (int or float). If not None, then add_value, value_field, and value_field_pos
    arguments are ignored
    :return: None
    """

    ext_pos = input_shp_file.rfind('.')
    sep_pos = input_shp_file.rfind(os.sep)
    if sep_pos == -1:
        sep_pos = input_shp_file.rfind('/')
    layer_name = input_shp_file[sep_pos + 1: ext_pos]
    shp_file = gpd.read_file(input_shp_file)
    output_crs = shp_file.crs
    if value_field is None:
        value_field = shp_file.columns[value_field_pos]
    if not ref_raster:
        minx, miny, maxx, maxy = shp_file.geometry.total_bounds
    else:
        _, ref_file = read_raster_as_arr(ref_raster)
        minx, miny, maxx, maxy = get_raster_extent(ref_file, is_rio_obj=True)
        xres, yres = ref_file.res
        output_crs = ref_file.crs.data['init']
    no_data_value = map_nodata()
    if burn_value is None:
        rasterize_options = gdal.RasterizeOptions(
            format='GTiff', outputType=gdal.GDT_Float32,
            outputSRS=output_crs,
            outputBounds=[minx, miny, maxx, maxy],
            xRes=xres, yRes=yres, noData=no_data_value,
            initValues=0., layers=[layer_name],
            add=add_value, attribute=value_field
        )
    else:
        rasterize_options = gdal.RasterizeOptions(
            format='GTiff', outputType=gdal.GDT_Float32,
            outputSRS=output_crs,
            outputBounds=[minx, miny, maxx, maxy],
            xRes=xres, yRes=yres, noData=no_data_value,
            initValues=0., layers=[layer_name],
            burnValues=[burn_value], allTouched=True
        )
    gdal.Rasterize(outfile_path, input_shp_file, options=rasterize_options)


def get_ensemble_avg(image_arr, index, categorical=False):
    """
    Subset image array based on the window size and calculate average
    :param image_arr: Image array whose subset is to be returned
    :param index: Central subset index
    :param categorical: Set True if input array is categorical.
    :return: Mean value
    """

    val = np.nan
    wsize = (1, 1)
    while np.isnan(val):
        startx = index[0] - wsize[0]
        starty = index[1] - wsize[1]
        if startx < 0:
            startx = 0
        if starty < 0:
            starty = 0
        endx = index[0] + wsize[0] + 1
        endy = index[1] + wsize[1] + 1
        limits = image_arr.shape[0] + 1, image_arr.shape[1] + 1
        if endx > limits[0] + 1:
            endx = limits[0] + 1
        if endy > limits[1] + 1:
            endy = limits[1] + 1
        image_subset = image_arr[startx: endx, starty: endy]
        if image_subset.size:
            image_subset = image_subset[~np.isnan(image_subset)]
            if image_subset.size:
                if not categorical:
                    val = np.mean(image_subset)
                else:
                    vals, counts = np.unique(image_subset, return_counts=True)
                    val = vals[np.argmax(counts)]
        wsize = wsize[0] + 1, wsize[1] + 1
    return val


def map_nodata():
    """
    Return fixed no data value for rasters
    :return: Pre-defined/hard-coded default no data value for the USGS MAP Project
    """

    return -32767


def generate_ssebop_raster_list(ssebop_dir, year, month_list):
    """
    Generate SSEBop raster list based on a list of months
    :param ssebop_dir: Input SSEBOp directory
    :param year: SSEBop year in %Y format
    :param month_list: List of months in %m format
    :return: Raster array list and raster reference list as a tuple
    """

    ssebop_raster_arr_list = []
    ssebop_raster_file_list = []
    for month in month_list:
        month_str = str(month)
        if 1 <= month <= 9:
            month_str = '0' + month_str
        pattern = '*' + str(year) + month_str + '*.tif'
        ssebop_file = glob(ssebop_dir + pattern)[0]
        ssebop_raster_arr, ssebop_raster_file = read_raster_as_arr(ssebop_file)
        ssebop_raster_arr_list.append(ssebop_raster_arr)
        ssebop_raster_file_list.append(ssebop_raster_file)
    return ssebop_raster_arr_list, ssebop_raster_file_list


def generate_cummulative_ssebop(ssebop_dir, year_list, start_month, end_month, out_dir):
    """
    Generate cummulative SSEBop data
    :param ssebop_dir: SSEBop directory
    :param year_list: List of years
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param out_dir: Output directory
    :return: None
    """

    month_flag = False
    month_list = []
    actual_start_year = year_list[0]
    no_data_value = map_nodata()
    if end_month <= start_month:
        year_list = [actual_start_year - 1] + list(year_list)
        month_flag = True
    else:
        month_list = range(start_month, end_month + 1)
    for year in year_list:
        actual_year = year
        if month_flag:
            actual_year += 1
        print('Generating cummulative SSEBop for', actual_year, '...')
        if month_flag:
            month_list_y1 = range(start_month, 13)
            month_list_y2 = range(1, end_month + 1)
            ssebop_raster_arr_list1, ssebop_raster_file_list1 = generate_ssebop_raster_list(ssebop_dir, year,
                                                                                            month_list_y1)
            ssebop_raster_arr_list2, ssebop_raster_file_list2 = generate_ssebop_raster_list(ssebop_dir, year + 1,
                                                                                            month_list_y2)
            ssebop_raster_arr_list = ssebop_raster_arr_list1 + ssebop_raster_arr_list2
            ssebop_raster_file_list = ssebop_raster_file_list1 + ssebop_raster_file_list2
        else:
            ssebop_raster_arr_list, ssebop_raster_file_list = generate_ssebop_raster_list(ssebop_dir, year, month_list)
        sum_arr_ssebop = ssebop_raster_arr_list[0]
        for ssebop_raster_arr in ssebop_raster_arr_list[1:]:
            sum_arr_ssebop += ssebop_raster_arr
        sum_arr_ssebop[np.isnan(sum_arr_ssebop)] = no_data_value
        ssebop_raster_file = ssebop_raster_file_list[0]
        out_ssebop = out_dir + 'SSEBop_' + str(year) + '.tif'
        if month_flag:
            out_ssebop = out_dir + 'SSEBop_' + str(actual_year) + '.tif'
        write_raster(sum_arr_ssebop, ssebop_raster_file, transform_=ssebop_raster_file.transform,
                     outfile_path=out_ssebop, no_data_value=no_data_value)
        if month_flag and year == year_list[-1] - 1:
            return


def resample_raster(input_raster_file, resampling_factor=1, resampling_func='near', downsampling=True, ref_raster=None,
                    is_ref_rio=False):
    """
    Resample input raster
    :param input_raster_file: Input raster file
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param ref_raster: Reproject input raster considering another raster
    :param is_ref_rio: Set True if ref_raster is a rasterio object
    :return: If only_resample is False, None is returned. Otherwise the numpy raster array is returned
    """

    src_raster_file = rio.open(input_raster_file)
    rfile = src_raster_file
    if ref_raster:
        rfile = ref_raster
        if not is_ref_rio:
            rfile = rio.open(ref_raster)
    dst_nodata = src_raster_file.nodata
    if downsampling:
        resampling_factor = 1 / resampling_factor
    resampling_dict = {
        'near': Resampling.nearest, 'bilinear': Resampling.bilinear, 'cubic': Resampling.cubic,
        'cubicspline': Resampling.cubic_spline, 'lanczos': Resampling.lanczos,
        'average': Resampling.average, 'mode': Resampling.mode, 'max': Resampling.max,
        'min': Resampling.min, 'med': Resampling.med, 'q1': Resampling.q1, 'q3': Resampling.q3,
    }
    resampling = resampling_dict[resampling_func]
    src_imgdata = src_raster_file.read(
        out_shape=(
            src_raster_file.count,
            int(rfile.height * resampling_factor),
            int(rfile.width * resampling_factor)
        ),
        resampling=resampling
    )
    src_imgdata[src_imgdata == dst_nodata] = np.nan
    return src_imgdata.squeeze()


def get_raster_extent(input_raster, new_crs=None, is_rio_obj=False):
    """
    Get raster extents using rasterio
    :param input_raster: Input raster file path
    :param new_crs: Specify a new crs to convert original extents
    :param is_rio_obj: Set True if input_raster is a Rasterio object and not a string
    :return: Raster extents in a list
    """

    if not is_rio_obj:
        input_raster = rio.open(input_raster)
    raster_extent = input_raster.bounds
    left = raster_extent.left
    bottom = raster_extent.bottom
    right = raster_extent.right
    top = raster_extent.top
    if new_crs:
        raster_crs = input_raster.crs.to_string()
        if raster_crs != new_crs:
            new_coords = reproject_coords(raster_crs, new_crs, [[left, bottom], [right, top]])
            left, bottom = new_coords[0]
            right, top = new_coords[1]
    return [left, bottom, right, top]


def generate_prism_stat_raster(prism_dir, output_file, op='sum', start_month=4, end_month=8, file_ext='.bil'):
    """
    Generate a single prism raster based on 'sum', 'mean', or 'median'
    :param prism_dir: Input PRISM directory
    :param output_file: Output file name
    :param op: Operation to perform. Valid operations include 'sum', 'mean', or 'median'
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param file_ext: File extension
    :return: None
    """

    prism_files = []
    for month in range(start_month, end_month + 1):
        m = str(month)
        if month < 10:
            m = '0' + str(month)
        pattern = '*' + m + file_ext
        prism_files.append(glob(prism_dir + pattern)[0])
    prism_arr_list = []
    prism_obj = None
    for prism_file in prism_files:
        prism_arr, prism_obj = read_raster_as_arr(prism_file)
        prism_arr_list.append(prism_arr)
    prism_stack = np.stack(prism_arr_list)
    if op == 'sum':
        prism_final_arr = prism_stack.sum(axis=0)
    elif op == 'mean':
        prism_final_arr = prism_stack.mean(axis=0)
    else:
        prism_final_arr = np.median(prism_stack, axis=0)
    no_data_value = map_nodata()
    prism_final_arr[np.isnan(prism_final_arr)] = no_data_value
    write_raster(prism_final_arr, prism_obj, transform_=prism_obj.transform, outfile_path=output_file,
                 no_data_value=no_data_value)


def reproject_coords(src_crs, dst_crs, coords=([], []), lat_long_list=False):
    """
    Reproject coordinates. Copied from https://bit.ly/3mBtowB
    Author: user2856 (StackExchange user)
    :param src_crs: Source CRS
    :param dst_crs: Destination CRS
    :param coords: Coordinates as tuple of lists where the list items are long, lat pairs
    :param lat_long_list: If True then pass coords as ([long_list], [lat_list]) else pass coords as
    ([long1, lat1], [long2, lat2], ... [longn,latn])
    :return: Transformed coordinates as transformed long list and lat list (if lat_long_list=True) else as a list of
    transformed long-lat pairs
    """

    if lat_long_list:
        xs, ys = transform.transform(src_crs, dst_crs, coords[0], coords[1])
        return xs, ys
    else:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        xs, ys = transform.transform(src_crs, dst_crs, xs, ys)
        return [[x, y] for x, y in zip(xs, ys)]


def get_swb_var_dicts(raster_dir, year):
    """
    Get SWB variable dicts
    :param raster_dir: Input SWB directory
    :param year: Year in %y format
    """
    asc_file_name_dict = {
        'SWB_HSG': raster_dir + 'Hydrologic_soil_groups__as_read_into_SWB.asc',
        'SWB_AWC': raster_dir + 'Available_water_content__as_read_in_inches_per_foot.asc',
        'SWB_MRD': glob(f'{raster_dir}Maximum_rooting_depth_*{year}*.asc')[0],
        'SWB_SSM': glob(f'{raster_dir}Soil_Storage_Maximum_*{year}*.asc')[0]
    }
    nc_file_name_dict = {
        'SWB_ET': glob(raster_dir + '*actual_et*.nc'),
        'SWB_PPT': glob(raster_dir + '*precipitation*.nc'),
        'SWB_INT': glob(raster_dir + '*interception*.nc'),
        'SWB_IRR': glob(raster_dir + '*irrigation*.nc'),
        'SWB_INF': glob(raster_dir + '*mnet_infiltration*.nc'),
        'SWB_RINF': glob(raster_dir + '*rejected*.nc'),
        'SWB_RO': glob(raster_dir + '*runoff*.nc'),
        'SWB_SS': glob(raster_dir + '*storage*.nc')
    }
    return asc_file_name_dict, nc_file_name_dict


def get_monthly_raster_file_names(raster_dir, year, month, data_name):
    """
    Obtain list of monthly raster names for SSEBop, OpenET, and PRISM
    :param raster_dir: Input raster directory containing monthly rasters
    :param year: Year in %y format
    :param month: Month in %m format
    :param data_name: Name of the data set (used for PRISM and SWB data sets)
    :return: Raster name for GEE files and CDL or list of monthly raster names for SSEBop, OpenET, and PRISM
    depending on raster_dir
    """

    if month < 10:
        month = f'0{month}'
    if 'GEE' in raster_dir:
        raster_file_name = f'{raster_dir}{data_name}_{year}.tif'
    elif 'CDL' in raster_dir:
        raster_file_name = glob(f'{raster_dir}{year}*/*.img')[0]
    elif 'SWB' not in raster_dir:
        if 'SSEBop' in raster_dir:
            raster_file_name = glob(f'{raster_dir}*{year}{month}*.tif')[0]
        elif 'PRISM' in raster_dir:
            raster_file_name = glob(f'{raster_dir}{data_name}/monthly/{year}/prism*{month}.bil')[0]
        else:
            file_list = glob(f'{raster_dir}{year}/*month_{month}*.tif')
            pairs = []
            for rf in file_list:
                pairs.append((os.path.getsize(rf), rf))
            pairs.sort(key=lambda s: s[0])
            raster_file_name = pairs[-1][1]
    else:
        asc_file_name_dict, nc_file_name_dict = get_swb_var_dicts(raster_dir, year)
        if data_name in asc_file_name_dict.keys():
            raster_file_name = asc_file_name_dict[data_name]
        else:
            nc_file = nc_file_name_dict[data_name][0]
            raster_file_name = netcdf_to_tif(nc_file, year, month, raster_dir, data_name)
    return raster_file_name


def netcdf_to_tif(nc_file, year, month, output_dir, data_name):
    """
    Temporally slice a NetCDF file to multiple TIF files
    :param nc_file: Input NetCDF file
    :param year: Year in %y format
    :param month: Month in %m format
    :param output_dir: Output directory to store the TIFs
    :param data_name: Name of the SWB data set
    """

    out_dir = make_proper_dir_name(output_dir + 'TIFs')
    makedirs([out_dir])
    output_tif = f'{out_dir}{data_name}_{year}.tif'
    if not os.path.exists(output_tif):
        rds = rioxarray.open_rasterio(nc_file, decode_times=False)
        _, reference_date = rds.time.attrs['units'].split('days since')
        rds['time'] = pd.date_range(start=reference_date, periods=rds.sizes['time'], freq='D')
        end_day = 30
        if month in [1, 3, 5, 7, 8, 10, 12]:
            end_day += 1
        elif month == 2:
            if year % 4:
                end_day = 28
            else:
                end_day = 29
        if year > 2020:
            year = 2020
        rds_data = rds.sel(
            time=slice(
                '{}-{}-01'.format(year, month),
                '{}-{}-{}'.format(year, month, end_day)
            )
        )
        band_name = nc_file.split('__')[0].split('1000m')[1]
        rds_nodata = rds_data[band_name].attrs['_FillValue']
        if data_name not in ['SWB_RO', 'SWB_SS']:
            rds_data_reduced = rds_data[band_name].sum(dim='time')
            rds_data_reduced = rds_data_reduced.where(rds_data_reduced >= 0., rds_nodata)
        else:
            rds_data_reduced = rds_data[band_name].mean(dim='time')
        rds_data_reduced.rio.write_nodata(rds_nodata, inplace=True)
        rds_data_reduced.rio.write_crs(rds_data.attrs['crs#proj4_string'], inplace=True)
        rds_data_reduced.rio.to_raster(output_tif, nodata=rds_nodata)
    return output_tif


def crop_raster(input_raster_file, ref_file, output_raster_file):
    """
    Crop raster to bbox extents
    :param input_raster_file: Input raster file path
    :param ref_file: Reference raster or shape file to crop input_raster_file
    :param output_raster_file: Output raster file path
    :return: None
    """

    input_raster = rio.open(input_raster_file)
    if '.shp' in ref_file:
        ref_raster_ext = gpd.read_file(ref_file)
    else:
        ref_raster = rio.open(ref_file)
        minx, miny, maxx, maxy = get_raster_extent(ref_raster, is_rio_obj=True)
        ref_raster_ext = gpd.GeoDataFrame({'geometry': box(minx, miny, maxx, maxy)}, index=[0],
                                          crs=ref_raster.crs.to_string())
    ref_raster_ext = ref_raster_ext.to_crs(crs=input_raster.crs.data)
    ref_raster_ext.to_file('../Outputs/Test_Ref_Ext.shp')
    coords = [json.loads(ref_raster_ext.to_json())['features'][0]['geometry']]
    out_img, out_transform = mask(dataset=input_raster, shapes=coords, crop=True)
    out_img = out_img.squeeze()
    write_raster(out_img, input_raster, transform_=out_transform, outfile_path=output_raster_file,
                 no_data_value=input_raster.nodata)


def crop_rasters(input_raster_dir, input_mask_file, outdir, pattern='*.tif', prefix=''):
    """
    Crop multiple rasters in a directory
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param input_mask_file: Mask file (shapefile) used for cropping
    :param outdir: Output directory for storing masked rasters
    :param pattern: Raster extension
    :param prefix: Output file prefix
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + prefix + raster_file[raster_file.rfind(os.sep) + 1:]
        crop_raster(raster_file, input_mask_file, out_raster)


def create_long_lat_grid(input_raster_file, output_dir, target_crs='EPSG:4326', load_files=False, is_rio_object=False):
    """
    Create longitude and latitude grids for a given raster
    :param input_raster_file: Input raster file path
    :param output_dir: Output directory
    :param target_crs: Target CRS for the grid
    :param load_files: Set True to load existing grid files
    :param is_rio_object: Set True if input_raster_file is a Rasterio object
    :return: Longitude list, Latitude list
    """

    long_grid_file = output_dir + 'Long_Grid.npy'
    lat_grid_file = output_dir + 'Lat_Grid.npy'
    if not load_files:
        if not is_rio_object:
            input_raster_file = rio.open(input_raster_file)
            is_rio_object = True
        left, bottom, right, top = get_raster_extent(input_raster_file, is_rio_obj=is_rio_object)
        xres, yres = input_raster_file.res
        long_vals = np.arange(left, right, xres)
        lat_vals = np.arange(top, bottom, -yres)
        long_grid, lat_grid = np.meshgrid(long_vals, lat_vals)
        height, width = input_raster_file.height, input_raster_file.width
        grid_shape = long_grid.shape
        if grid_shape[0] != height:
            diff = np.abs(grid_shape[0] - height)
            long_grid = long_grid[:-diff, :]
            lat_grid = lat_grid[:-diff, :]
        if grid_shape[1] != width:
            diff = np.abs(grid_shape[1] - width)
            long_grid = long_grid[:, :-diff]
            lat_grid = lat_grid[:, :-diff]
        target_crs = CRS.from_string(target_crs)
        long_grid, lat_grid = rio_transform(input_raster_file.crs, target_crs, long_grid.ravel(), lat_grid.ravel())
        np.save(long_grid_file, long_grid)
        np.save(lat_grid_file, lat_grid)
    else:
        long_grid = np.load(long_grid_file).ravel()
        lat_grid = np.load(lat_grid_file).ravel()
    return long_grid, lat_grid


def create_raster_file_dict(file_dirs, gee_files, prism_files, swb_files, month, year):
    """
    Create raster file dictionary containing file paths required for create AIWUM 2.0 prediction maps
    :param file_dirs: Input file directories in the order of SSEBop, CDL, PRISM, GEE Files
    :param gee_files: List of GEE files
    :param prism_files: List of PRISM files
    :param swb_files: List of SWB files
    :param month: Month in %m format
    :param year: Year for which raster file dictionary will be created
    :return: Raster file dictionary. Note: for PRISM, SSEBop, and OpenET, the dictionary value will be a list
    of raster files corresponding to data_start_month and data_end_month
    """

    raster_file_dict = {}
    for file_dir in file_dirs:
        if 'GEE_Files' in file_dir:
            for gee_file in gee_files:
                raster_file = get_monthly_raster_file_names(file_dir, year, 1, gee_file)
                raster_file_dict[gee_file] = raster_file
        elif 'PRISM' in file_dir:
            for prism_file in prism_files:
                raster_file = get_monthly_raster_file_names(file_dir, year, prism_file, month)
                raster_file_dict[prism_file] = raster_file
        elif 'SSEBop' in file_dir:
            raster_file = get_monthly_raster_file_names(file_dir, year, 'SSEBop', month)
            raster_file_dict['SSEBop'] = raster_file
        elif 'SWB' in file_dir:
            asc_file_name_dict, _ = get_swb_var_dicts(file_dir, year)
            for swb_file in swb_files:
                if swb_file in asc_file_name_dict.keys():
                    raster_file = asc_file_name_dict[swb_file]
                else:
                    raster_file = glob(file_dir + f'TIFs/{swb_file}*{year}.tif')[0]
                raster_file_dict[swb_file] = raster_file
        elif 'OpenET' in file_dir:
            if year < 2016:
                raster_file = get_monthly_raster_file_names(file_dirs[0], year, 'SSEBop', month)
            else:
                raster_file = get_monthly_raster_file_names(file_dir, year, 'OpenET', month)
            raster_file_dict['OpenET'] = raster_file
    return raster_file_dict


def generate_predictor_raster_values(raster_file_list, cdl_file, output_dir, year, data, input_extent_file,
                                     verbose=False):
    """
    Generate predictor raster values for AIWUM 2.0 predictions over entire MAP for each year
    :param raster_file_list: Raster file list containing list of raster paths
    :param cdl_file: Rasterio object of the CDL file for the given year
    :param output_dir: Output directory
    :param year: Year for which the predictor raster values are obtained
    :param data: Data name, e.g., 'MOD16', 'ppt', etc.
    :param input_extent_file: Input MAP extent raster file path
    :param verbose: Set True to get additional details about intermediate steps
    :return: Predictor raster values as a linear list
    """

    reproj_dir = make_proper_dir_name(output_dir + 'Reproj_Predictors/' + str(year) + '/' + data)
    makedirs([reproj_dir])
    rf_arr_list = []
    rf_file = None
    raster_vals = []
    for rf in raster_file_list:
        os_sep = rf.rfind(os.sep)
        if os_sep == -1:
            os_sep = rf.rfind('/')
        out_rf = reproj_dir + rf[os_sep + 1:]
        out_rf_cropped = out_rf[: out_rf.rfind('.')] + '_Crop.tif'
        if data in ['OpenET', 'SSEBop', 'ppt', 'tmin', 'tmax']:
            crop_raster(rf, input_extent_file, output_raster_file=out_rf_cropped)
            out_rf_crop_arr, rf_file = read_raster_as_arr(out_rf_cropped)
            rf_arr_list.append(out_rf_crop_arr)
        else:
            if verbose:
                print('Upsampling', rf, '...')
            reproject_raster_gdal(rf, out_rf_cropped, from_raster=cdl_file)
            raster_vals = read_raster_as_arr(out_rf_cropped, get_file=False).ravel()
    if rf_arr_list and rf_file:
        rf_arr_stack = np.stack(rf_arr_list)
        rf_final_arr = np.array([])
        if data in ['OpenET', 'SSEBop', 'ppt']:
            rf_final_arr = rf_arr_stack.sum(axis=0)
        elif data in ['tmin', 'tmax']:
            rf_final_arr = np.median(rf_arr_stack, axis=0)
        rf_out = reproj_dir + data + '_' + str(year) + '.tif'
        write_raster(rf_final_arr, rf_file, outfile_path=rf_out, transform_=rf_file.transform,
                     no_data_value=rf_file.nodata)
        if verbose:
            print('Upsampling', rf_out, '...')
        rf_out_upsampled = reproj_dir + data + '_' + str(year) + '_upsampled.tif'
        reproject_raster_gdal(rf_out, rf_out_upsampled, from_raster=cdl_file)
        raster_vals = read_raster_as_arr(rf_out_upsampled, get_file=False).ravel()
    return raster_vals


def reproject_raster_gdal_syscall(input_raster_file, outfile_path, resampling_factor=1, resampling_func='near',
                                  downsampling=True, from_raster=None, keep_original=False, gdal_path='/usr/bin/',
                                  verbose=True, dst_xres=None, dst_yres=None):
    """
    Reproject raster using GDAL system call.
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster
    :param keep_original: Set True to only use the new projection system from 'from_raster'. The original raster extent
    is not changed
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :param dst_xres: Target xres in input_raster_file units. Set resampling_factor to None
    :param dst_yres: Target yres in input_raster_file units. Set resampling factor to None
    :return: None
    """

    src_raster_file = rio.open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        if isinstance(from_raster, str):
            rfile = rio.open(from_raster)
        else:
            rfile = from_raster
        resampling_factor = 1
    xres, yres = rfile.res
    extent = get_raster_extent(rfile, is_rio_obj=True)
    dst_proj = rfile.crs.to_string()
    no_data = src_raster_file.nodata
    if dst_xres and dst_yres:
        xres, yres = dst_xres, dst_yres
    elif resampling_factor:
        if not downsampling:
            resampling_factor = 1 / resampling_factor
        xres, yres = xres * resampling_factor, yres * resampling_factor
    args = ['-t_srs', dst_proj, '-te', str(extent[0]), str(extent[1]), str(extent[2]), str(extent[3]),
            '-dstnodata', str(no_data), '-r', str(resampling_func), '-tr', str(xres), str(yres), '-ot', 'Float32',
            '-overwrite', input_raster_file, outfile_path]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdalwarp', args=args, verbose=verbose)
    subprocess.call(sys_call)


def reproject_raster_gdal(input_raster_file, outfile_path, resampling_factor=1, resampling_func='near',
                          downsampling=True, from_raster=None, keep_original=False, dst_xres=None, dst_yres=None,
                          output_dtype='float32'):
    """
    Reproject raster using GDALWarp Python API. Use this from Linux only for gdal.GRA_Sum
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster (either raster path or rasterio object)
    :param keep_original: Set True to only use the new projection system from 'from_raster'. The original raster extent
    is not changed
    :param dst_xres: Target xres in input_raster_file units. Set resampling_factor to None
    :param dst_yres: Target yres in input_raster_file units. Set resampling factor to None
    :param output_dtype:  Output data type
    :return: None
    """

    src_raster_file = rio.open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        if isinstance(from_raster, str):
            rfile = rio.open(from_raster)
        else:
            rfile = from_raster
        resampling_factor = 1
    xres, yres = rfile.res
    extent = get_raster_extent(rfile, is_rio_obj=True)
    dst_proj = rfile.crs.to_string()
    no_data = src_raster_file.nodata
    if dst_xres and dst_yres:
        xres, yres = dst_xres, dst_yres
    elif resampling_factor:
        if not downsampling:
            resampling_factor = 1 / resampling_factor
        xres, yres = xres * resampling_factor, yres * resampling_factor
    resampling_dict = {
        'near': gdal.GRA_NearestNeighbour, 'bilinear': gdal.GRA_Bilinear, 'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline, 'lanczos': gdal.GRA_Lanczos,  'sum': gdal.GRA_Sum,
        'average': gdal.GRA_Average, 'mode': gdal.GRA_Mode, 'max': gdal.GRA_Max,
        'min': gdal.GRA_Min, 'med': gdal.GRA_Med, 'q1': gdal.GRA_Q1, 'q3': gdal.GRA_Q3,
    }
    output_dtype_dict = {
        'byte': gdal.GDT_Byte,
        'int16': gdal.GDT_Int16,
        'int32': gdal.GDT_Int32,
        'float32': gdal.GDT_Float32
    }
    warp_options = gdal.WarpOptions(
        outputBounds=extent,
        dstNodata=no_data,
        dstSRS=dst_proj,
        resampleAlg=resampling_dict[resampling_func],
        xRes=xres, yRes=yres,
        outputType=output_dtype_dict[output_dtype],
        multithread=True,
        format='GTiff',
        options=['-overwrite']
    )
    gdal.Warp(outfile_path, input_raster_file, options=warp_options)


def create_cdl_raster_aiwum1(aiwum1_cdl_dir, output_dir, year_list):
    """
    Create yearly CDL rasters based on the high resolution AIWUM 1.1 CDL directory
    :param aiwum1_cdl_dir: AIWUM 1.1 high resolution CDL directory
    :param output_dir: Output directory
    :param year_list: List of years
    :return: Dictionary containing the output CDL file paths as values and year as keys
    """

    cdl_dict = {
        'Corn': 1,
        'Cotton': 2,
        'Rice': 3,
        'Soybeans': 5,
        'Catfish': 92,
        'Other': 0
    }
    cdl_data_dict = {}
    cdl_rio_file = None
    for year in year_list:
        aiwum1_cdl_files = sorted(glob(aiwum1_cdl_dir + '*' + str(year) + '*.tif'))
        cdl_arr_list = []
        for cdl_file in aiwum1_cdl_files:
            if str(year) in cdl_file:
                cdl_file_name = cdl_file[cdl_file.rfind(os.sep) + 1: cdl_file.rfind('.')]
                crop_name = cdl_file_name[:cdl_file_name.find('_')]
                cdl_arr, cdl_rio_file = read_raster_as_arr(cdl_file)
                cdl_arr[cdl_arr > 0] = cdl_dict[crop_name]
                cdl_arr_list.append(cdl_arr)
        cdl_arr = np.stack(cdl_arr_list).sum(axis=0)
        cdl_output = output_dir + 'CDL_' + str(year) + '.tif'
        write_raster(cdl_arr, cdl_rio_file, transform_=cdl_rio_file.transform, outfile_path=cdl_output,
                     no_data_value=0)
        cdl_data_dict[year] = cdl_output
    return cdl_data_dict


def correct_cdl_rasters(input_cdl_dir, nhd_shp_file, lanid_dir, field_shp_dir, output_dir, year_list, cdl_ext='img',
                        lanid_ext='tif'):
    """
    Create yearly CDL rasters (100 m) based on the high resolution CDL data (30 m)
    :param input_cdl_dir: CDL directory
    :param nhd_shp_file: MAP NHD shapefile
    :param lanid_dir: LANID directory
    :param field_shp_dir: Field shapefile directory for permitted boundaries
    :param output_dir: Output directory
    :param year_list: List of years
    :param cdl_ext: Extension of the CDL files (e.g., tif, img, etc.)
    :param lanid_ext: Extension of the LANID files (e.g., tif, img, etc.)
    :return: Dictionary containing the output CDL file paths as values and year as keys
    """

    cdl_data_dict = {}
    cdl_100m_tifs = []
    lanid_100m_tifs = []
    field_100m_tifs = []
    for year in year_list:
        cdl_file = glob(f'{input_cdl_dir}*/{year}*.{cdl_ext}', recursive=True)[0]
        cdl_file_name = cdl_file[cdl_file.rfind(os.sep) + 1: cdl_file.rfind('.')].replace('30m', '100m')
        cdl_100m_tif = f'{output_dir}{cdl_file_name}.tif'
        cdl_100m_tifs.append(cdl_100m_tif)
        reproject_raster_gdal(
            cdl_file,
            outfile_path=cdl_100m_tif,
            resampling_func='mode',
            dst_xres=100,
            dst_yres=100,
            output_dtype='byte'
        )
        lanid_file = glob(f'{lanid_dir}*{year}.{lanid_ext}')[0]
        landid_100m_file = output_dir + lanid_file[lanid_file.rfind(os.sep) + 1: lanid_file.rfind('.')] + '_100m.tif'
        lanid_100m_tifs.append(landid_100m_file)
        reproject_raster_gdal(
            lanid_file,
            outfile_path=landid_100m_file,
            resampling_func='mode',
            from_raster=cdl_100m_tif,
            output_dtype='byte'
        )
        field_shp_file = glob(f'{field_shp_dir}*{year}.shp')[0]
        field_shp_name = field_shp_file[field_shp_file.rfind(os.sep) + 1: field_shp_file.rfind('.')]
        field_100m_tif = f'{output_dir}{field_shp_name}_100m.tif'
        shp2raster(
            field_shp_file,
            outfile_path=field_100m_tif,
            xres=100,
            yres=100,
            burn_value=1.
        )
        field_100m_full_tif = f'{output_dir}{field_shp_name}_100m_full.tif'
        field_100m_tifs.append(field_100m_full_tif)
        reproject_raster_gdal(
            field_100m_tif,
            field_100m_full_tif,
            from_raster=cdl_100m_tif
        )
    nhd_100m_tif = output_dir + 'NHD_MAP_100m.tif'
    shp2raster(
        nhd_shp_file,
        outfile_path=nhd_100m_tif,
        ref_raster=cdl_100m_tifs[0],
        burn_value=1.
    )
    nhd_arr = read_raster_as_arr(nhd_100m_tif, get_file=False)
    for cdl_100m_tif, lanid_100m_tif, field_100m_tif, year in zip(
            cdl_100m_tifs, lanid_100m_tifs, field_100m_tifs, year_list
    ):
        cdl_arr, cdl_rio_file = read_raster_as_arr(cdl_100m_tif, change_dtype=False)
        lanid_arr = read_raster_as_arr(lanid_100m_tif, get_file=False, change_dtype=False)
        field_arr = read_raster_as_arr(field_100m_tif, get_file=False, change_dtype=False)
        cdl_arr[(cdl_arr == 111) | (cdl_arr == 83)] = 92
        cdl_arr[nhd_arr == 1] = 0
        cdl_arr[(cdl_arr != 3) & (cdl_arr != 92) & (lanid_arr == 0) & (field_arr != 1)] = 0
        cdl_output = output_dir + f'CDL_100m_Corrected_{year}.tif'
        write_raster(
            cdl_arr,
            cdl_rio_file,
            transform_=cdl_rio_file.transform,
            outfile_path=cdl_output,
            no_data_value=0
        )
        cdl_data_dict[year] = cdl_output
    return cdl_data_dict
