"""
Provides methods for different raster data operations required for the MAP project.
"""

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import rasterio as rio
import numpy as np
import json
import os
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
from sysops import make_proper_dir_name, makedirs


def read_raster_as_arr(
        raster_file: str | rio.DatasetReader,
        band: int = 1,
        get_file: bool = True,
        change_dtype: bool = True
) -> tuple[np.ndarray, rio.DatasetReader] | np.ndarray:
    """Read a raster band as a numpy array.

    Args:
        raster_file (str or rio.DatasetReader): Input raster file path or rasterio DatasetReader object.
        band (int): Selected band to read (Default 1).
        get_file (bool): Get rasterio DatasetReader object file if set to True.
        change_dtype (bool): Change raster data type to float if True. Also, if a no data value exists, then it is set
                             to np.nan if change_dtype is True.

    Returns:
        np.ndarray: Raster numpy array (if, get_file is False).
        tuple (np.ndarray, rio.DatasetReader): A tuple of raster numpy array and rasterio object file (if,
                                               get_file is True and raster_file is a raster file path).
    """
    rasterio_obj = isinstance(raster_file, rio.DatasetReader)
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


def write_raster(
        raster_data: np.ndarray,
        raster_file: rio.DatasetReader,
        transform_: rio.transform,
        outfile_path: str,
        no_data_value: float,
        ref_file: str | None = None,
        out_crs: str | None = None
) -> None:
    """Write raster file in GeoTIFF format.

    Args:
        raster_data (np.ndarray): Raster data (numpy array) to be written.
        raster_file (rio.DatasetReader): Original rasterio raster DatasetReader object containing geo-coordinates.
        transform_ (rio.transform): Affine transformation matrix.
        outfile_path (str): Outfile file path.
        no_data_value (float): No data value for raster (default float32 type is considered).
        ref_file (str): Write output raster considering parameters from reference raster file path
        out_crs (str): Output crs.

    Returns:
        None
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


def shp2raster(
        input_shp_file: str,
        outfile_path: str,
        value_field: str | None = None,
        value_field_pos: int = 0,
        xres: float = 1000.,
        yres: float = 1000.,
        add_value: bool = True,
        ref_raster: str | None = None,
        burn_value: float | None = None
) -> None:
    """Convert Shapefile to Raster TIFF file using GDAL rasterize.

    Args:
        input_shp_file (str): Input shapefile path.
        outfile_path (str): Output TIFF file path.
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing). Only used if value_field is None.
        xres (float): Pixel width in geographic units.
        yres (float): Pixel height in geographic units.
        add_value (bool): Set False to disable adding value to existing raster cell.
        ref_raster (str): Set to reference raster file path for creating the new raster as per this reference raster
                          CRS and resolution.
        burn_value (float or None): Set burn value. If not None, then add_value, value_field, and value_field_pos
                                    arguments are ignored.

    Returns:
        None
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
        minx, miny, maxx, maxy = get_raster_extent(ref_file)
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
    gdal.Rasterize(
        outfile_path,
        input_shp_file,
        options=rasterize_options
    )


def get_ensemble_avg(
        image_arr: np.ndarray,
        index: tuple[int, int],
        categorical: bool = False
) -> float:
    """Subset image array based on the window size and calculate average.

    Args:
        image_arr (np.ndarray): Image array whose subset is to be returned.
        index (tuple (int, int)): Central subset index.
        categorical (bool): Set True if input array is categorical.

    Returns:
        float: Mean value.
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


def map_nodata() -> float:
    """Return fixed no data value for rasters.

    Returns
        float: Pre-defined/hard-coded default no data value for the USGS MAP Project.
    """
    return -32767


def resample_raster(
        input_raster_file: str,
        resampling_factor: int = 1,
        resampling_func: str = 'near',
        downsampling: bool = True,
        ref_raster: str | rio.DatasetReader | None = None,
) -> np.array:
    """Resample input raster.

    Args:
    input_raster_file (str): Input raster file path.
    resampling_factor (int): Resampling factor (default 1).
    resampling_func (str): Resampling function. Valid names include 'near', 'bilinear', 'cubic', 'cubicspline',
                           'lanczos', 'sum', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'.
    downsampling (bool): Downsample raster (default True).
    ref_raster (str or rio.DatasetReader): Reproject input raster considering another raster.

    Returns:
        np.array: Resampled numpy raster array.
    """
    src_raster_file = rio.open(input_raster_file)
    rfile = src_raster_file
    if ref_raster:
        rfile = ref_raster
        is_ref_rio = isinstance(ref_raster, rio.DatasetReader)
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


def get_raster_extent(
        input_raster: str | rio.DatasetReader,
        new_crs: str | None = None
) -> tuple[float, float, float, float]:
    """Get raster extents using rasterio.

    Args:
        input_raster (str or rio.DatasetReader): Input raster file path or rasterio DatasetReader object.
        new_crs (str): Specify a new crs to convert original extents.

    Returns:
        tuple (float, float, float, float): A tuple containing raster extents as (left, bottom, right, top).
    """
    is_rio_obj = isinstance(input_raster, rio.DatasetReader)
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
            new_coords = reproject_coords(
                raster_crs,
                new_crs,
                ((left, bottom), (right, top))
            )
            left, bottom = new_coords[0]
            right, top = new_coords[1]
    return left, bottom, right, top


def reproject_coords(
        src_crs: str,
        dst_crs: str,
        coords: tuple[tuple[float, float], ...]
) -> list[tuple[float, float]]:
    """Reproject coordinates. Copied from https://bit.ly/3mBtowB.

    Author: user2856 (StackExchange user).

    Args:
        src_crs (str): Source CRS.
        dst_crs (str): Destination CRS.
        coords (tuple( tuple(float, float), ...): Coordinates as a tuple of long, lat pairs as tuples.

    Returns:
        list (tuple (float, float)): Transformed coordinates as a list of long, lat pairs as tuples.
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = transform.transform(src_crs, dst_crs, xs, ys)
    return [(x, y) for x, y in zip(xs, ys)]


def get_swb_var_dicts(
        raster_dir: str,
        year: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Get SWB variable dicts.

    Args:
        raster_dir (str): Input SWB directory.
        year (int): Year in YYYY format, i.e, 2014.

    Returns:
        tuple (dict (str, str), dict (str, list (str)): A tuple of two dictionaries corresponding to the SWB .asc files
                                                        and SWB NetCDF files.
    """
    asc_file_name_dict = {
        'SWB_HSG': f'{raster_dir}SWB_HSG/Hydrologic_soil_groups__as_read_into_SWB.asc',
        'SWB_AWC': f'{raster_dir}SWB_AWC/Available_water_content__as_read_in_inches_per_foot.asc',
    }
    swb_mrd = glob(f'{raster_dir}SWB_MRD/Maximum_rooting_depth_*{year}*.asc'),
    swb_ssm = glob(f'{raster_dir}SWB_SSM/Soil_Storage_Maximum_*{year}*.asc')
    if swb_mrd:
        asc_file_name_dict['SWB_MRD'] = swb_mrd[0]
    if swb_ssm:
        asc_file_name_dict['SWB_SSM'] = swb_ssm[0]
    nc_file_name_dict = {
        'SWB_ET': glob(f'{raster_dir}SWB_ET/*actual_et*.nc'),
        'SWB_PPT': glob(f'{raster_dir}SWB_PPT/*precipitation*.nc'),
        'SWB_INT': glob(f'{raster_dir}SWB_INT/*interception*.nc'),
        'SWB_IRR': glob(f'{raster_dir}SWB_IRR/*irrigation*.nc'),
        'SWB_INF': glob(f'{raster_dir}SWB_INF/*mnet_infiltration*.nc'),
        'SWB_RINF': glob(f'{raster_dir}SWB_RINF/*rejected*.nc'),
        'SWB_RO': glob(f'{raster_dir}SWB_RO/*runoff*.nc'),
        'SWB_SS': glob(f'{raster_dir}SWB_SS/*storage*.nc')
    }
    return asc_file_name_dict, nc_file_name_dict


def get_monthly_raster_file_names(
        raster_dir: str,
        year: int,
        month: int,
        data_name: str
) -> str:
    """Get the raster file path for a particular year and month (SSEBop, OpenET, and PRISM 800 m datasets only).

    Args:
        raster_dir (str): Input raster directory containing monthly rasters.
        year (int): Year in YYYY format, e.g., 2015
        month (int): Month in integer format, e.g., 4
        data_name (str): Name of the dataset (used for PRISM 800 m and SWB data sets), e.g., ppt, tmax, SWB_IRR.

    Returns:
        str: The raster file path.
    """
    if month < 10:
        month = f'0{month}'
    if 'GEE' in raster_dir:
        raster_file_name = f'{raster_dir}{data_name}_{year}_{month}.tif'
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


def netcdf_to_tif(
        nc_file: str,
        year: int,
        month: int,
        output_dir: str,
        data_name: str
) -> str:
    """Temporally slice a NetCDF file to a TIF file for a particular year and month.

    Args:
        nc_file (str): Input NetCDF file.
        year (int): Year in YYYY format, e.g., 2015.
        month (int): Month in integer format, e.g., 4.
        output_dir (str): Output directory to store the TIF file.
        data_name (str): Name of the SWB dataset, e.g., SWB_IRR, SWB_MRD, etc.

    Returns:
        str: Output TIF file path.
    """
    out_dir = make_proper_dir_name(output_dir + 'TIFs')
    makedirs(out_dir)
    output_tif = f'{out_dir}{data_name}_{year}_{month}tif'
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


def crop_raster(
        input_raster_file: str,
        ref_file: str,
        output_raster_file: str
) -> None:
    """Crop raster to bbox extents.

    Args:
        input_raster_file (str): Input raster file path.
        ref_file (str): Reference raster or shape file to crop input_raster_file.
        output_raster_file (str): Output raster file path.

    Returns:
        None.
    """
    input_raster = rio.open(input_raster_file)
    if '.shp' in ref_file:
        ref_raster_ext = gpd.read_file(ref_file)
    else:
        ref_raster = rio.open(ref_file)
        minx, miny, maxx, maxy = get_raster_extent(ref_raster)
        ref_raster_ext = gpd.GeoDataFrame({'geometry': box(minx, miny, maxx, maxy)}, index=[0],
                                          crs=ref_raster.crs.to_string())
    ref_raster_ext = ref_raster_ext.to_crs(crs=input_raster.crs.data)
    ref_raster_ext.to_file('../Outputs/Test_Ref_Ext.shp')
    coords = [json.loads(ref_raster_ext.to_json())['features'][0]['geometry']]
    out_img, out_transform = mask(dataset=input_raster, shapes=coords, crop=True)
    out_img = out_img.squeeze()
    write_raster(out_img, input_raster, transform_=out_transform, outfile_path=output_raster_file,
                 no_data_value=input_raster.nodata)


def crop_rasters(
        input_raster_dir: str,
        input_mask_file: str,
        outdir: str,
        pattern: str = '*.tif',
        prefix: str = ''
) -> None:
    """Crop multiple rasters in a directory.

    Args:
        input_raster_dir (str): Directory containing raster files which are named as *_<YYYY>.*
        input_mask_file (str): Mask file (shapefile) used for cropping.
        outdir (str): Output directory for storing masked rasters.
        pattern (str): Raster extension.
        prefix (str): Output file prefix.

    Returns:
        None
    """
    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + prefix + raster_file[raster_file.rfind(os.sep) + 1:]
        crop_raster(raster_file, input_mask_file, out_raster)


def create_long_lat_grid(
        input_raster_file: str | rio.DatasetReader,
        output_dir: str,
        target_crs: str = 'EPSG:4326',
        load_files: bool = False
) -> tuple[np.array, np.array]:
    """Create longitude and latitude grids for a given raster.

    Args:
        input_raster_file (str or rio.DatasetReader): Input raster file path.
        output_dir (str): Output directory.
        target_crs (str): Target CRS for the grid.
        load_files (bool): Set True to load existing grid files.

    Returns:
        tuple (np.array, np.array): Tuple of longitude and latitude arrays.
    """
    long_grid_file = output_dir + 'Long_Grid.npy'
    lat_grid_file = output_dir + 'Lat_Grid.npy'
    if not load_files:
        is_rio_object = isinstance(input_raster_file, rio.DatasetReader)
        if not is_rio_object:
            input_raster_file = rio.open(input_raster_file)
        left, bottom, right, top = get_raster_extent(input_raster_file)
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


def create_raster_file_dict(
        file_dirs: tuple[str, ...],
        gee_files: tuple[str, ...],
        prism_files: tuple[str, ...],
        swb_files: tuple[str, ...],
        month: int,
        year: int
) -> dict[str, str | list[str]]:
    """Create raster file dictionary containing file paths required for create AIWUM 2.0 prediction maps.

    Args:
        file_dirs (tuple (str, ...)): Input file directories in the order of SSEBop, CDL, PRISM, GEE Files.
        gee_files (tuple (str, ...)): List of GEE data names.
        prism_files (tuple (str, ...): List of PRISM 800 m data names.
        swb_files (tuple (str, ...)): List of SWB data names.
        month (int): Month in integer format, i.e., 4, 5, etc.
        year (int): Year (YYYY, i.e., 2014, 2015, etc.) for which raster file dictionary will be created.

    Returns:
        dict (str, str): Raster file dictionary with the raster data names as keys and the corresponding
                         raster file path for the provided year and month as value.
    """
    raster_file_dict = {}
    for file_dir in file_dirs:
        if 'GEE_Files' in file_dir:
            for gee_file in gee_files:
                raster_file = get_monthly_raster_file_names(file_dir, year, 1, gee_file)
                raster_file_dict[gee_file] = raster_file
        elif 'PRISM' in file_dir:
            for prism_file in prism_files:
                raster_file = get_monthly_raster_file_names(file_dir, year, month, prism_file)
                raster_file_dict[prism_file] = raster_file
        elif 'SSEBop' in file_dir:
            raster_file = get_monthly_raster_file_names(file_dir, year, month, 'SSEBop')
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
                raster_file = get_monthly_raster_file_names(file_dirs[0], year, month, 'SSEBop')
            else:
                raster_file = get_monthly_raster_file_names(file_dir, year, month, 'OpenET')
            raster_file_dict['OpenET'] = raster_file
    return raster_file_dict


def generate_predictor_raster_values(
        raster_file_path: str,
        cdl_file: rio.DatasetReader,
        output_dir: str,
) -> list[float]:
    """Generate predictor raster values for AIWUM 2.0 predictions over entire MAP.

    Args:
        raster_file_path (str): Raster file path.
        cdl_file (rio.DatasetReader): Rasterio object of the CDL file for the given year.
        output_dir (str): Output directory.

    Returns:
        list (float): Predictor raster values as a linear list.
    """
    reproj_dir = make_proper_dir_name(f'{output_dir}Reproj_Predictors/')
    makedirs(reproj_dir)
    os_sep = raster_file_path.rfind(os.sep)
    if os_sep == -1:
        os_sep = raster_file_path.rfind('/')
    out_rf = reproj_dir + raster_file_path[os_sep + 1:]
    reproject_raster_gdal(raster_file_path, out_rf, from_raster=cdl_file)
    raster_vals = read_raster_as_arr(out_rf, get_file=False).ravel()
    return raster_vals


def reproject_raster_gdal(
        input_raster_file: str,
        outfile_path: str,
        resampling_factor: int | None = 1,
        resampling_func: str = 'near',
        downsampling: bool = True,
        from_raster: str | rio.DatasetReader | None = None,
        keep_original: bool = False,
        dst_xres: float | None = None,
        dst_yres: float | None = None,
        output_dtype: str = 'float32'
) -> None:
    """Reproject raster using GDALWarp Python API.

    Args:
        input_raster_file (str): Input raster file.
        outfile_path (str): Output file path.
        resampling_factor (int or None): Resampling factor (default 1).
        resampling_func (str): Resampling function. Valid names include 'near', 'bilinear', 'cubic', 'cubicspline',
                               'lanczos', 'sum', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'.
        downsampling (bool): Downsample raster (default True).
        from_raster (str or rio.DatasetReader or None): Reproject input raster considering another raster
                                                        (either raster path or rasterio object).
        keep_original (bool): Set True to only use the new projection system from 'from_raster'.
                              The original raster extent is not changed.
        dst_xres (float or None): Target xres in input_raster_file units. Set resampling_factor to None.
        dst_yres (float or None): Target yres in input_raster_file units. Set resampling factor to None.
        output_dtype (str):  Output data type. Valid data types include 'byte', 'int16', 'int32', 'float32'.

    Returns:
        None
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
    extent = get_raster_extent(rfile)
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


def correct_cdl_rasters(
        input_cdl_dir: str,
        nhd_shp_file: str,
        lanid_dir: str,
        field_shp_dir: str,
        output_dir: str,
        year_list: tuple[int, ...],
        cdl_ext: str = 'img',
        lanid_ext: str = 'tif'
) -> dict[int, str]:
    """Create yearly CDL rasters (100 m) based on the high resolution CDL data (30 m).

    Args:
        input_cdl_dir (str): CDL 30 m directory.
        nhd_shp_file (str): MAP NHD shapefile.
        lanid_dir (str): LANID directory.
        field_shp_dir (str): Field shapefile directory for permitted boundaries.
        output_dir (str): Output directory.
        year_list (tuple (int, ...)): Tuple of years in YYYY format, i.e., (2014, 2015).
        cdl_ext (str): Extension of the CDL files (e.g., tif, img, etc.).
        lanid_ext (str): Extension of the LANID files (e.g., tif, img, etc.).

    Returns:
        dict (int, str): Dictionary containing the output CDL file paths as values and year as keys.
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
