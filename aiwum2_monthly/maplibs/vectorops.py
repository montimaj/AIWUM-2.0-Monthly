# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu

import geopandas as gpd
import rasterio as rio
import pandas as pd


def reproject_vector(input_vector_file, outfile_path, ref_file, crs='epsg:4326', crs_from_file=True, raster=True):
    """
    Reproject a vector file
    :param input_vector_file: Input vector file path
    :param outfile_path: Output vector file path
    :param crs: Target CRS
    :param ref_file: Reference file (raster or vector) for obtaining target CRS
    :param crs_from_file: If true (default) read CRS from file (raster or vector)
    :param raster: If true (default) read CRS from raster else vector
    :return: Reprojected vector file in GeoPandas format
    """

    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    else:
        crs = {'init': crs}
    output_vector_file = input_vector_file.to_crs(crs)
    output_vector_file.to_file(outfile_path)
    return output_vector_file


def csv2shp(input_csv_file, outfile_path, delim=',', source_crs='epsg:4326', target_crs='epsg:4326',
            long_lat_field_names=('Longitude', 'Latitude')):
    """
    Convert CSV to Shapefile
    :param input_csv_file: Input CSV file path
    :param outfile_path: Output file path
    :param delim: CSV file delimiter
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param long_lat_field_names: Tuple containing names of the longitude and latitude columns respectively
    :return: None
    """

    input_df = pd.read_csv(input_csv_file, delimiter=delim)
    input_df = input_df.dropna(axis=1)
    long, lat = input_df[long_lat_field_names[0]], input_df[long_lat_field_names[1]]
    crs = {'init': source_crs}
    gdf = gpd.GeoDataFrame(input_df, crs=crs, geometry=gpd.points_from_xy(long, lat))
    gdf.to_file(outfile_path)
    if target_crs != source_crs:
        reproject_vector(outfile_path, outfile_path=outfile_path, crs=target_crs, crs_from_file=False, ref_file=None)
