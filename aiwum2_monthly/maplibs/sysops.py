# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu

import os
import shutil
from glob import glob


def make_gdal_sys_call_str(gdal_path, gdal_command, args, verbose=True):
    """
    Make GDAL system call string
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param gdal_command: GDAL command to use
    :param args: GDAL arguments as a list
    :param verbose: Set True to print system call info
    :return: GDAL system call string,
    """

    sys_call = [gdal_path + gdal_command] + args
    if os.name == 'nt':
        gdal_path += 'OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if directory_name is not None:
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)


def make_proper_dir_name(dir_str):
    """
    Append os.sep to dir if not present
    :param dir_str: Directory path
    :return: Corrected directory path
    """

    if dir_str is None:
        return None
    sep = [os.sep, '/']
    if dir_str[-1] not in sep:
        return dir_str + os.sep
    return dir_str


def boolean_string(s):
    """
    Return True/False based on a boolean string
    :param s: String object
    :return: True or False
    """

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def copy_files(input_dir, target_dir, pattern='*.tif', prefix='', verbose=True):
    """
    Copy files from input directories to target directory
    :param input_dir: Input directory
    :param target_dir: Target directory
    :param pattern: File pattern list ordered according to input_dir_list
    :param prefix: Prefix string for output file
    :param verbose: Set True to get info on copy
    :return: None
    """

    file_list = glob(input_dir + pattern)
    for f in file_list:
        file_name = f[f.rfind(os.sep) + 1:]
        outfile = target_dir + prefix + file_name
        if verbose:
            print('Copying', f, 'to', outfile, '...')
        shutil.copyfile(f, outfile)
