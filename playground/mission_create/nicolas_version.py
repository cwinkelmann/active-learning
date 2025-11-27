# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:05:28 2021

@author: Nicolas

@author: Nicolas Cadieux
njacadieux.gitlab@gmail.com
https://gitlab.com/njacadieux
https://www.youtube.com/channel/UCalCXF9dWWDw3jJ4t26Prhg


GPL-3.0-or-later
This program is free software: you can redistribute it and/or modify it under
the terms of theGNU General Public License as published by the Free Software
Foundation, either version 3 of theLicense, or (at your option) any later
version.This program is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.You should have received a copy of the GNU
General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.

"""
import os
import sys
import geopandas as gpd
import simplekml
from pathlib import Path
# =============================================================================
#                          USER VARIABLES
# =============================================================================

INPUT_PATH = r'/Users/christian/PycharmProjects/hnee/active_learning/mapping/database/mapping/dji_missions_kml/'  # input path to all files
NAME = 'site_code'  # field name for 'name' in input .shp
DESCRIPTION = 'site_code'  # field name for 'description' in input .shp

# =============================================================================
#                      OPTIONAL USER VARIABLES
# =============================================================================

OUTPUT_PATH = Path(INPUT_PATH) / 'DJI_kml_output'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
SCANDIRECTORY_AND_SUBDIRECTORIES = 'no'  # no or yes
EXTENSION_FILTRE = '.shp'  # search for all shape files
STR_FILE_FILTER = ''  # filter for input files name ex: 'Rigaud'= Rigaud_1.shp

# =============================================================================
#                         FUNCTIONS
# =============================================================================


def create_directory(path: str):
    """Create a directory. Will pass if directory has been added
      by another tread."""
    # err = ''
    if os.path.isdir(path):
        pass
    else:
        try:
            os.makedirs(path)
        except Exception as err:
            logger.error(f"Error creating directory {path}: {err}")
            pass
            return err


def scan_directory(path: list):
    file_lst = []
    for (directory, sub_directory, file_name) in os.walk(path):
        for files in file_name:
            if files.lower().endswith(EXTENSION_FILTRE) and STR_FILE_FILTER.lower() in files.lower():# and files.lower().startswith("variable")
                file_filter = os.path.join(directory, files)
                file_lst.append(file_filter)
        if SCANDIRECTORY_AND_SUBDIRECTORIES == 'no':
            break  # this break will exit the loop and limite to 1 directory
        elif SCANDIRECTORY_AND_SUBDIRECTORIES == 'yes':
            pass
    return file_lst


def read_input_file(file: str):
    '''
    Parameters
    ----------
    file : file name str

    Returns
    -------
    None.

    '''
    input_file = gpd.read_file(file)
    fn = os.path.basename(os.path.splitext(file)[0])
    # Check CRS and reproject if needed
    crs = input_file.crs
    if crs is None:
        print(os.path.basename(file)+' has no CRS. Output .kml will not be'
              ' well georeferenced if file is not already in EPSG 4326.')
    elif crs == 'epsg:4326':
        # print(crs)
        pass
    else:
        print(os.path.basename(file) + ' has been reprojected to EPSG 4326.')
        input_file = input_file.to_crs(4326)

    # Check input file fields
    field_in_file = NAME in input_file.columns
    if field_in_file is False:
        sys.exit('"'+NAME + '" field is not found in '
                 + os.path.basename(file)
                 + '. Please verify NAME user variable')
    field_in_file = DESCRIPTION in input_file.columns
    if field_in_file is False:
        sys.exit('"'+DESCRIPTION + '" field is not found in '
                 + os.path.basename(file)
                 + '. Please verify DESCRIPTION user variable')

    # Check geometries for Polygon or LineString
    geom_type = ''
    for q in input_file.geometry:
        if q.type == 'Polygon':
            # print ('Polygon')
            geom_type = 'Polygon'
        elif q.type == 'LineString':
            # print ('LineString')
            geom_type = 'LineString'
        else:
            message = os.path.basename(file) + ' contains objects other then Polygons '\
                     'or LineStrings. Please remove file from input directory'\
                     ' or modify file objects.  Note that MultiPolygon' \
                     ' and MultiLineStrings are not supported.'
            sys.exit(message)

    if geom_type == 'Polygon':
        for index, row in input_file.iterrows():
            geom = (row.geometry)
            ext = list(geom.exterior.coords)
            int_ring = []
            for interior in geom.interiors:
                int_ring.append(list(interior.coords))
            kml = simplekml.Kml()
            pg = kml.newpolygon(name=(row[NAME]), description=(row[DESCRIPTION]))
            pg.outerboundaryis = ext
            if int_ring == []:
                pass
            else:
                pg.innerboundaryis = int_ring
            kml.save(os.path.join(OUTPUT_PATH, fn + '_' + row[NAME] + '.kml'))

    elif geom_type == 'LineString':
        for index, row in input_file.iterrows():
            geom = (row.geometry)
            xyz = list(geom.coords)
            kml = simplekml.Kml()
            l = kml.newlinestring(name=(row[NAME]), description=(row[DESCRIPTION]))
            l.coords = xyz
            kml.save(os.path.join(OUTPUT_PATH, fn + '_' + row[NAME] + '.kml'))
    else:
        print('Only polygons and linestring file are support for now!')

    return os.path.basename(file), '--> .kml'


if __name__ == '__main__':

    # files = (scan_directory(INPUT_PATH))
    # create_directory(OUTPUT_PATH)
    # print('The following files where found in the input directory:')
    # for q in files:
    #     print(os.path.basename(q))
    # print('\n')

    # read_input_file("/Users/christian/PycharmProjects/hnee/active_learning/mapping/database/mapping/dji_missions_kml/FCD_merged_buffer.geojson")
    read_input_file("/Users/christian/PycharmProjects/hnee/active_learning/mapping/database/mapping/dji_missions_kml/FCD07_04052024.geojson")

    # print('Translating file geometries to kml...', '\n')
    # 
    # for message in (map(read_input_file, files)):
    #     print(message[0], message[1])
    # 
    # print('Done')
