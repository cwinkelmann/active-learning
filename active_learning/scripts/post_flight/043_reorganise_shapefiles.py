"""
The shapefiles are not very well organised, named or encoded
This script will reorganise the shapefiles into a more structured form, correct the CRS and convert them to geojson
"""
import typing

import pandas as pd
import shutil
from loguru import logger

from pathlib import Path

from active_learning.util.projection import project_gdfcrs
from geospatial_transformations import convert_to_cog, batch_convert_to_cog
import geopandas as gpd
shapefile_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Counts shp')

shp_files = [f for f in shapefile_base_path.glob('**/*counts.shp') if not f.name.startswith('.')]


# copy images to new folder
output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations")
orthomosaic_paths = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL")
progress_report_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')

df_progress_report = pd.read_csv(progress_report_file)

missing_report_entries: typing.List[str] = []
missing_orthomosaics_entries: typing.List[str] = []

import re
def get_prefix(s):
    match = re.match(r'^[A-Za-z]+', s)  # Match only leading letters
    return match.group(0) if match else ''

for i, shp_file_path in enumerate(shp_files):
    shp_name = shp_file_path.name
    logger.info(f"Processing {shp_name}")
    gdf = gpd.read_file(shp_file_path)
    island_code = shp_name.split('_')[0]
    possible_orthomosaic_name = shp_name.split(' ')[0]

    images_path = orthomosaic_paths / island_code / f"{possible_orthomosaic_name}.tif"
    if not images_path.exists():
        logger.error(f"Orthomosaic not found: {images_path}")
        missing_orthomosaics_entries.append(images_path)
        continue
    gdf = project_gdfcrs(gdf, images_path)
    expert_value = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Expert']
    gdf["island"]  = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Island']
    gdf["field_phase"]  = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Field phase']
    gdf["expert"] = expert_value

    if expert_value.empty:
        logger.error(f"Report Entry not found for {shp_name}")
        missing_report_entries.append(shp_name)
        gdf["expert"] = None
    gdf["species"] = "iguana"
    gdf["island_code"] = island_code
    gdf["site_code"] = get_prefix(shp_name.split('_')[1])


    output_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
    output_file = output_dir / island_code / shp_name
    output_file = output_file.with_suffix('.geojson')
    if output_file.exists():
        logger.info(f"File already exists: {output_file}")
    else:
        logger.info(f"Creating {output_file.name} in {island_code}, full path: {output_file}")

        gdf.to_file(output_file, driver='GeoJSON')

logger.info(f"Missing report entries: {missing_report_entries}")

with open(output_dir / 'missing_report_entries.txt', 'w') as f:
    for item in missing_report_entries:
        f.write(f"{item}\n")

with open(output_dir / 'missing_orthomosaics_entries.txt', 'w') as f:
    for item in missing_orthomosaics_entries:
        f.write(f"{item}\n")
