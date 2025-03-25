"""
The shapefiles are not very well organised, named or encoded
This script will reorganise the shapefiles into a more structured form, correct the CRS and convert them to geojson
"""

import geopandas as gpd
import pandas as pd
import typing
from loguru import logger
from pathlib import Path

from active_learning.util.projection import project_gdfcrs
from active_learning.util.rename import get_site_code

shapefile_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Counts shp')

shp_files = [f for f in shapefile_base_path.glob('**/*counts.shp') if not f.name.startswith('.')]

# copy images to new folder
output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations")
orthomosaic_paths = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL")
progress_report_file = Path(
    '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')

df_progress_report = pd.read_csv(progress_report_file)

missing_report_entries: typing.List[str] = []
missing_orthomosaics_entries: typing.List[str] = []

shapefile_orthomosaic_mapping = []

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
    else:
        logger.info(f"Orthomosaic found: {images_path}")

    if possible_orthomosaic_name == "Esp_EM04_13012021":
        print("debug")

    gdf = project_gdfcrs(gdf, images_path)
    expert_value = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Expert']

    island = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Island']

    field_phase = df_progress_report.loc[
        df_progress_report['Orthophoto/Panorama name'] == possible_orthomosaic_name, 'Field phase']

    gdf["expert"] = expert_value
    gdf["field_phase"] = field_phase
    gdf["island"] = island

    if expert_value.empty:
        logger.error(f"Report Entry not found for {shp_name}")
        missing_report_entries.append(shp_name)
        gdf["expert"] = None
    gdf["species"] = "iguana"
    gdf["island_code"] = island_code
    gdf["site_code"] = get_site_code(shp_name.split('_')[1])

    output_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
    output_file = output_dir / island_code / shp_name
    output_file = output_file.with_suffix('.geojson')

    logger.info(f"Creating {output_file.name} in {island_code}, full path: {output_file}")

    gdf.to_file(output_file, driver='GeoJSON')

    shapefile_orthomosaic_mapping.append(
        {"images_path": images_path, "shp_file_path":
            shp_file_path, "shp_name": shp_name,
         "geojson_path": output_file,
         "expert": expert_value,
         "island_code": island_code,
         "island": island,

         }
    )

logger.info(f"Missing report entries: {missing_report_entries}")

pd.DataFrame(shapefile_orthomosaic_mapping).to_csv(output_dir / 'shapefile_orthomosaic_mapping.csv')

with open(output_dir / 'missing_report_entries.txt', 'w') as f:
    for item in missing_report_entries:
        f.write(f"{item}\n")

with open(output_dir / 'missing_orthomosaics_entries.txt', 'w') as f:
    for item in missing_orthomosaics_entries:
        f.write(f"{item}\n")
