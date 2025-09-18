"""
Convert the geoparquet from 010_image_db.py for the basic metadata extraction
"""

import geopandas as gpd
from loguru import logger

## copy the template and the nearby images to a new folder
from pathlib import Path

database_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024')
database_path = database_base_path / 'database_analysis_ready.parquet'

gdf_all = gpd.read_parquet(database_path)

# gdf_all = gdf_all.dropna(subset=['image_width'])
gdf_clean = gdf_all[gdf_all.geometry.notna() & gdf_all.geometry.is_valid].copy()

logger.info(f"Write cleaned database with {gdf_clean.shape[0]} records to various formats")
logger.info(f"to shapefile")
gdf_clean.to_file(database_base_path / 'database_analysis_ready.shp', driver="ESRI Shapefile")
# logger.info(f"to geojson")
# gdf_clean.to_file(Path('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.geojson'),
#                                  driver="GeoJSON")
logger.info(f"to geopackage")
gdf_clean.to_file(database_base_path / 'database_analysis_ready.gpkg',
                  layer='iguana_survey_data', driver="GPKG")
# gdf_clean.to_csv('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.csv')
logger.info(f"write completed")