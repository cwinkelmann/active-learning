"""
plot the training data to illustrate what is annotated and what is not

1. get the database from all images every taken and the hasty annotateed images. The latter are still in the old name schema

There are two types: the orthomosaics and the direct drone shots



"""


import pandas as pd
import geopandas as gpd
from pathlib import Path

from active_learning.database import images_data_extraction, derive_image_metadata
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

CRS_utm_zone_15 = "32715"
EPSG_WGS84 = "4326"

flight_database_path= Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")
flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=EPSG_WGS84)

full_hasty_annotation_file_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")
hasty_images_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_images")

gdf_hasty_image_metadata = images_data_extraction(hasty_images_path)

# TODO get the right projections
gdf_hasty_image_metadata.to_crs(epsg="32715", inplace=True)
gdf_hasty_images = derive_image_metadata(gdf_hasty_image_metadata)

gdf_hasty_images
# TODO get the mission name too
df_flight_database_new_names = flight_database[["image_hash", "image_name", "island", "site_code",
                                "datetime_digitized", "mission_folder"]].copy()
df_flight_database_new_names.rename(columns={"image_name": "new_name_schema",
                             "island":"island_new_name",
                             "site_code": "site_code_new_name",
                             "mission_folder": "mission_folder_new_name",
                             "datetime_digitized": "datetime_digitized_new_name"}, inplace=True)

gdf_hasty_images_merged = gdf_hasty_images.merge(df_flight_database_new_names,
                                                left_on="image_hash",
                                                right_on="image_hash",
                                                how="inner")

# filter the flight_database for the images that are in the hasty images
flight_database_filtered = flight_database[flight_database["image_hash"].isin(gdf_hasty_images_merged["image_hash"])]
flight_database_filtered.to_file("labelled_hasty_images.geojson", driver="GeoJSON")
# get the full mission
flight_database_full_missions_filtered = flight_database[flight_database["mission_folder"].isin(flight_database_filtered["mission_folder"])]

# create a polygon from each mission
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import LineString

mission_polygons = []
mission_lines = []

buffer_radius = 10  # meters

for mission_folder, group in flight_database_full_missions_filtered.groupby("mission_folder"):
    group = group.dropna(subset=["geometry"])
    group = group.to_crs(epsg=CRS_utm_zone_15) # TODO get the right project for each zone
    if group.empty:
        continue

    # Sort for LineString
    group_sorted = group.sort_values("timestamp") if "timestamp" in group.columns else group.sort_index()
    coords = group_sorted.geometry.apply(lambda p: (p.x, p.y)).tolist()

    if len(coords) >= 2:
        flight_path = LineString(coords)
        mission_lines.append({
            "mission_folder": mission_folder,
            "geometry": flight_path
        })

    # Create buffer union (tight polygon)
    buffered_points = group.geometry.buffer(buffer_radius).to_crs(flight_database_full_missions_filtered.crs)
    merged = unary_union(buffered_points)

    mission_polygons.append({
        "mission_folder": mission_folder,
        "geometry": merged
    })

gdf_mission_lines = gpd.GeoDataFrame(mission_lines, crs=flight_database_full_missions_filtered.crs)
gdf_mission_polygons = gpd.GeoDataFrame(mission_polygons, crs=flight_database_full_missions_filtered.crs)

gdf_mission_polygons.to_file("labelled_hasty_mission_polygons.geojson", driver="GeoJSON")

mission_names_filter = [
    # Floreana, clockwise order
    ["FLMO04_03022021", "FLMO05_03022021", "FLMO06_03022021"],
    ["FLMO01_02022021", "FLMO02_02022021", "FLMO03_02022021"],
    ["FLMO02_28012023"],
    ["FLBB01_28012023"], # they have an intersection with a annoated raster
    ["FLPC07_22012021"], # they have an intersection with a annoated raster
    ["FLPA03_21012021"],
    ["FLSCA02_23012021"],

    # Genovesa, clockwise order
    ["GES06_04122021",  "GES07_04122021"], # they have an intersection with a annoated raster
    ["GES13_05122021"], # they have an intersection with a annoated raster

    # Santiago
    ["STJB01_10012023"],

    # Fernandina
    ["FCD01_20122021", "FCD02_20122021", "FCD03_20122021"],
    ["FPE01_18122021"],
    ["FEA01_18122021"], # accidally assigned to floreana in hasty

]

# Get unique pairs/combinations of both columns
unique_combinations = gdf_hasty_images_merged[["mission_folder_new_name", "mission_folder"]].drop_duplicates()


# ==========

# TODO annotated hasty images
hA = HastyAnnotationV2.from_file(full_hasty_annotation_file_path)

# TODO rename the images in the hasty annotation
# TODO rename the images and folders

# TODO create a LaTeX table with metadata from the flights

raster_mask_dd_path = Path("/Volumes/2TB/SamplingIssues/RasterMask/DD/")
raster_mask_ms_path = Path("/Volumes/2TB/SamplingIssues/RasterMask/MS/")



annotations_path = Path("/Volumes/2TB/Manual_Counting/Geospatial_Annotations/")

# orthomosaic_shapefile_mapping_path = Path(
#     "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/enriched_GIS_progress_report_with_stats.csv")
# df_orthomosaic_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)
#
# df_orthomosaic_mapping