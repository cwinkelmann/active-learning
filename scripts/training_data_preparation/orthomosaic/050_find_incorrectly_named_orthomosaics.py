"""



As an example
The Drone Deploy Orthomosaic Flo_FLPO01_28012023.tif does not match the image in the folder FLPO01_28012023



"""


import geopandas as gpd

import geopandas as gpd
import os
import pandas as pd
from pathlib import Path


def load_all_geojsons(folder_path):
    """
    Load all geojson files from a folder into a single GeoDataFrame.
    Each polygon is projected to WGS84 (EPSG:4326).

    Parameters:
    folder_path (str): Path to the folder containing geojson files

    Returns:
    geopandas.GeoDataFrame: Combined GeoDataFrame with all polygons
    """
    # List all geojson files in the folder
    geojson_files = list(Path(folder_path).glob("*.geojson"))

    # Create an empty list to store the GeoDataFrames
    gdf_list = []

    # Load each geojson file, project to WGS84, and append to the list
    for file_path in geojson_files:
        try:
            # Read the file
            gdf = gpd.read_file(file_path)

            # Store the filename as a column
            gdf['source_file'] = file_path.name

            # Project to WGS84 if not already in that CRS
            if gdf.crs is not None and gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            elif gdf.crs is None:
                # If CRS is not defined, assume it's in WGS84
                gdf.crs = "EPSG:4326"

            # Append to the list
            gdf_list.append(gdf)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Combine all GeoDataFrames into one
    if gdf_list:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        # Ensure the CRS is set
        combined_gdf.crs = "EPSG:4326"
        return combined_gdf
    else:
        # Return an empty GeoDataFrame if no files were loaded
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# # TODO load the Raster masks of all orthomosaics
# DD_folder_path = "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/RasterMask/DD"
# DD_all_polygons = load_all_geojsons(DD_folder_path)
#
# # Display info about the loaded polygons
# print(f"Loaded {len(DD_all_polygons)} polygons from {DD_folder_path}")
# print(f"CRS: {DD_all_polygons.crs}")
# print(DD_all_polygons.head())
#
# MS_folder_path = "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/RasterMask/MS"
# MS_all_polygons = load_all_geojsons(MS_folder_path)
#
# # Display info about the loaded polygons
# print(f"Loaded {len(MS_all_polygons)} polygons from {MS_folder_path}")
# print(f"CRS: {MS_all_polygons.crs}")
# print(MS_all_polygons.head())


# TODO Load the database of all images
all_images_database = gpd.read_parquet('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet')


# Just for fun simply load
gdf_flp001 = gpd.read_file('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/RasterMask/DD/raster_mask_Flo_FLPO01_28012023.geojson')
polygon_geometry = gdf_flp001.geometry.iloc[0]
print(f"Loaded {len(gdf_flp001)} polygons")
# find all images which are withing gdf_flp001
gdf_flp001 = gdf_flp001.to_crs("EPSG:4326")
images_gdf = all_images_database.to_crs("EPSG:4326")



# TODO do that later for the floreana FMO1 - FMO5 image which seem to have dissapeared
images_within_polygon = images_gdf[images_gdf.geometry.within(polygon_geometry)]


