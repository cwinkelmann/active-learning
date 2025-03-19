# """
#
# """
# import random
#
# import geopandas
# import geopandas as gpd
# import pandas as pd
# from loguru import logger
# from matplotlib import pyplot as plt
# from sklearn.cluster import DBSCAN
# import numpy as np
#
# from active_learning.config.mapping import get_island_code, drone_mapping
# from active_learning.util.drone_flight_check import get_analysis_ready_image_metadata, get_flight_metrics
# from active_learning.util.rename import get_site_code
#
# ## copy the template and the nearby images to a new folder
# import shutil
# from pathlib import Path
# import matplotlib.patches as mpatches
#
# # Define the destination folder where you want to copy the images
# # dest_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/selected_images")
# dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")
#
# # Create the destination folder if it doesn't exist
# dest_folder.mkdir(exist_ok=True, parents=True)
#
#
# gdf_all = gpd.read_parquet('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database.parquet')
# gdf_all.to_crs(epsg="32715", inplace=True)
#
# gdf_all = gdf_all[gdf_all["model"] != "MAVIC2-ENTERPRISE-ADVANCED"] # remove the thermal drone images
# gdf_all = get_analysis_ready_image_metadata(gdf_all)
#
# groups = []
# grouped = gdf_all.groupby(['YYYYMMDD', 'flight_code', 'site_code'])
# #   TODO refine these diagrams
# for (date, site, site_code), group_data in grouped:
#
#     gdf_group_data_metrics = get_flight_metrics(group_data)
#     groups.append(gdf_group_data_metrics)
# gdf_all = gpd.GeoDataFrame(pd.concat(groups, ignore_index=True))
#
#
# # TODO detect which of the flighs where manual flights and which were automatic
# # remove inf values and find out where they come from
# gdf_all = gdf_all.replace([np.inf, -np.inf], None)
#
# gdf_all['time_diff_seconds'] = pd.to_numeric(gdf_all['time_diff_seconds'], errors='coerce')
# gdf_all['distance_to_prev'] = pd.to_numeric(gdf_all['distance_to_prev'], errors='coerce')
# gdf_all['risk_score'] = pd.to_numeric(gdf_all['risk_score'], errors='coerce')
# gdf_all['shift_mm'] = pd.to_numeric(gdf_all['shift_mm'], errors='coerce')
# gdf_all['shift_pixels'] = pd.to_numeric(gdf_all['shift_pixels'], errors='coerce')
# gdf_all['shift_gsd_distance_frac'] = pd.to_numeric(gdf_all['shift_gsd_distance_frac'], errors='coerce')
#
# gdf_all.to_parquet('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.parquet')
# # gdf_all.to_file('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.shp', driver="ESRI Shapefile")
# gdf_all.to_file('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.gpkg', driver="GPKG")
