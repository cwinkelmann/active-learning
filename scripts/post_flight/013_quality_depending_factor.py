"""
Combine Image Quality Report data of Drone Flights with Imaage Quality Report data of Manual Flights
"""
import random

import geopandas
import geopandas as gpd
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

from active_learning.config.mapping import get_island_code, drone_mapping
from active_learning.util.drone_flight_check import get_near_images, find_anomalous_images, get_flight_metrics
from active_learning.util.image_quality.image_quality_from_grid import process_grid, aggregate_results
from active_learning.util.rename import get_site_code

## copy the template and the nearby images to a new folder
import shutil
from pathlib import Path


dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")

site_code_filter = 'FCD'
island_filter = 'Fernandina'
site_polygon = gpd.read_file("/Users/christian/data/Iguanas_From_Above/sites/Fernandina_Punta_Mangle.shp")

gdf_all = gpd.read_parquet('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.parquet')

fernandina_grid = Path("/Users/christian/PycharmProjects/hnee/active_learning/playground/meetup/image_quality_results_Fernandina/image_quality.parquet")
grid_size=(10, 10)
# Define metrics
metrics = {
    'laplacian_variance': None,
    'gradient_magnitude': None,
    'local_contrast': None,
    'tenengrad': None
}


gdf_all['year'] = gdf_all['datetime_digitized'].dt.year
gdf_fernandina = gdf_all[gdf_all['island'] == island_filter]

# Define bins for shift_pixels
bins = [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,8.0,9.0, 10.0, float('inf')]
labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']

# Create a new column with the binned values
gdf_fernandina['shift_pixels_bin'] = pd.cut(gdf_fernandina['shift_pixels'], bins=bins, labels=labels)

# Define bins for shift_pixels
bins = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 25.0,50.0, float('inf')]
labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-25', '25-50', '50+']

# Create a new column with the binned values
gdf_fernandina['risk_score_bin'] = pd.cut(gdf_fernandina['risk_score'], bins=bins, labels=labels)

df_metrics = process_grid(grid_file_path=fernandina_grid, grid_size=grid_size, metrics=metrics)

gdf_all_quality = gdf_fernandina.merge(df_metrics, left_on='image_hash', right_on='image_id', how='left')


# TODO calculate the front_overlap

grouping_field = 'year'
grouping_field = 'body_serial_number'
grouping_field = 'drone_name'
# grouping_field = 'risk_score_bin'
# grouping_field = 'shift_pixels_bin'
for group_id, gdf_group_data in gdf_all_quality.groupby(grouping_field):
    # gdf_group_data = gdf_group_data[gdf_group_data['site_code'] == site_code_filter]

    aggregated_results = aggregate_results(gdf_group_data, grid_size=grid_size, metrics=metrics)

    lc_mean= aggregated_results["local_contrast"].mean()
    lpv_mean= aggregated_results["laplacian_variance"].mean()

    print(f"local_contrast mean: {grouping_field}: {group_id} with {len(gdf_group_data)} samples: local_contrast: {lc_mean:.2f} laplacian_variance: {lpv_mean:.2f}")

