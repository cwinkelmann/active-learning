"""
Instead of look at each individual image, we want to look at the whole flight as a whole and inpect flight length, duration, average speed, max speed, frontlap, sidelap, etc. if it is a
nadir or oblique flight, etc.

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
from active_learning.util.mapping.helper import get_mission_flight_length, get_mission_type
from active_learning.util.rename import get_site_code

## copy the template and the nearby images to a new folder
import shutil
from pathlib import Path
import matplotlib.patches as mpatches

from active_learning.util.visualisation.drone_flights import visualise_drone_model, visualise_flights, \
    visualise_flights_speed_multiple, visualise_flights_speed_multiple_2, visualize_height_distribution, \
    visualize_height_boxplot, visualize_height_ridgeplot

base_path = Path("/Users/christian/data/Iguanas_From_Above/")
dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")

site_code_filter = 'FCD'
site_polygon = gpd.read_file("/Users/christian/data/Iguanas_From_Above/sites/Fernandina_Punta_Mangle.shp")

gdf_all = gpd.read_parquet('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.parquet')
gdf_all['year'] = gdf_all['datetime_digitized'].dt.year


# figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/drone_fernandina_punta_mangle")
figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/mission_metrics")
figure_path.mkdir(parents=True, exist_ok=True)

# fig1, ax1 = visualise_drone_model(gdf_all, figure_path=figure_path / 'drone_model_site.png')
# fig2, ax2 = visualise_drone_model(gdf_all, aggregation="week", figure_path=figure_path / 'drone_model_globally_week.png')
# plt.show()

# gdf_all = gdf_all[gdf_all.flight_code == "FLSF03"]
gdf_all = gdf_all[gdf_all.flight_code == "FSL01"]

grouped = gdf_all.groupby(['YYYYMMDD', 'flight_code', 'site_code'])

# To get the count of entries in each group
group_counts = grouped.size().reset_index(name='count')

grouped_global = gdf_all.groupby(['year', 'island'])

group_global_stats = grouped_global.agg({
    'exposure_time': 'mean',
    'photographic_sensitivity': 'mean',
    'AbsoluteAltitude': 'mean',
    'RelativeAltitude': 'mean',
    'exposure_program': lambda x: list(x.unique()),
    'f_number': 'mean',
    'gsd_rel_avg_cm': 'mean',
    'shift_pixels': 'mean',
    'risk_score': 'mean',
    'speed_m_per_s': 'mean',
    # Add more aggregations as needed
}).reset_index()

group_global_stats.to_csv(base_path / 'global_flight_stats_by_year_and_island.csv', index=False)

groups = []



# TODO refine these diagrams
for (date, site, site_code), group_data in grouped:
    logger.info(f"Date: {date}, Site: {site}, Count: {len(group_data)}")
    if len(group_data)  < 10:
        logger.warning(f"Skipping Date: {date}, Site: {site}, Count: {len(group_data)}")
        continue
    gdf_group_data_metrics = get_flight_metrics(group_data)

    mission_length = get_mission_flight_length(gdf_group_data_metrics)
    # TODO visualise each group
    get_mission_type(gdf_group_data_metrics)

    groups.append(gdf_group_data_metrics)

gdf_missions = pd.concat(groups, ignore_index=True)
grouped_missions = gdf_missions.groupby(['folder_name'])
df_grouped_missions = grouped_missions.agg({
    'exposure_time': 'mean',
    'photographic_sensitivity': 'mean',
    'AbsoluteAltitude': 'mean',
    'RelativeAltitude': 'mean',
    'exposure_program': lambda x: list(x.unique()),
    'f_number': 'mean',
    'gsd_rel_avg_cm': 'mean',
    'shift_pixels': 'mean',
    'risk_score': 'mean',
    'speed_m_per_s': 'mean',
    'is_oblique': 'mean',
    'is_nadir': 'mean',
    # Add more aggregations as needed
}).reset_index()

# for groubed in grouped_missions:
#     print(groubed)

df_grouped_missions.to_csv(base_path / 'mission_flight_stats.csv', index=False)
df_grouped_missions.to_json(base_path / 'mission_flight_stats.json', index=False, orient='records')

list(df_grouped_missions[df_grouped_missions["is_oblique"] < 0.5]['folder_name'])