"""
Instead of look at each individual image, we want to look at the whole flight as a whole and inpect flight length, duration, average speed, max speed, front overlap, (side overlap), etc. if it is a
nadir or oblique flight, etc.

"""

import geopandas as gpd
import pandas as pd
from loguru import logger
## copy the template and the nearby images to a new folder
from pathlib import Path

from active_learning.util.mapping.helper import get_mission_flight_length, get_flight_route_type

# base_path = Path("/Users/christian/data/Iguanas_From_Above/")
base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/")
# gdf_all = gpd.read_parquet(base_path / 'database/database_analysis_ready.parquet')
gdf_all = gpd.read_parquet(base_path / 'database_analysis_ready.parquet')


dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")

site_code_filter = 'FCD'
# site_polygon = gpd.read_file("/Users/christian/data/Iguanas_From_Above/sites/Fernandina_Punta_Mangle.shp")

gdf_all['year'] = gdf_all['datetime_digitized'].dt.year


# figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/drone_fernandina_punta_mangle")
figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/mission_metrics")
figure_path.mkdir(parents=True, exist_ok=True)


table_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/tables/mission_metrics")
table_path.mkdir(parents=True, exist_ok=True)

# fig1, ax1 = visualise_drone_model(gdf_all, figure_path=figure_path / 'drone_model_site.png')
# fig2, ax2 = visualise_drone_model(gdf_all, aggregation="week", figure_path=figure_path / 'drone_model_globally_week.png')
# plt.show()

gdf_all_

# gdf_all = gdf_all[gdf_all.flight_code == "FLSF03"]
# gdf_all = gdf_all[gdf_all.flight_code == "FSL01"]

grouped = gdf_all.groupby(['expedition_phase', 'YYYYMMDD', 'flight_code', 'site_code'])

# To get the count of entries in each group
group_counts = grouped.size().reset_index(name='count')

grouped_global = gdf_all.groupby(['island', 'expedition_phase', 'year'])

group_global_stats = grouped_global.agg(
    image_count=('exposure_time', 'count'),  # Count per group using any column
    avg_exposure_time=('exposure_time', 'mean'),
    avg_photographic_sensitivity=('photographic_sensitivity', 'mean'),
    avg_absolute_altitude=('AbsoluteAltitude', 'mean'),
    avg_relative_altitude=('RelativeAltitude', 'mean'),
    unique_exposure_programs=('exposure_program', lambda x: list(x.unique())),
    avg_f_number=('f_number', 'mean'),
    avg_gsd_rel_cm=('gsd_rel_avg_cm', 'mean'),
    avg_shift_pixels=('shift_pixels', 'mean'),
    avg_time_diff_seconds=('time_diff_seconds', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_speed_m_per_s=('speed_m_per_s', 'mean'),
    avg_forward_overlap_pct=('forward_overlap_pct', 'mean'),
    median_forward_overlap_pct=('forward_overlap_pct', 'median'),
).reset_index()

group_global_stats.to_csv(table_path / 'global_flight_stats_by_year_and_island.csv', index=False)
group_global_stats.to_latex(
    table_path / 'table_global_flight_stats_by_year_and_island.tex',
    index=False,
    caption='Global Flight Statistics by Year and Island',
    label='tab:global_flight_stats',
    position='htbp',
    column_format='|c|l|r|r|r|r|l|r|r|r|r|r|r|',  # 13 columns
    escape=False,
    float_format='%.2f'
)

groups = []






for (expedition_phase, date, site, site_code), group_data in grouped:
    logger.info(f"Date: {date}, Site: {site}, Count: {len(group_data)}")
    if len(group_data)  < 10:
        logger.warning(f"Skipping Expedition Phase {expedition_phase}, Date:  {date}, Site: {site}, Count: {len(group_data)}")
        continue

    gdf_group_data_metrics = group_data
    # gdf_group_data_metrics = get_flight_metrics(group_data)
    # 
    mission_length = get_mission_flight_length(gdf_group_data_metrics)
    gdf_group_data_metrics['mission_length_m'] = mission_length
    # # TODO visualise each group
    # get_mission_type(gdf_group_data_metrics)
    # gdf_group_data_metrics["survey_type"] = get_flight_route_type(gdf_group_data_metrics)

    groups.append(gdf_group_data_metrics)



# Analyse the single missions
gdf_missions = pd.concat(groups, ignore_index=True)
grouped_missions = gdf_missions.groupby(['expedition_phase', 'YYYYMMDD','flight_code', 'site_code', 'folder_name'])
df_grouped_missions = grouped_missions.agg(
    image_count=('exposure_time', 'count'),  # Count per group using any column
    flight_duration=('exposure_time', 'count'),  # Count per group using any column
    avg_exposure_time=('exposure_time', 'mean'),
    avg_time_diff_seconds=('time_diff_seconds', 'mean'),
    avg_photographic_sensitivity=('photographic_sensitivity', 'mean'),
    avg_absolute_altitude=('AbsoluteAltitude', 'mean'),
    avg_relative_altitude=('RelativeAltitude', 'mean'),
    unique_exposure_programs=('exposure_program', lambda x: list(x.unique())),
    avg_f_number=('f_number', 'mean'),
    avg_gsd_rel_cm=('gsd_rel_avg_cm', 'mean'),
    avg_shift_pixels=('shift_pixels', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_speed_m_per_s=('speed_m_per_s', 'mean'),
    avg_forward_overlap_pct=('forward_overlap_pct', 'mean'),
    median_forward_overlap_pct=('forward_overlap_pct', 'median'),
    mission_length_m=('mission_length_m', 'median'),
    survey_type=('bearing_to_prev', lambda group_series: get_flight_route_type(group_series.to_frame())),
    # Apply to entire group

).reset_index()

# for groubed in grouped_missions:
#     print(groubed)

df_grouped_missions.to_csv(base_path / 'mission_flight_stats.csv', index=False)
df_grouped_missions.to_json(base_path / 'mission_flight_stats.json', index=False, orient='records')
df_grouped_missions.to_latex(
    table_path / 'table_mission_flight_stats.tex',
    index=False,
    caption='Flight Statistics by Mission',
    label='tab:mission_flight_stats',
    position='htbp',
    column_format='|c|l|r|r|r|r|l|r|r|r|r|r|r|',  # 13 columns
    escape=False,
    float_format='%.2f'
)

list(df_grouped_missions[df_grouped_missions["is_oblique"] < 0.5]['folder_name'])


# TODO visualise the missions