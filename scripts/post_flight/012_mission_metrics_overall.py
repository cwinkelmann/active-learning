"""
Instead of look at each individual image, we want to look at the whole flight as a whole and inpect flight length, duration, average speed, max speed, front overlap, (side overlap), etc. if it is a
nadir or oblique flight, etc.

"""

import geopandas as gpd
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
## copy the template and the nearby images to a new folder
from pathlib import Path
import numpy as np

from active_learning.util.mapping.helper import get_mission_flight_length, get_flight_route_type
from active_learning.util.visualisation.drone_flights import visualise_drone_model

# base_path = Path("/Users/christian/data/Iguanas_From_Above/")
database_base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/")
# gdf_all = gpd.read_parquet(base_path / 'database/database_analysis_ready.parquet')
database_path =  database_base_path / 'database_analysis_ready.parquet'
# /Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/database_analysis_ready.parquet

gdf_all = gpd.read_parquet(database_path)

dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")

### DEBUGGING FILTER ###
#site_code_filter = 'FWK'
#gdf_all = gdf_all[gdf_all['site_code'] == site_code_filter].copy()
# site_polygon = gpd.read_file("/Users/christian/data/Iguanas_From_Above/sites/Fernandina_Punta_Mangle.shp")

gdf_all['year'] = gdf_all['datetime_digitized'].dt.year


# figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/drone_fernandina_punta_mangle")
figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/mission_metrics")
figure_path.mkdir(parents=True, exist_ok=True)


table_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/tables/mission_metrics_tmp")
table_path.mkdir(parents=True, exist_ok=True)

# per expedition
expedition_counts = gdf_all.groupby('expedition_phase').size().reset_index(name='image_count')
expedition_counts['percentage'] = (expedition_counts['image_count'] / len(gdf_all) * 100).round(2)
expedition_counts = expedition_counts.sort_values('image_count', ascending=False)
expedition_counts.rename(columns={'expedition_phase': 'Expedition Phase', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)

logger.info("\nImages per Expedition:")
logger.info(expedition_counts)


exposure_mode_counts = gdf_all.groupby('exposure_mode').size().reset_index(name='image_count')
exposure_mode_counts['percentage'] = (exposure_mode_counts['image_count'] / len(gdf_all) * 100).round(2)
exposure_mode_counts = exposure_mode_counts.sort_values('image_count', ascending=False)
exposure_mode_counts.rename(columns={'exposure_mode': 'Exposure Mode', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)

logger.info("\nImages per Exposure Mode:")
logger.info(exposure_mode_counts)

# per island
island_counts = gdf_all.groupby('island').size().reset_index(name='image_count')
island_counts['percentage'] = (island_counts['image_count'] / len(gdf_all) * 100).round(2)
island_counts = island_counts.sort_values('image_count', ascending=False)
island_counts.rename(columns={'island': 'Island', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)


logger.info("\nImages per Island:")
logger.info(island_counts)

# exposure_program
exposure_program_counts = gdf_all.groupby('exposure_program').size().reset_index(name='image_count')
exposure_program_counts['percentage'] = (exposure_program_counts['image_count'] / len(gdf_all) * 100).round(2)
exposure_program_counts = exposure_program_counts.sort_values('image_count', ascending=False)
exposure_program_counts.rename(columns={'exposure_program': 'Exposure Program', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)

logger.info("\nImages per Exposure Program:")
logger.info(exposure_program_counts)

# camera body (serial number)
camera_counts = gdf_all.groupby('drone_name').agg({
    'datetime_digitized': 'min',  # First usage date
    'image_name': 'count'         # Image count
}).reset_index()

# Rename columns for clarity
camera_counts.columns = ['drone_name', 'first_usage', 'image_count']

# Optional: Format the date nicely
camera_counts['first_usage'] = camera_counts['first_usage'].dt.strftime('%Y-%m-%d')
camera_counts['percentage'] = (camera_counts['image_count'] / len(gdf_all) * 100).round(2)
camera_counts = camera_counts.sort_values('image_count', ascending=False)
camera_counts.rename(columns={'drone_name': 'Drone Name', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)

logger.info("\nImages per Camera Body:")
logger.info(camera_counts)


# NADIR
nadir_counts = gdf_all.groupby('is_nadir').size().reset_index(name='image_count')
nadir_counts['percentage'] = (nadir_counts['image_count'] / len(gdf_all) * 100).round(2)
nadir_counts.sort_values('image_count', ascending=False, inplace=True)
nadir_counts.rename(columns={'is_nadir': 'perspective', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)
nadir_counts.replace({'perspective': {True: 'Nadir', False: 'Oblique'}}, inplace=True)


# REMOVE all oblique images from the analysis
gdf_all = gdf_all[gdf_all.is_nadir]

# route_type: Corridor vs Area
grouped = gdf_all.groupby(['expedition_phase', 'YYYYMMDD', 'flight_code', 'site_code', 'mission_folder', 'island'])
groups = []
for (expedition_phase, date, flight_code, site_code, mission_folder, island), group_data in grouped:
    logger.info(f"Date: {date}, Site: {site_code}, Count: {len(group_data)}")
    if len(group_data)  < 10:
        logger.warning(f"Skipping Expedition Phase {expedition_phase}, Date:  {date}, Site: {site_code}, Count: {len(group_data)}")
        continue

    # if mission_folder == "FLCC02_29012023":
    #     pass

    # TODO create hull out of the points
    # TODO
    group_summary_data = {}
    group_summary_data['expedition_phase'] = expedition_phase
    group_summary_data['YYYYMMDD'] = date
    group_summary_data['flight_code'] = flight_code
    group_summary_data['flight_code'] = flight_code
    group_summary_data['site_code'] = site_code
    group_summary_data['mission_folder'] = mission_folder
    group_summary_data['image_count'] = len(group_data)
    group_summary_data['island'] = island

    gdf_group_data_metrics = group_data
    # gdf_group_data_metrics = get_flight_metrics(group_data)
    #
    geometry, mission_length = get_mission_flight_length(gdf_group_data_metrics)

    group_summary_data['mission_length_m'] = mission_length
    if mission_length > 10000:
        logger.warning(f"Something is wrong with the mission length {mission_length} m for Expedition Phase {expedition_phase}, Date:  {date}, Site: {site_code}, Count: {len(group_data)}. The distance between first and last shot cannot be more than 10 km")
        logger.warning(f"This happened with three images in STESS01_09012023 ( Snt_STESS01_DJI_0609_09012023.JPG ), SCVE02_13012023 ( Scruz_SCVE02_DJI_0527_13012023.JPG )  and FSQ01_23012023 ( Fer_FSQ01_DJI_0705_23012023.JPG )")
    group_summary_data['geometry'] = geometry

    if mission_folder == "FSP02_23012023":
        pass

    # get_mission_type(gdf_group_data_metrics)
    group_summary_data["drone_model"] = gdf_group_data_metrics['drone_name'].mode()[0]
    group_summary_data["survey_type"] = get_flight_route_type(gdf_group_data_metrics)
    group_summary_data["nadir_frac"] = gdf_group_data_metrics['is_nadir'].mean()
    group_summary_data["oblique_frac"] = gdf_group_data_metrics['is_oblique'].mean()

    groups.append(group_summary_data)

df_missions = pd.DataFrame(groups)
gdf_missions = gpd.GeoDataFrame(df_missions, geometry='geometry', crs=gdf_all.crs)


mission_aggregations = grouped.agg(
    image_count=('exposure_time', 'count'),  # Count per group using any column
    avg_exposure_time=('exposure_time', 'mean'),
    avg_photographic_sensitivity=('photographic_sensitivity', 'mean'),
    avg_absolute_altitude=('AbsoluteAltitude', 'mean'),
    avg_relative_altitude=('RelativeAltitude', 'mean'),
    avg_f_number=('f_number', 'mean'),
    # avg_pitch_degree=('pitch_degree', 'mean'),
     avg_gsd_rel_cm=('gsd_rel_avg_cm', 'mean'),
    avg_shift_pixels=('shift_pixels', 'mean'),
    unique_exposure_programs=('exposure_program', lambda x: list(x.unique())),
    avg_time_diff_seconds=('time_diff_seconds', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_speed_m_per_s=('speed_m_per_s', 'mean'),
    avg_forward_overlap_pct=('forward_overlap_pct', 'mean'),
    median_forward_overlap_pct=('forward_overlap_pct', 'median'),
).reset_index()

gdf_missions = gdf_missions.merge(mission_aggregations, on='mission_folder', how='left',suffixes=('', '_dup'))
gdf_missions = gdf_missions.drop(columns=[col for col in gdf_missions.columns if col.endswith('_dup')])

gdf_missions.to_file(database_base_path / 'database_analysis_ready.gpkg',
                  layer='iguana_missions', overwrite=True,
                     driver="GPKG")

gdf_missions_latex = gdf_missions.drop(columns=['geometry', 'oblique_frac', 'mission_folder', 'expedition_phase', 'unique_exposure_programs'])

# gdf_missions_latex_isa = gdf_missions_latex[gdf_missions_latex['island'] == 'Fernandina']
# gdf_missions_latex_isa = gdf_missions_latex[gdf_missions_latex['island'] == 'Floreana']



def _flatten_cell(x):
    """Turn ['A'] -> 'A', ['A','B'] -> 'A,B', keep scalars as-is, None stays None."""
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x[0]
        return ",".join(map(str, x))
    return x

def format_gdf_for_latex(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Clean & format a (Geo)DataFrame for LaTeX export:
    - round/format numeric columns
    - flatten list-like/categorical cells
    - cast ISO to int
    - ensure date is YYYY-MM-DD string
    """
    df = gdf.copy()

    # --- Column names you likely have (adjust if yours differ) ---
    # If a column is missing, it's skipped safely.
    round_1 = ['avg_f_number', 'avg_time_diff_seconds', 'avg_overlap_pct', 'median_overlap_pct']                               # e.g., 2.8
    round_2 = ['avg_risk_score', 'avg_relative_altitude', 'avg_absolute_altitude', 'avg_shift_pixels']         # e.g., 27.32
    round_3 = ['nadir_frac', 'oblique_frac', 'avg_gsd_cm_per_px', 'avg_shift_px']  # e.g., 0.579
    round_4 = ['avg_exposure_time', 'avg_gsd_rel_cm']                         # e.g., 0.0005

    round_2_misc = ['avg_risk', 'avg_speed_m_per_s', 'avg_time_diff_s']  # e.g., 3.08, 1.56

    int_like = ['avg_forward_overlap_pct','median_forward_overlap_pct', 'images', 'avg_iso', 'mission_length_m', 'avg_photographic_sensitivity']                     # keep as integers if present

    # A column that contained arrays in your sample:
    flatten_like = ['unique_exposure_programs', 'mode', 'shooting_mode']  # whichever exists will be flattened

    date_cols = ['date']  # ensure YYYY-MM-DD

    # --- Apply conversions safely if columns exist ---
    for c in flatten_like:
        if c in df.columns:
            df[c] = df[c].map(_flatten_cell)

    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(0).astype('Int64')  # nullable int

    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce').dt.strftime('%Y-%m-%d')

    def round_if_present(cols, ndigits):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').round(ndigits)

    round_if_present(round_1, 1)
    round_if_present(round_2, 2)
    round_if_present(round_3, 3)
    round_if_present(round_4, 4)
    round_if_present(round_2_misc, 2)
    
    return df

# gdf_missions_latex_isa = format_gdf_for_latex(gdf_missions_latex_isa)


# gdf_missions_latex_isa.to_latex(
#     table_path / 'table_mission_flight_stats_floreana.tex',
#     index=False)
#
# gdf_missions_latex_isa.to_csv(
#     table_path / 'table_mission_flight_stats_floreana.csv',
#     index=False)

nadir_counts = gdf_all.groupby('is_nadir').size().reset_index(name='image_count')
nadir_counts['percentage'] = (nadir_counts['image_count'] / len(gdf_all) * 100).round(2)
nadir_counts.sort_values('image_count', ascending=False, inplace=True)
nadir_counts.rename(columns={'is_nadir': 'perspective', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)
nadir_counts.replace({'perspective': {True: 'Nadir', False: 'Oblique'}}, inplace=True)

survey_type_counts = gdf_missions.groupby('survey_type').size().reset_index(name='image_count')
survey_type_counts['percentage'] = (survey_type_counts['image_count'] / len(gdf_missions) * 100).round(2)
survey_type_counts.sort_values('image_count', ascending=False, inplace=True)
survey_type_counts.rename(columns={'survey_type': 'Survey Type', 'image_count': 'Image Count', 'percentage': '\%'}, inplace=True)
survey_type_counts.replace({'perspective': {True: 'Nadir', False: 'Oblique'}}, inplace=True)


island_aggregations_grouped = gdf_all.groupby(['expedition_phase','island'])

island_metrics = island_aggregations_grouped.agg(
    image_count=('exposure_time', 'count'),  # Count per group using any column
    avg_exposure_time=('exposure_time', 'mean'),
    avg_photographic_sensitivity=('photographic_sensitivity', 'mean'),
    avg_absolute_altitude=('AbsoluteAltitude', 'mean'),
    avg_relative_altitude=('RelativeAltitude', 'mean'),
    avg_f_number=('f_number', 'mean'),
    avg_gsd_rel_cm=('gsd_rel_avg_cm', 'mean'),
    avg_shift_pixels=('shift_pixels', 'mean'),
    unique_exposure_programs=('exposure_program', lambda x: list(x.unique())),
    avg_time_diff_seconds=('time_diff_seconds', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_speed_m_per_s=('speed_m_per_s', 'mean'),
    avg_forward_overlap_pct=('forward_overlap_pct', 'mean'),
    median_forward_overlap_pct=('forward_overlap_pct', 'median'),
).reset_index()

island_metrics = format_gdf_for_latex(island_metrics)


# Save all tables to CSV and LaTeX
tables = {
    'images_by_expedition': expedition_counts,
    'images_by_island': island_counts,
    'metrics_by_island': island_metrics,
    'images_by_exposure_program': exposure_program_counts,
    'images_by_camera_body': camera_counts,
    'images_by_exposure_mode': exposure_mode_counts,
    'images_by_perspective': nadir_counts,
    'missions_by_route_type': survey_type_counts,
}

for table_name, df in tables.items():
    # Save to CSV

    df = format_gdf_for_latex(df)

    df.to_csv(table_path / f"{table_name}.csv", index=False)

    # Save to LaTeX
    latex_table = df.to_latex(
        table_path / f"{table_name}.tex",
        index=False,
        # float_format="%.1f",
        escape=False)


logger.info(f"\nTables saved to: {table_path}")
logger.info("Files created:")
for table_name in tables.keys():
    logger.info(f"  - {table_name}.csv")
    logger.info(f"  - {table_name}.tex")




# group_columns = ['expedition_phase']
# group_columns = ['island', 'expedition_phase', 'year']
group_columns = ['island', 'expedition_phase']
grouped_global = gdf_all.groupby(group_columns)
# grouped_global = gdf_all.groupby(['island', 'expedition_phase', 'year'])

group_global_stats = grouped_global.agg(
    image_count=('exposure_time', 'count'),  # Count per group using any column
    avg_exposure_time=('exposure_time', 'mean'),
    avg_photographic_sensitivity=('photographic_sensitivity', 'mean'),
    avg_absolute_altitude=('AbsoluteAltitude', 'mean'),
    avg_relative_altitude=('RelativeAltitude', 'mean'),
    avg_f_number=('f_number', 'mean'),
    avg_gsd_rel_cm=('gsd_rel_avg_cm', 'mean'),
    avg_shift_pixels=('shift_pixels', 'mean'),
    avg_time_diff_seconds=('time_diff_seconds', 'mean'),
    avg_risk_score=('risk_score', 'mean'),
    avg_speed_m_per_s=('speed_m_per_s', 'mean'),
    avg_forward_overlap_pct=('forward_overlap_pct', 'mean'),
    median_forward_overlap_pct=('forward_overlap_pct', 'median'),
).reset_index()

group_global_stats = format_gdf_for_latex(group_global_stats)

group_global_stats.to_csv(table_path / 'global_flight_stats_by_island_expedition.csv', index=False)
group_global_stats.to_latex(
    table_path / 'global_flight_stats_by_island_expedition.tex',
    index=False,
    caption='Global Flight Statistics by Year and Island',
    label='tab:global_flight_stats',
    position='htbp',
    column_format='|c|l|r|r|r|r|l|r|r|r|r|r|r|',  # 13 columns
    escape=False,

)

fig1, ax1 = visualise_drone_model(gdf_all, figure_path=figure_path / 'drone_model_site.png',
                                  title="Drone Model Usage by day of Expedition",
                                  )

fig2, ax2 = visualise_drone_model(gdf_all, aggregation="month",
                                    title="Drone Model Usage by Month",
                                    legend_title="Drone Model",
                                  figure_path=figure_path / 'drone_model_globally_month.png')

fig3, ax3 = visualise_drone_model(gdf_all,
                                  title ="Drone Model Usage by Expedition Phase",
                                  legend_title="Drone Model",
                                  aggregation="expedition_phase",
                                  figure_path=figure_path / 'drone_model_globally_expedition.png')

fig4, ax4 = visualise_drone_model(gdf_all,
                                  aggregation="expedition_phase",
                                  title="Exposure Program by Expedition Phase",
                                  legend_title="Exposure Program",
                                  group_col_metric="exposure_program",
                                  figure_path=figure_path / 'drone_model_globally_exposure_program.png')
plt.show()
