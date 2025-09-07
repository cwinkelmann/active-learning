"""
Main Code for the Image Quality Report of Drone Flights

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
from active_learning.util.rename import get_site_code

## copy the template and the nearby images to a new folder
import shutil
from pathlib import Path
import matplotlib.patches as mpatches

from active_learning.util.visualisation.drone_flights import visualise_drone_model, visualise_flights, \
    visualise_flights_speed_multiple, visualise_flights_speed_multiple_2, visualize_height_distribution, \
    visualize_height_boxplot, visualize_height_ridgeplot

dest_folder = Path("/Users/christian/data/Iguanas_From_Above/selected_images")

site_code_filter = 'FCD'
site_polygon = gpd.read_file("/Users/christian/data/Iguanas_From_Above/sites/Fernandina_Punta_Mangle.shp")

gdf_all = gpd.read_parquet('/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.parquet')
gdf_all['year'] = gdf_all['datetime_digitized'].dt.year


# figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/drone_fernandina_punta_mangle")
figure_path = Path("/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/drone_fernandina_cabo_douglas")
figure_path.mkdir(parents=True, exist_ok=True)
gdf_anomaly = find_anomalous_images(df=gdf_all)


# fcd_entries = gdf_all[gdf_all['site_code'] == site_code_filter]
# filter the entries by a polygon
fcd_entries = gpd.sjoin(gdf_all, site_polygon, how="inner")

# Define bins for shift_pixels
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,8.0,9.0, 10.0, float('inf')]
labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']

# Create a new column with the binned values
fcd_entries['shift_pixels_bin'] = pd.cut(fcd_entries['shift_pixels'], bins=bins, labels=labels)

fcd_entries[['image_name', 'exposure_time', 'photographic_sensitivity', 'speed_m_per_s', 'AbsoluteAltitude', 'RelativeAltitude', 'gsd_rel_avg_cm', 'shift_pixels']]

# Group by the binned column and count
fcd_entries['shift_pixels_bin'] = pd.cut(fcd_entries['shift_pixels'], bins=bins, labels=labels)
grouped_stats = fcd_entries.groupby('shift_pixels_bin').agg(
    count=('image_name', 'count'),
    sample_image=('image_name', 'first'),
    avg_shift=('shift_pixels', 'mean'),
    avg_risk=('risk_score', 'mean')
).reset_index()

fcd_entries.to_file('/Users/christian/data/Iguanas_From_Above/FCD_2020_2024_database.geojson', driver='GeoJSON')

# template_image_name="Fer_FCD03_DJI_0915_04052024.JPG"
# template_rows = fcd_entries[fcd_entries["image_name"] == template_image_name]

# template_rows.to_file('/Users/christian/data/Iguanas_From_Above/FCD_2020_2024_database_template.geojson', driver='GeoJSON')

# template_image = template_rows.iloc[0]

# random_image, nearby_different = get_near_images(fcd_entries, template_image=template_image, grouping_column="folder_name",
#                                                  difference="photographic_sensitivity", n=8, max_distance=15 )

#nearby_different.to_file("/Users/christian/data/Iguanas_From_Above/FCD_2020_2024_database_nearby.geojson", driver='GeoJSON')


def move_images_inspecteion(template_image, nearby_different):
    # Copy the template image
    template_path = Path(template_image["filepath"])  # Assuming the file path is stored in a column called "file_path"
    template_dest = dest_folder / template_path.name
    shutil.copy2(template_path, template_dest)
    print(f"Copied template image: {template_path.name}")
    # Copy the nearby images
    for idx, image in nearby_different.iterrows():
        image_path = Path(image["filepath"])  # Adjust column name if needed
        image_dest = dest_folder / image_path.name
        shutil.copy2(image_path, image_dest)
        print(f"Copied nearby image: {image_path.name}")

# move_images_inspecteion()

print(f"All images copied to {dest_folder}")


grouped = fcd_entries.groupby(['YYYYMMDD', 'flight_code', 'site_code'])
fig1, ax1 = visualise_drone_model(fcd_entries, figure_path=figure_path / 'drone_model_site.png')
fig2, ax2 = visualise_drone_model(gdf_all, aggregation="week", figure_path=figure_path / 'drone_model_globally_week.png')
plt.show()

# To get the count of entries in each group
group_counts = grouped.size().reset_index(name='count')


# To get additional statistics for each group
# Replace ['column1', 'column2'] with your columns of interest
group_stats = grouped.agg({
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
    'body_serial_number': lambda x: list(x.unique()),
    # Add more aggregations as needed
}).reset_index()


print(group_stats)


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

print(group_global_stats)

groups = []



# TODO refine these diagrams
for (date, site, site_code), group_data in grouped:
    print(f"Date: {date}, Site: {site}, Count: {len(group_data)}")

    gdf_group_data_metrics = get_flight_metrics(group_data)
    groups.append(gdf_group_data_metrics)

    # TODO visualise each group
    # visualise_flights(date, site, site_code, gdf_group_data_metrics, figure_path)

gdf_missions = pd.concat(groups, ignore_index=True)

# remove entries with speed > 10 m/s
gdf_missions = gdf_missions[gdf_missions['speed_m_per_s'] <= 6]
gdf_missions = gdf_missions[gdf_missions['speed_m_per_s'] > 1]
# get the speed between shots for each flight_code
fig, ax = visualise_flights_speed_multiple(gdf_missions, smoothing_window = 40, column='speed_m_per_s',
                                           title='Flight Speed Comparison', y_text = 'Flight Speed (m/s)')
fig.savefig( figure_path / 'speed_m_per_s.png')
plt.show()

fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 40, column='height', unit = 'm',
                                           title='Absolute Height Comparison', y_text = 'Flight Height (m)')
fig.savefig( figure_path / 'height.png')
plt.show()

fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 40, column='RelativeAltitude', unit = 'm',
                                           title='Relative Height Comparison', y_text = 'Relative Flight Height (m)')
fig.savefig( figure_path / 'RelativeAltitude.png')
plt.show()

fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 40, column='photographic_sensitivity', unit = '',
                                           title='Photographic Sensitivity', y_text = 'Photographic Sensitivity')
fig.savefig( figure_path / 'IsoComparison.png')
plt.show()


fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 10, column='exposure_time', unit = '',
                                           title='Exposure Time', y_text = 'Exposure Time')
fig.savefig( figure_path / 'ExposureTime.png')
plt.show()

# Risk Factor
fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 5, column='risk_score', unit = '',
                                           title='Risk Score', y_text = 'risk_score')
fig.savefig( figure_path / 'risk_score.png')
plt.show()

# Shift pixels
fig, ax = visualise_flights_speed_multiple_2(gdf_missions, smoothing_window = 5, column='shift_pixels', unit = '',
                                           title='Shift in pixels during Exposure', y_text = 'shift_pixels')
fig.savefig( figure_path / 'shift_pixels.png')
plt.show()


""" Now plot some rough overview charts """

## average height per season
# Visualize height distribution using histograms
# Height Distribution Histogram
fig1, ax1 = visualize_height_distribution(fcd_entries, title="Sample Drone Absolute Height Distribution by Year")
fig1.savefig(figure_path / 'height_distribution_histogram.png', bbox_inches='tight', dpi=300)

fig1, ax1 = visualize_height_distribution(fcd_entries, title="Sample Drone Relative Height Distribution by Year",
                                          height_col='RelativeAltitude')
fig1.savefig(figure_path / 'height_distribution_histogram.png', bbox_inches='tight', dpi=300)

# Speed Distribution Histogram
fig8, ax8 = visualize_height_distribution(fcd_entries, height_col="speed_m_per_s",
                                          title="Sample Drone Speed Distribution by Year",
                                          unit="m/s")
fig8.savefig(figure_path / 'speed_distribution_histogram.png', bbox_inches='tight', dpi=300)

# Height Boxplot
fig2, ax2 = visualize_height_boxplot(fcd_entries, title="Sample Drone Height by Year")
fig2.savefig(figure_path / 'height_boxplot.png', bbox_inches='tight', dpi=300)

# Height Ridge Plot
fig3, ax3 = visualize_height_ridgeplot(fcd_entries, title="Sample Drone Height Distribution by Year", unit="m")
fig3.savefig(figure_path / 'height_ridgeplot.png', bbox_inches='tight', dpi=300)

# Speed Ridge Plot
fig4, ax4 = visualize_height_ridgeplot(fcd_entries, height_col="speed_m_per_s", title="Sample Drone Speed Distribution by Year")
fig4.savefig(figure_path / 'speed_ridgeplot.png', bbox_inches='tight', dpi=300)

# ISO Ridge Plot
fig5, ax5 = visualize_height_ridgeplot(fcd_entries, height_col="photographic_sensitivity", title="Sample ISO Distribution by Year")
fig5.savefig(figure_path / 'iso_ridgeplot.png', bbox_inches='tight', dpi=300)

# ISO Boxplot
fig7, ax7 = visualize_height_boxplot(fcd_entries, height_col="photographic_sensitivity",
                                     title="Sample ISO by Year",
                                     unit="")
fig7.savefig(figure_path / 'iso_boxplot.png', bbox_inches='tight', dpi=300)

# ISO Boxplot
fig10, ax10 = visualize_height_boxplot(fcd_entries, height_col="shift_pixels",
                                       title="Sample Pixel Shift by Year",
                                       unit = "px")
fig10.savefig(figure_path / 'shift_pixels_bb.png', bbox_inches='tight', dpi=300)

# Exposure Time Ridge Plot
fig6, ax6 = visualize_height_ridgeplot(fcd_entries, height_col="exposure_time", title="Sample Exposure Time Distribution by Year")
fig6.savefig(figure_path / 'exposure_time_ridgeplot.png', bbox_inches='tight', dpi=300)


plt.show()


# TODO dispay "exposure_mode" as stacked bar chart or pie chart