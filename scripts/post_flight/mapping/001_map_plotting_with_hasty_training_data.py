"""
Creates an overall Galápagos Islands map add the hasty  training data of hasty origin
"""
from pathlib import Path

import geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box
import pyproj
import numpy as np
import pandas as pd
from loguru import logger

from active_learning.database import images_data_extraction, derive_image_metadata
from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks, find_closest_island

web_mercator_projection_epsg = 3857

def create_galapagos_expedition_map(
        gdf_flight_database: geopandas.GeoDataFrame,
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        output_path="galapagos_hasty_training_data_map.png",
        dpi=300):
    """
    Creates a map of the Galápagos Islands showing expedition data locations with different colors for each phase.
    """

    # Load GeoPackage layers
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')

    # Load flight database and convert to WGS84 first, then to Web Mercator
    flight_database = gdf_flight_database.to_crs(epsg=4326)

    # Add expedition phase mapping
    flight_database['year_month'] = flight_database['datetime_digitized'].dt.strftime('%Y_%m')

    expedition_mapping = {
        "2020_01": 1,
        "2021_01": 2,
        "2021_02": 2,
        "2021_12": 3,
        "2023_01": 4,
        "2023_02": 4,
        "2024_01": 5,  # from the El Niño folder
        "2024_04": 6,  # from the El Niño folder
        "2024_05": 6,  # from the El Niño folder
    }

    # Map the expedition phases
    flight_database['expedition_phase'] = flight_database['year_month'].map(expedition_mapping)

    # Remove unmapped data points
    flight_database = flight_database[flight_database['expedition_phase'].notna()]

    # Convert to Web Mercator for plotting
    flight_database_wm = flight_database.to_crs(epsg=web_mercator_projection_epsg)

    # Prepare base plot
    fig, ax = plt.subplots(figsize=(14, 12))
    islands_wm = islands.to_crs(epsg=web_mercator_projection_epsg)
    islands_wm = islands_wm[islands_wm['tipo'] == 'Isla']

    # Filter for important islands to plot nicely
    important_islands = ["Santiago", "Santa Fé", "Española", "Isabela",
                         "Fernandina", "Floreana", "Santa Cruz", "San Cristóbal",
                         "Genovesa", "San Cristobal", "Marchena", "Pinta"]



    islands_wm_f = islands_wm[islands_wm['nombre'].isin(important_islands)]

    # Determine name column
    name_col = 'nombre'
    islands_wm_f = islands_wm_f.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

    # Plot all islands
    islands_wm.plot(ax=ax, alpha=0.7, edgecolor='black', color='lightgrey')

    # Define colors for each expedition phase
    expedition_colors = {
        1: '#E63946',  # Red - 2020
        2: '#F77F00',  # Orange - 2021 (Jan/Feb)
        3: '#FCBF49',  # Yellow - 2021 (Dec)
        4: '#277DA1',  # Blue - 2023 (Jan/Feb)
        5: '#4D908E',  # Teal - 2024 (Jan)
        6: '#90E0EF',  # Light Blue - 2024 (Apr/May)
    }

    expedition_labels = {
        1: "Phase 1 (Jan 2020)",
        2: "Phase 2 (Jan-Feb 2021)",
        3: "Phase 3 (Dec 2021)",
        4: "Phase 4 (Jan-Feb 2023)",
        5: "Phase 5 (Jan 2024)",
        6: "Phase 6 (Apr-May 2024)",
    }

    # Plot expedition points by phase
    for phase in sorted(expedition_colors.keys()):
        phase_data = flight_database_wm[flight_database_wm['expedition_phase'] == phase]
        if not phase_data.empty:
            phase_data.plot(ax=ax,
                          marker='o',
                          color=expedition_colors[phase],
                          markersize=125,
                          alpha=0.7,
                          label=f"{expedition_labels[phase]} ({len(phase_data)} photos)")

    # Dictionary of label offsets (dx, dy) in map units
    label_offsets = {
        "Santiago": (30000, 10000),
        "Wolf": (1000, 5000),
        "Darwin": (1000, 5000),
        "Santa Fé": (0, -9000),
        "Española": (10000, 9000),
        "Isabela": (-10000, 0),
        "Fernandina": (-15000, -21000),
        "Floreana": (0, -14000),
        "Santa Cruz": (35000, 10000),
        "San Cristóbal": (70000, -10000),
        "Genovesa": (10000, 10000),
        "San Cristobal": (10000, 23000),
        "Marchena": (10000, 10000),
        "Pinta": (10000, 10000),
    }

    # Place island labels with manual offsets
    for idx, row in islands_wm_f.iterrows():
        # Get the largest polygon for this island
        island_poly = get_largest_polygon(row.geometry)
        if island_poly is None:
            continue

        # Get centroid of island
        centroid = island_poly.centroid

        # Get island name
        island_name = row['nombre']

        # Get offset from dictionary (or default if not defined)
        offset = label_offsets.get(island_name, (0, 0))

        # Apply offset to centroid
        label_x = centroid.x + offset[0]
        label_y = centroid.y + offset[1]

        # Add the label with white background for better visibility
        ax.text(label_x, label_y, island_name,
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))

    # Title
    ax.set_title('Galápagos Islands - Marine Iguana Single Image Training Data ', fontsize=16, pad=20)

    # Load countries for inset map
    countries = gpd.read_file(gpkg_path, layer='ne_10m_admin_0_countries')
    south_america = countries[countries['CONTINENT'].isin(['South America', 'North America'])]
    area_of_interest = gpd.read_file(gpkg_path, layer='broader_area')
    bounds = area_of_interest.geometry.union_all().bounds

    # Create a box from the bounds
    box_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
    south_america = south_america.clip(box_polygon)
    south_america = south_america.to_crs(epsg=web_mercator_projection_epsg)
    ecuador = south_america[south_america['ADMIN'] == 'Ecuador']

    # Add information text box
    info_text = """Data source: Marine Iguana Survey Data (2020-2024)
            Inset map: Made with Natural Earth
            Map projection: Web Mercator (EPSG:3857)
            Marine Iguana population assessment in the Galápagos Archipelago
            HNEE/Winkelmann, 2025"""

    add_text_box(
        ax,
        info_text,
        # location='upper left',
        fontsize=9,
        alpha=0.8,
        pad=0.5,
        # xy=(0.55, 0.84),  # Custom coordinates in axes fraction
        xy=(0.05, 0.98),  # Custom coordinates in axes fraction
    )

    # Set up coordinate grid
    x_ticks_proj, x_ticks_geo, y_ticks_proj, y_ticks_geo = get_geographic_ticks(ax,
                                                                                epsg_from=web_mercator_projection_epsg,
                                                                                epsg_to=4326)

    # Set the tick positions to the projected coordinates
    ax.set_xticks(x_ticks_proj)
    ax.set_yticks(y_ticks_proj)

    # Format the tick labels with the geographic coordinates
    ax.set_xticklabels([format_lat_lon(x, 0, is_latitude=False) for x in x_ticks_geo])
    ax.set_yticklabels([format_lat_lon(y, 0, is_latitude=True) for y in y_ticks_geo])

    # Add a light grid
    ax.grid(linestyle='--', alpha=0.5, zorder=0)

    # Style the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(9)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(9)

    # Add north arrow
    na = NorthArrow(location="center left", rotation={"degrees": 0}, scale=0.6)
    ax.add_artist(na.copy())

    # Add scale bar
    draw_accurate_scalebar(ax, islands_wm,
                           location=(islands_wm.total_bounds[0] + 100,
                                     islands_wm.total_bounds[1] + 100),
                           length_km=100,  # Total length in km
                           segments=4,
                           height=6500)

    # Add legend for expedition phases
    legend = ax.legend(loc='upper right',
                      title='Expedition Phases',
                      fontsize=10,
                      title_fontsize=11,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      framealpha=0.9)

    # Adjust legend marker size
    # legend.legend_handles
    # for handle in legend.legend_handles:
    #     handle.set
    #     handle.set_markersize(8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Map saved to {output_path}")

    # Print summary statistics
    print("\nExpedition Summary:")
    for phase in sorted(expedition_colors.keys()):
        phase_data = flight_database[flight_database['expedition_phase'] == phase]
        if not phase_data.empty:
            print(f"{expedition_labels[phase]}: {len(phase_data)} photos")


if __name__ == "__main__":
    CRS_utm_zone_15 = "32715"
    EPSG_WGS84 = "4326"

    flight_database_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")
    flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=EPSG_WGS84)

    full_hasty_annotation_file_path = Path(
        "/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")
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
                                                 "island": "island_new_name",
                                                 "site_code": "site_code_new_name",
                                                 "mission_folder": "mission_folder_new_name",
                                                 "datetime_digitized": "datetime_digitized_new_name"}, inplace=True)

    gdf_hasty_images_merged = gdf_hasty_images.merge(df_flight_database_new_names,
                                                     left_on="image_hash",
                                                     right_on="image_hash",
                                                     how="inner")

    # filter the flight_database for the images that are in the hasty images
    flight_database_filtered = flight_database[
        flight_database["image_hash"].isin(gdf_hasty_images_merged["image_hash"])]
    flight_database_filtered.to_file("labelled_hasty_images.geojson", driver="GeoJSON")
    # get the full mission
    flight_database_full_missions_filtered = flight_database[
        flight_database["mission_folder"].isin(flight_database_filtered["mission_folder"])]

    create_galapagos_expedition_map(
        gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg",
        gdf_flight_database=flight_database_filtered
    )