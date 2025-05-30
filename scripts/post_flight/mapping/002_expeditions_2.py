"""
Plots the Expedition Data for multiple years
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box
import pyproj
import numpy as np
from loguru import logger

from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, get_geographic_ticks, format_lat_lon, \
    draw_accurate_scalebar, find_closest_island

# Corrected Galápagos Island UTM Zone Assignments
# Using proper hemisphere-based UTM zones
island_utm_zones = {
    "Bartolome": "15S",    # ~-90.56°, south of equator
    "Caamaño": "15S",      # ~-90.3°, south of equator near Santa Cruz
    "Santiago": "15S",     # ~-90.8°, straddles equator but mostly south
    "Wolf": "15N",         # ~-91.8°, north of equator
    "San Cristobal": "16S", # ~-89.6°, south of equator
    "Lobos": "16S",        # ~-89.6°, south of equator, near San Cristobal
    "Santa Fé": "15S",     # ~-90.4°, south of equator
    "Santa Cruz": "15S",   # ~-90.3°, south of equator
    "Rabida": "15S",       # ~-90.7°, south of equator
    "Pinzón": "15S",       # ~-90.6°, south of equator
    "Pinta": "15N",        # ~-90.75°, north of equator
    "Marchena": "15N",     # ~-90.5°, north of equator
    "Isabela": "15S",      # ~-91.1°, straddles equator but mostly south
    "Tortuga": "15S",      # ~-91.4°, south of equator
    "Genovesa": "16N",     # ~-89.95°, north of equator
    "Fernandina": "15S",   # ~-91.6°, south of equator
    "Floreana": "15S",     # ~-90.3°, south of equator
    "Gardner por Floreana": "15S", # ~-90.3°, south of equator near Floreana
    "Caldwell": "16S",     # ~-89.6°, south of equator near Española
    "Albany": "16S",       # ~-89.6°, south of equator near Española
    "Española": "16S",     # ~-89.5°, south of equator
    "Daphne Major": "15S", # ~-90.3°, south of equator near Santa Cruz
}

def get_utm_epsg(utm_zone):
    """
    Get the EPSG code for a given UTM zone
    UTM zones are in the format '15N', '15S', etc.

    For UTM north zones: EPSG = 32600 + zone_number
    For UTM south zones: EPSG = 32700 + zone_number
    """
    zone_number = int(utm_zone[:-1])
    hemisphere = utm_zone[-1]

    if hemisphere == 'N':
        return 32600 + zone_number
    elif hemisphere == 'S':
        return 32700 + zone_number
    else:
        raise ValueError(f"Invalid UTM zone format: {utm_zone}")

def create_galapagos_map(
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        flight_database_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet",
        output_path="galapagos_map.png",
        dpi=300):
    """
    Creates a map of the Galápagos Islands showing data locations and annotations.
    """
    # Define the name column to use
    name_col = "nombre"

    # Dynamically build the island_plot_config dictionary using the corrected UTM zones
    island_plot_config = {}

    # Hardcoded island_plot_config with correct UTM zones but preserving original scalebar settings
    island_plot_config = {
        "Bartolome": {
            "utm_zone": island_utm_zones["Bartolome"],
            "epsg": get_utm_epsg(island_utm_zones["Bartolome"]),
            "scalebar": {"location": (1200, 30), "length_km": 4, "segments": 2, "height": 50},
        },
        "Caamaño": {
            "utm_zone": island_utm_zones["Caamaño"],
            "epsg": get_utm_epsg(island_utm_zones["Caamaño"]),
            "scalebar": {"location": (0, 20), "length_km": 0.1, "segments": 4, "height": 50},
        },
        "Santiago": {
            "utm_zone": island_utm_zones["Santiago"],
            "epsg": get_utm_epsg(island_utm_zones["Santiago"]),
            "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
        },
        "Wolf": {
            "utm_zone": island_utm_zones["Wolf"],
            "epsg": get_utm_epsg(island_utm_zones["Wolf"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "San Cristobal": {
            "utm_zone": island_utm_zones["San Cristobal"],
            "epsg": get_utm_epsg(island_utm_zones["San Cristobal"]),
            "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
        },
        "Lobos": {
            "utm_zone": island_utm_zones["Lobos"],
            "epsg": get_utm_epsg(island_utm_zones["Lobos"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Santa Fé": {
            "utm_zone": island_utm_zones["Santa Fé"],
            "epsg": get_utm_epsg(island_utm_zones["Santa Fé"]),
            "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
        },
        "Santa Cruz": {
            "utm_zone": island_utm_zones["Santa Cruz"],
            "epsg": get_utm_epsg(island_utm_zones["Santa Cruz"]),
            "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
        },
        "Rabida": {
            "utm_zone": island_utm_zones["Rabida"],
            "epsg": get_utm_epsg(island_utm_zones["Rabida"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Pinzón": {
            "utm_zone": island_utm_zones["Pinzón"],
            "epsg": get_utm_epsg(island_utm_zones["Pinzón"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Pinta": {
            "utm_zone": island_utm_zones["Pinta"],
            "epsg": get_utm_epsg(island_utm_zones["Pinta"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Marchena": {
            "utm_zone": island_utm_zones["Marchena"],
            "epsg": get_utm_epsg(island_utm_zones["Marchena"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Isabela": {
            "utm_zone": island_utm_zones["Isabela"],
            "epsg": get_utm_epsg(island_utm_zones["Isabela"]),
            "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
        },
        "Tortuga": {
            "utm_zone": island_utm_zones["Tortuga"],
            "epsg": get_utm_epsg(island_utm_zones["Tortuga"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Genovesa": {
            "utm_zone": island_utm_zones["Genovesa"],
            "epsg": get_utm_epsg(island_utm_zones["Genovesa"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Fernandina": {
            "utm_zone": island_utm_zones["Fernandina"],
            "epsg": get_utm_epsg(island_utm_zones["Fernandina"]),
            "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
        },
        "Floreana": {
            "utm_zone": island_utm_zones["Floreana"],
            "epsg": get_utm_epsg(island_utm_zones["Floreana"]),
            "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
        },
        "Gardner por Floreana": {
            "utm_zone": island_utm_zones["Gardner por Floreana"],
            "epsg": get_utm_epsg(island_utm_zones["Gardner por Floreana"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Caldwell": {
            "utm_zone": island_utm_zones["Caldwell"],
            "epsg": get_utm_epsg(island_utm_zones["Caldwell"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Albany": {
            "utm_zone": island_utm_zones["Albany"],
            "epsg": get_utm_epsg(island_utm_zones["Albany"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
        "Española": {
            "utm_zone": island_utm_zones["Española"],
            "epsg": get_utm_epsg(island_utm_zones["Española"]),
            "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
        },
        "Daphne Major": {
            "utm_zone": island_utm_zones["Daphne Major"],
            "epsg": get_utm_epsg(island_utm_zones["Daphne Major"]),
            "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
        },
    }

    # Load GeoPackage layers
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')

    # Load flight database and convert to WGS84 (EPSG:4326) first for flexibility
    flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=4326)

    # Add year to dataframe to simplify grouping later
    flight_database['year'] = flight_database['datetime_digitized'].dt.year
    flight_database['month'] = flight_database['datetime_digitized'].dt.month
    flight_database['year_month'] = flight_database['datetime_digitized'].dt.strftime('%Y_%m')

    expedition_mapping = {
        "2020_01": 1,
        "2021_01": 2,
        "2021_02": 2,
        "2021_12": 3,
        "2023_01": 4,
        "2023_02": 4,
        "2024_01": 5, # from the El Niño folder
        "2024_04": 6, # from the El Niño folder
        "2024_05": 6, # from the El Niño folder

    }
    # Map the expedition phases
    flight_database['expedition_phase'] = flight_database['year_month'].map(expedition_mapping)

    # Convert islands to WGS84 for easier reprojection later
    islands_wgs84 = islands.to_crs(epsg=4326)
    islands_isla = islands_wgs84[islands_wgs84['tipo'] == 'Isla']

    # Sort and deduplicate islands
    if not name_col:
        raise ValueError("No valid name column found for labeling.")
    islands_wgs84_f = islands_isla.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

    # Group by island
    islands_wgs84_f = islands_wgs84_f.dissolve(by=name_col, as_index=False)

    # Dictionary to store folder -> island mapping
    folder_island_map = {}

    # First project flight database points to Web Mercator for distance calculations
    flight_database_web_mercator = flight_database.to_crs(epsg=3857)
    islands_web_mercator = islands_wgs84_f.to_crs(epsg=3857)

    # Find the closest island for each folder
    for folder_name, group in flight_database_web_mercator.groupby("folder_name"):
        # Calculate the centroid of all points in this folder
        points_unary = group.geometry.union_all()

        # If we got a single point, use it directly; otherwise get the centroid
        if hasattr(points_unary, 'centroid'):
            representative_point = points_unary.centroid
        else:
            representative_point = points_unary  # It's already a point

        # Find the closest island to this representative point
        closest_island = find_closest_island(representative_point, islands_web_mercator, name_col)

        # Store in our mapping dictionary
        folder_island_map[folder_name] = closest_island

        logger.info(f"Folder {folder_name}: Closest island is {closest_island}")

    # Assign the folder name to the islands dataframe
    flight_database['main_island'] = flight_database['folder_name'].map(folder_island_map)

    # Process each island
    for idx, isla in islands_wgs84_f.iterrows():
        island_name = isla[name_col]  # Get the island name from the row

        # Skip if island not in our configuration
        if island_name not in island_plot_config:
            logger.warning(f"Island {island_name} not found in configuration, skipping.")
            continue

        # Get UTM zone and EPSG code for this island
        utm_zone = island_plot_config[island_name]["utm_zone"]
        utm_epsg = island_plot_config[island_name]["epsg"]

        # Filter flight database for this island
        flight_database_island = flight_database[flight_database['main_island'] == island_name]

        # If no data for this island, skip it
        if flight_database_island.empty:
            logger.info(f"No data for island: {island_name}")
            continue

        # Get distinct years for this island
        distinct_expedition = sorted(flight_database_island['expedition_phase'].unique())
        if not distinct_expedition:
            logger.error(f"No year data for island: {island_name}")
            continue

        # Create a figure with subplots - one for each year
        fig, axes = plt.subplots(1, len(distinct_expedition), figsize=(6 * len(distinct_expedition), 8),
                                 squeeze=False)

        # Project the island to its UTM zone
        gdf_island_utm = gpd.GeoDataFrame(pd.DataFrame([isla]), geometry="geometry", crs=islands_wgs84_f.crs)
        gdf_island_utm = gdf_island_utm.to_crs(epsg=utm_epsg)

        # Loop through each year for this island
        for i, expedition_phase in enumerate(distinct_expedition):
            ax = axes[0, i]  # Get the appropriate subplot

            # Filter data for this year
            year_data = flight_database_island[flight_database_island['expedition_phase'] == expedition_phase]

            # Project year data to the island's UTM zone
            year_data_utm = year_data.to_crs(epsg=utm_epsg)

            # Plot the island in UTM coordinates
            gdf_island_utm.plot(ax=ax, alpha=0.7, edgecolor='black', color='lightgrey')

            # Plot the points for this year in UTM coordinates
            if not year_data_utm.empty:
                year_data_utm.plot(ax=ax, marker='o', color='red', markersize=5,
                               label=f"{len(year_data_utm)} photos")

            # Get scale bar configuration for this island
            scale_cfg = island_plot_config[island_name]["scalebar"]

            # # Add scale bar appropriate for UTM coordinates (in meters)
            # draw_accurate_scalebar(ax, gdf_island_utm,
            #                        location=(gdf_island_utm.total_bounds[0] + scale_cfg["location"][0],
            #                                  gdf_island_utm.total_bounds[1] + scale_cfg["location"][1]),
            #                        length_km=scale_cfg["length_km"],
            #                        segments=scale_cfg["segments"],
            #                        height=scale_cfg["height"])

            # Add coordinate grid in UTM coordinates
            bounds = gdf_island_utm.total_bounds
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]

            # Calculate tick intervals for fewer ticks (aim for 4-5 ticks per axis)
            # We'll use a larger interval to reduce the number of ticks
            x_interval_multiplier = max(1, int(x_range / (3 * 1000)))  # Aim for ~4 ticks along x-axis
            y_interval_multiplier = max(1, int(y_range / (3 * 1000)))  # Aim for ~4 ticks along y-axis

            # Round to a nice interval (1km, 2km, 5km, 10km, etc.)
            def nice_interval(value):
                magnitude = 10 ** np.floor(np.log10(value))
                if value / magnitude <= 1:
                    return magnitude
                elif value / magnitude <= 2:
                    return 2 * magnitude
                elif value / magnitude <= 5:
                    return 5 * magnitude
                else:
                    return 10 * magnitude

            x_interval = nice_interval(x_interval_multiplier * 1000)  # Convert km to meters
            y_interval = nice_interval(y_interval_multiplier * 1000)  # Convert km to meters

            # Generate fewer ticks
            x_min = np.floor(bounds[0] / x_interval) * x_interval
            x_max = np.ceil(bounds[2] / x_interval) * x_interval
            y_min = np.floor(bounds[1] / y_interval) * y_interval
            y_max = np.ceil(bounds[3] / y_interval) * y_interval

            x_ticks = np.arange(x_min, x_max + 1, x_interval)
            y_ticks = np.arange(y_min, y_max + 1, y_interval)

            # Set the reduced ticks
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # Format tick labels to show kilometers instead of meters
            ax.set_xticklabels([f"{x / 1000:.0f}" for x in x_ticks])
            ax.set_yticklabels([f"{y / 1000:.0f}" for y in y_ticks])

            # Set title and labels with UTM zone info
            ax.set_title(f'Expedition Phase {expedition_phase}', fontsize=12)
            ax.set_xlabel(f'Easting (km, UTM {utm_zone})')
            ax.set_ylabel(f'Northing (km, UTM {utm_zone})')

            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)

            # Add north arrow (already correct in UTM)
            na = NorthArrow(location="upper left", rotation={"degrees": 0}, scale=0.5)
            ax.add_artist(na.copy())

        # Adjust title to include UTM zone
        # plt.suptitle(f'{island_name} (UTM Zone {utm_zone})', fontsize=16, y=0.98)

        plt.tight_layout()

        # Save the figure
        output_file = f"island_{island_name.replace(' ', '_')}_by_year.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot for {island_name} to {output_file}")
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    create_galapagos_map(gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg")