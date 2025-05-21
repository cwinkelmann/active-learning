"""
Plots the Expedition Data for multiple years

TODO next map: where are the data points for volunteers

TODO: One map per island
"""

import geopandas as gpd
# Packages used by this tutorial
import matplotlib.pyplot as plt  # visualization
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
# Importing the main package
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box
import pyproj
import numpy as np
import numpy as np
from loguru import logger

from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, get_geographic_ticks, format_lat_lon, \
    draw_accurate_scalebar, find_closest_island

web_mercator_projection_epsg = 3857


def create_galapagos_map(
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        fligth_database_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet",
        output_path="galapagos_map.png",
        dpi=300):
    """
    Creates a map of the Galápagos Islands showing data locations and annotations.
    """

    name_col = "gr_isla"
    name_col = "nombre"

    island_plot_config = {
"Bartolome": {
    "utm_zone": "16M",
    "scalebar": {"location": (1200, 30), "length_km": 1, "segments": 4, "height": 50},
},
"Caamaño": {
    "utm_zone": "16M",
    "scalebar": {"location": (0, 20), "length_km": 0.1, "segments": 4, "height": 50},
},
"Santiago": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
},
"Wolf": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"San Cristobal": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
},
"Lobos": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Santa Fé": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
},
"Santa Cruz": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
},
"Rabida": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Pinzón": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Pinta": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Marchena": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Isabela": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
},
"Tortuga": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Genovesa": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Fernandina": {
    "utm_zone": "15M",
    "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
},
"Floreana": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
},
"Gardner por Floreana": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Caldwell": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Albany": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
"Española": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
},
"Daphne Major": {
    "utm_zone": "16M",
    "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
},
}


    # Load GeoPackage layers
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')
    # TODO use the geoparquet instead
    flight_database = gpd.read_parquet(fligth_database_path).to_crs(epsg=web_mercator_projection_epsg)

    # add year to dataframe to simplify grouping later
    flight_database['year'] = flight_database['datetime_digitized'].dt.year

    # Prepare base plot
    islands_wm = islands.to_crs(epsg=web_mercator_projection_epsg)

    islands_isla = islands_wm[islands_wm['tipo'] == 'Isla']
    #
    # Dictionary to store folder -> island mapping
    folder_island_map = {}
    # Determine name column
    if not name_col:
        raise ValueError("No valid name column found for labeling.")
    islands_wm_f = islands_isla.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

    # group by island
    islands_wm_f = islands_wm_f.dissolve(by=name_col, as_index=False)

    for folder_name, group in flight_database.groupby("folder_name"):
        # Calculate the centroid of all points in this folder
        # This is more representative than a single point
        points_unary = group.geometry.union_all()

        # If we got a single point, use it directly; otherwise get the centroid
        if hasattr(points_unary, 'centroid'):
            representative_point = points_unary.centroid
        else:
            representative_point = points_unary  # It's already a point

        # Find the closest island to this representative point
        closest_island = find_closest_island(representative_point, islands_wm_f, name_col)

        # Store in our mapping dictionary
        folder_island_map[folder_name] = closest_island

        logger.info(f"Folder {folder_name}: Closest island is {closest_island}")

    # assign the folder name to the islands dataframe
    flight_database['main_island'] = flight_database['folder_name'].map(folder_island_map)

    for idx, isla in islands_wm_f.iterrows():
        island_name = isla[name_col]  # Get the island name from the row

        # Filter flight database for this island
        flight_database_island = flight_database[flight_database['main_island'] == island_name]

        # If no data for this island, skip it
        if flight_database_island.empty:
            print(f"No data for island: {island_name}")
            continue

        # Get distinct years for this island
        distinct_years = sorted(flight_database_island['year'].unique())
        if not distinct_years:
            logger.error(f"No year data for island: {island_name}")
            continue

        # Create a figure with subplots - one for each year
        fig, axes = plt.subplots(1, len(distinct_years), figsize=(6 * len(distinct_years), 8),
                                 squeeze=False)

        # Loop through each year for this island
        for i, year in enumerate(distinct_years):
            ax = axes[0, i]  # Get the appropriate subplot

            # Filter data for this year
            year_data = flight_database_island[flight_database_island['year'] == year]

            # Create a GeoDataFrame for just this island
            gdf_island = gpd.GeoDataFrame(pd.DataFrame([isla]), geometry="geometry", crs=islands_wm_f.crs)
            # gdf_island = gdf_island.to_crs(epsg=web_mercator_projection_epsg)
            # Plot the island
            gdf_island.plot(ax=ax, alpha=0.7, edgecolor='black', color='lightgrey')

            # Plot the points for this year
            if not year_data.empty:
                year_data.plot(ax=ax, marker='o', color='red', markersize=5,
                               label=f"{len(year_data)} photos")
            #
            # Replace your draw_segmented_scalebar call with:

            cfg = island_plot_config[island_name]

            draw_accurate_scalebar(ax, gdf_island,
                                   location=(gdf_island.total_bounds[0] + cfg["scalebar"]["location"][0],
                                             gdf_island.total_bounds[1] + cfg["scalebar"]["location"][1]),
                                   length_km=cfg["scalebar"]["length_km"],  # Total length in km
                                   segments=4,
                                   height=150)

            x_ticks_proj, x_ticks_geo, y_ticks_proj, y_ticks_geo = get_geographic_ticks(ax,
                                                                                        epsg_from=web_mercator_projection_epsg,
                                                                                        epsg_to=4326)

            # Set the tick positions to the projected coordinates
            ax.set_xticks(x_ticks_proj)
            ax.set_yticks(y_ticks_proj)

            # Format the tick labels with the geographic coordinates
            ax.set_xticklabels([format_lat_lon(x, 0, is_latitude=False, rounding=3) for x in x_ticks_geo])
            ax.set_yticklabels([format_lat_lon(y, 0, is_latitude=True, rounding=3) for y in y_ticks_geo])

            # Set title and labels
            ax.set_title(f'{year} ({len(year_data)} photos)', fontsize=12)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend()

            # Add grid lines
            ax.grid(alpha=0.3)



        # fig.suptitle(f'Photo Locations on {island_name} by Year', fontsize=16)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)  # Make room for suptitle
        plt.subplots_adjust(top=0.6)  # Uncomment this line and put it AFTER tight_layout

        # Save the figure
        output_file = f"island_{island_name.replace(' ', '_')}_by_year.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot for {island_name} to {output_file}")
        plt.show()
        plt.close(fig)  # Close the figure to free up memory








if __name__ == "__main__":
    create_galapagos_map(gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg")