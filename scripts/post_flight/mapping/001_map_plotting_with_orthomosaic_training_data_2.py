"""
Creates an overall Galápagos Islands map, then add the training data of orhomosaics origin

001_map_plotting_with_orthomosaic_training_data_2

See 044_prepare_orthomosaic_classifiation for the complete pipeline
"""
import typing

import geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from matplotlib_map_utils.core.north_arrow import NorthArrow
from pathlib import Path
from shapely.geometry import box
from shapely.lib import unary_union

from active_learning.database import images_data_extraction, derive_image_metadata
from active_learning.types.Exceptions import NoLabelsError, AnnotationFileNotSetError, ProjectionError, \
    OrthomosaicNotSetError, NotEnoughLabelsError
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid
from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks, get_utm_epsg, island_utm_zones
from active_learning.util.mapping.plots import plot_orthomomsaic_training_data
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.converter.HastyConverter import ImageFormat
from geospatial_transformations import get_geotiff_compression, get_gsd
import networkx as nx




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





def group_nearby_polygons_simple(gdf, buffer_distance=250):
    """
    Simple and fast: buffer each polygon, find intersections, group connected ones
    """
    # Step 1: Create buffered polygons
    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance)

    # Step 2: Find intersections using spatial join
    intersections = gpd.sjoin(gdf_buffered, gdf, how='inner', predicate='intersects')

    # Remove self-intersections
    intersections = intersections[intersections.index != intersections.index_right]

    # Step 3: Build graph of connections
    G = nx.Graph()
    G.add_nodes_from(gdf.index)

    # Add edges for intersecting polygons
    for idx, row in intersections.iterrows():
        G.add_edge(idx, row.index_right)

    # Step 4: Find connected components (groups)
    groups = list(nx.connected_components(G))

    # Step 5: Keep only groups with more than 1 polygon
    multi_groups = [group for group in groups if len(group) > 1]

    # Create result
    if multi_groups:
        result_indices = []
        group_labels = {}

        for group_id, group in enumerate(multi_groups):
            for idx in group:
                result_indices.append(idx)
                group_labels[idx] = group_id

        result_gdf = gdf.loc[result_indices].copy()
        result_gdf['group_id'] = result_gdf.index.map(group_labels)
        return result_gdf
    else:
        # Return empty GeoDataFrame
        empty_gdf = gdf.iloc[0:0].copy()
        empty_gdf['group_id'] = []
        return empty_gdf



if __name__ == "__main__":
    CRS_utm_zone_15 = 32715
    EPSG_WGS84 = 4326
    web_mercator_projection_epsg = 3857

    stats_collection = {}
    analysis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_analysis/")
    vis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/visualisation")

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    herdnet_annotations = []
    problematic_data_pairs = []
    usable_training_data = []
    usable_training_annotations = []

    flight_database_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")
    gdf_flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=CRS_utm_zone_15)

    gdf_usable_training_data_raster_mask = gpd.read_file(filename="usable_training_data_raster_mask.geojson")
    gdf_usable_training_annotations = gpd.read_file(filename="usable_training_data_annotations.geojson").to_crs(CRS_utm_zone_15)

    # TODO seperate the generation of these files and this clustering step


    gdf_usable_training_data_raster_mask = group_nearby_polygons_simple(gdf=gdf_usable_training_data_raster_mask.to_crs(CRS_utm_zone_15), buffer_distance=250)
    # islands_wm['utm_zone'] = islands_wm['nombre'].map(lambda x: island_utm_zones.get(x, "Unknown"))

    group_config = {
        # inset_map_location: 'upper right', 'upper left', 'lower left', 'lower right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        "San Cristobal_10.01.2020_SRL": {"inset_map_location": "lower right", "legend_location": "lower left",
                                         "scalebar_location": "upper left", "color": "#FF5733",
                                         "island": "San Cristobal",
                                         "textbox_location": (0.05, 0.98),
                                         "epsg": get_utm_epsg(island_utm_zones["San Cristobal"]), },
        "Espanola_12.01.2021_EGB": {"inset_map_location": "center left", "legend_location": "lower right",
                                    "scalebar_location": "upper left", "color": "#FF5733", "island": "Española",
                                    "textbox_location": (0.05, 0.98),
                                    "epsg": get_utm_epsg(island_utm_zones["Española"])},
        "Espanola_26.01.2021_13.01.2021_12.01.2021_EM": {"inset_map_location": "lower left",
                                                         "legend_location": "lower right",
                                                         "scalebar_location": "upper left", "color": "#FF5733",
                                                         "textbox_location": (0.05, 0.98),
                                                         "island": "Española",
                                                         "epsg": get_utm_epsg(island_utm_zones["Española"])},
        'Floreana_22.01.2021_FLPC': {"inset_map_location": "lower left", "legend_location": "lower right",
                                     "scalebar_location": "upper left", "color": "#FF5733", "island": "Floreana",
                                     "textbox_location": (0.05, 0.98),
                                     "epsg": get_utm_epsg(island_utm_zones["Floreana"])},
        'Espanola_26.01.2021_EPCN': {"inset_map_location": "center right", "legend_location": "lower left",
                                     "scalebar_location": "upper left", "color": "#FF5733", "island": "Española",
                                     "textbox_location": (0.05, 0.98),
                                     "epsg": get_utm_epsg(island_utm_zones["Española"])},
        'Floreana_04.02.2021_FLPO': {"inset_map_location": "lower right", "legend_location": "lower left",
                                     "scalebar_location": "upper left", "color": "#FF5733", "island": "Floreana",
                                     "textbox_location": (0.05, 0.98),
                                     "epsg": get_utm_epsg(island_utm_zones["Floreana"])},
        'Genovesa_04.12.2021_05.12.2021_GES': {"inset_map_location": "upper left", "legend_location": "lower left",
                                               "scalebar_location": "upper left", "color": "#FF5733",
                                               "textbox_location": (0.55, 0.05),
                                               "island": "Genovesa",
                                               "epsg": get_utm_epsg(island_utm_zones["Genovesa"])},
        'Marchena_07.12.2021_09.12.2021_MNW_MBBE': {"inset_map_location": "lower right", "legend_location": "lower left",
                                                    "scalebar_location": "upper left", "color": "#FF5733",
                                                    "textbox_location": (0.05, 0.98),
                                                    "island": "Marchena",
                                                    "epsg": get_utm_epsg(island_utm_zones["Marchena"])},
        'Isabela_16.12.2021_ISPF': {"inset_map_location": "lower right", "legend_location": "lower left",
                                    "scalebar_location": "upper left", "color": "#FF5733", "island": "Isabela",
                                    "textbox_location": (0.05, 0.98),
                                    "epsg": get_utm_epsg(island_utm_zones["Isabela"])},
        'Fernandina_19.12.2021_FNE_FNF': {"inset_map_location": "lower right", "legend_location": "lower left",
                                          "scalebar_location": "upper left", "color": "#FF5733", "island": "Fernandina",
                                          "textbox_location": (0.05, 0.95),
                                          "epsg": get_utm_epsg(island_utm_zones["Fernandina"])},
        'Isabela_17.01.2023_ISPDA': {"inset_map_location": "upper left", "legend_location": "lower left",
                                     "scalebar_location": "upper left", "color": "#FF5733", "island": "Isabela",
                                     "textbox_location": (0.35, 0.98),
                                     "epsg": get_utm_epsg(island_utm_zones["Isabela"])},
        'Isabela_27.01.2023_ISVI_ISVP': {"inset_map_location": "lower left", "legend_location": "lower right",
                                         "scalebar_location": "upper left", "color": "#FF5733", "island": "Isabela",
                                         "textbox_location": (0.25, 0.98),
                                         "epsg": get_utm_epsg(island_utm_zones["Isabela"])},
        'Fernandina_19.12.2021_FNI_FNJ': {"inset_map_location": "lower right", "legend_location": "lower left",
                                         "scalebar_location": "upper left", "color": "#FF5733", "island": "Fernandina",
                                          "textbox_location": (0.05, 0.98),
                                         "epsg": get_utm_epsg(island_utm_zones["Fernandina"])},
    }

    for group_id, group_gdf in gdf_usable_training_data_raster_mask.groupby("group_id"):
        if group_id == -1:  # Skip ungrouped/noise points
            continue

        # get the extent of the group
        group_extent = group_gdf.total_bounds

        gdf_group_usable_training_annotations = gdf_usable_training_annotations.clip(group_extent)

        # TODO plot the group extent containing the polygons, island, and annotations
        # Plot the group
        if len(group_gdf['Date'].unique()) > 5:
            raise ValueError("Group contains data from multiple dates, which is not supported in this version.")
        config_key = f"{'_'.join(group_gdf['island'].unique())}_{'_'.join(group_gdf['Date'].unique())}_{'_'.join(group_gdf['Site code'].unique())}"


        plot_orthomomsaic_training_data(
            group_extent=group_extent,
            group_gdf=group_gdf,
            gdf_group_annotations=gdf_group_usable_training_annotations,
            group_id=config_key,
            vis_config = group_config.get(config_key, {"inset_map_location": "lower right", "legend_location": "upper left", "scalebar_location": "upper left", "color": "#FF5733", "island": "Unknown"}),
            output_dir=vis_output_dir
        )

        # Save as GeoJSON
        filename = f"group_{config_key}.geojson"
        group_gdf.to_file(filename, driver='GeoJSON')

        print(f"Saved group {config_key} with {len(group_gdf)} polygons to {filename}")

    print(f"stats_collection: {stats_collection}")
    with open("stats_collection.json", "w") as f:
        import json
        json.dump(stats_collection, f, indent=4)

    # Visualize the groups
    import matplotlib.pyplot as plt


