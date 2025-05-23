import typing
from loguru import logger
from matplotlib_map_utils import inset_map, indicate_extent

from active_learning.util.mapping.helper import get_islands
from active_learning.util.visualisation.drone_flights import visualise_flights

"""
plot the training data to illustrate what is annotated and what is not

1. get the database from all images every taken and the hasty annotateed images. The latter are still in the old name schema

There are two types: the orthomosaics and the direct drone shots



"""





import pandas as pd
import geopandas as gpd
from pathlib import Path

from active_learning.database import images_data_extraction, derive_image_metadata
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path
import typing
from shapely.geometry import Point, LineString
import rasterio
from rasterio.plot import plotting_extent
import contextily as ctx


def visualise_flight_path_geospatial(mission_names_set: list,
                                     filtered_df: gpd.GeoDataFrame,
                                     gdf_mission_polygons: gpd.GeoDataFrame,
                                     flight_database_full: gpd.GeoDataFrame = None,
                                     islands_gdf: gpd.GeoDataFrame = None,
                                     orthomosaics: typing.Optional[typing.List[Path]] = None,
                                     output_path: str = None):
    """
    Display the flight path, the coastline of the island and the hasty images

    Parameters:
    -----------
    mission_names_set : list
        List of mission names to visualize
    filtered_df : gpd.GeoDataFrame
        GeoDataFrame containing labeled/hasty images
    gdf_mission_polygons : gpd.GeoDataFrame
        GeoDataFrame containing mission polygons
    flight_database_full : gpd.GeoDataFrame, optional
        Full flight database with all points
    islands_gdf : gpd.GeoDataFrame, optional
        GeoDataFrame containing island coastlines
    orthomosaics : List[Path], optional
        List of paths to orthomosaic files to overlay
    output_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    gdf_mission_polygons = gdf_mission_polygons.to_crs(epsg=islands_gdf.crs.to_epsg())
    filtered_df = filtered_df.to_crs(epsg=islands_gdf.crs.to_epsg())
    flight_database_full = flight_database_full.to_crs(epsg=islands_gdf.crs.to_epsg())

    # Filter data for the specified missions
    mission_polygons = gdf_mission_polygons[gdf_mission_polygons['mission_folder'].isin(mission_names_set)]
    mission_labeled_images = filtered_df[filtered_df['mission_folder'].isin(mission_names_set)]

    # Get flight paths for these missions if available
    if flight_database_full is not None:
        mission_flight_data = flight_database_full[flight_database_full['mission_folder'].isin(mission_names_set)]
    else:
        raise ValueError("Flight database is not provided or empty.")

    # Determine the overall bounds for all data
    all_bounds = []
    if not mission_polygons.empty:
        all_bounds.append(mission_polygons.total_bounds)
    if not mission_labeled_images.empty:
        all_bounds.append(mission_labeled_images.total_bounds)
    if not mission_flight_data.empty:
        all_bounds.append(mission_flight_data.total_bounds)

    if all_bounds:
        minx = min(bounds[0] for bounds in all_bounds)
        miny = min(bounds[1] for bounds in all_bounds)
        maxx = max(bounds[2] for bounds in all_bounds)
        maxy = max(bounds[3] for bounds in all_bounds)

        # Add buffer around the data
        buffer_x = (maxx - minx) * 0.6
        buffer_y = (maxy - miny) * 0.4

        # Calculate width and height of the bounding box
        width = (maxx + buffer_x) - (minx - buffer_x)
        height = (maxy + buffer_y) - (miny - buffer_y)

        # Set a base size (e.g. 10 units on the smaller dimension)
        base_size = 18.0
        if width > height:
            fig_width = base_size
            fig_height = base_size * (height / width)
        else:
            fig_height = base_size
            fig_width = base_size * (width / height)

        # Create figure with aspect-ratio-respecting size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#fafafa')

        ax.set_xlim(minx - buffer_x, maxx + buffer_x)
        ax.set_ylim(miny - buffer_y, maxy + buffer_y)
    else:
        raise ValueError("No data available to plot.")



    # 1. Plot island coastlines first (background)
    if islands_gdf is not None and not islands_gdf.empty:
        # Clip islands to the current view if needed
        islands_gdf.plot(ax=ax,
                         facecolor='#f0f0f0',
                         edgecolor='#888888',
                         linewidth=1.5,
                         alpha=0.8,
                         zorder=1)

    # 2. Plot orthomosaics if provided
    if orthomosaics:
        for i, ortho_path in enumerate(orthomosaics):
            try:
                with rasterio.open(ortho_path) as src:
                    # Get the extent of the raster
                    extent = plotting_extent(src)

                    # Read and plot the raster
                    from rasterio.plot import show
                    show(src, ax=ax, extent=extent, alpha=0.6, zorder=2)

            except Exception as e:
                print(f"Could not load orthomosaic {ortho_path}: {e}")

    # 3. Plot mission polygons
    colors = plt.cm.Set3(np.linspace(0, 1, len(mission_names_set)))
    mission_color_map = dict(zip(mission_names_set, colors))

    for mission_name in mission_names_set:
        mission_poly = mission_polygons[mission_polygons['mission_folder'] == mission_name]
        if not mission_poly.empty:
            mission_poly.plot(ax=ax,
                              facecolor=mission_color_map[mission_name],
                              edgecolor='black',
                              linewidth=2,
                              alpha=0.3,
                              label=f'Mission Area: {mission_name}',
                              zorder=3)

    # 4. Plot flight paths
    if not mission_flight_data.empty:
        for mission_name in mission_names_set:
            mission_points = mission_flight_data[mission_flight_data['mission_folder'] == mission_name]
            if len(mission_points) > 1:
                # Sort by timestamp or index to get proper flight order
                if 'timestamp' in mission_points.columns:
                    mission_points = mission_points.sort_values('timestamp')
                else:
                    mission_points = mission_points.sort_index()

                # Extract coordinates
                x = [point.x for point in mission_points.geometry]
                y = [point.y for point in mission_points.geometry]

                # Create flight path line
                if len(x) > 1:
                    # Color by altitude if available
                    if 'height' in mission_points.columns:
                        color_values = mission_points['height']
                        color_label = 'Height (m)'
                    elif 'RelativeAltitude' in mission_points.columns:
                        color_values = mission_points['RelativeAltitude']
                        color_label = 'Relative Altitude (m)'
                    else:
                        color_values = np.arange(len(mission_points))
                        color_label = 'Flight Sequence'

                    # Create segments for colored line
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Create LineCollection
                    norm = plt.Normalize(color_values.min(), color_values.max())
                    lc = LineCollection(segments,
                                        cmap='viridis',
                                        norm=norm,
                                        linewidth=3,
                                        alpha=0.8,
                                        zorder=5)
                    lc.set_array(color_values)
                    line = ax.add_collection(lc)

                    # Add start and end markers
                    ax.scatter(x[0], y[0], s=200, marker='o',
                               color='green', edgecolor='white', linewidth=3,
                               label=f'Start: {mission_name}', zorder=7)
                    ax.scatter(x[-1], y[-1], s=200, marker='s',
                               color='red', edgecolor='white', linewidth=3,
                               label=f'End: {mission_name}', zorder=7)

    # 5. Plot labeled/hasty images
    if not mission_labeled_images.empty:
        # Plot all labeled images as red dots
        mission_labeled_images.plot(ax=ax,
                                    marker='o',
                                    color='red',
                                    markersize=50,
                                    alpha=0.8,
                                    label=f'Labeled Images ({len(mission_labeled_images)})',
                                    zorder=8)

        # Add image count annotations for each mission
        for mission_name in mission_names_set:
            mission_images = mission_labeled_images[mission_labeled_images['mission_folder'] == mission_name]
            if not mission_images.empty:
                # Get centroid of labeled images for this mission
                centroid = mission_images.geometry.unary_union.centroid
                ax.annotate(f'{len(mission_images)} images',
                            (centroid.x, centroid.y),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor='white',
                                      edgecolor='red',
                                      alpha=0.9),
                            fontsize=10,
                            fontweight='bold',
                            zorder=9)
    else:
        logger.warning("No labeled images found for the specified missions.")

    # 6. Add colorbar for flight paths if there are any
    if not mission_flight_data.empty and 'height' in mission_flight_data.columns:
        # Create a dummy scatter for the colorbar
        scatter = ax.scatter([], [], c=[], cmap='viridis', s=0)
        scatter.set_array(mission_flight_data['height'])
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046, aspect=30)
        cbar.set_label('Flight Height (m)', fontsize=12, color='#555555')
        cbar.ax.tick_params(colors='#666666')

    # Adding an inset map to the plot
    iax = inset_map(ax, location="upper right", size=2.5, pad=0.1, xticks=[], yticks=[])
    # Plotting alaska in the inset map
    islands_gdf.plot(ax=iax, color='lightgrey', edgecolor='black', linewidth=0.5, alpha=0.7)
    # if not islands_gdf.empty:
    #     islands_gdf.plot(ax=iax, color='#AAAAFF', edgecolor='black', linewidth=0.8)

    # Creating the extent indicator, which appears by-default as a red square on the map
    indicate_extent(iax, ax, islands_gdf.crs.to_epsg(), islands_gdf.crs.to_epsg())

    # 7. Styling and annotations
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.3, color='#dddddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Title and labels
    mission_names_str = ', '.join(mission_names_set)
    if len(mission_names_str) > 50:
        mission_names_str = f"{len(mission_names_set)} missions"

    ax.set_title(f"Flight Analysis: {island_name}",
                 fontsize=16, fontweight='bold', color='#333333', pad=20)

    ax.set_xlabel('Longitude', fontsize=12, color='#666666')
    ax.set_ylabel('Latitude', fontsize=12, color='#666666')

    # 8. Add statistics box
    stats_lines = []
    stats_lines.append(f"Missions: {len(mission_names_set)}")

    if not mission_labeled_images.empty:
        stats_lines.append(f"Labeled images: {len(mission_labeled_images)}")

    if not mission_flight_data.empty:
        stats_lines.append(f"Total Photos: {len(mission_flight_data)}")

        # Calculate total flight distance if possible
        total_distance = 0
        for mission_name in mission_names_set:
            mission_points = mission_flight_data[mission_flight_data['mission_folder'] == mission_name]
            if len(mission_points) > 1 and 'distance_to_prev' in mission_points.columns:
                total_distance += mission_points['distance_to_prev'].sum()

        if total_distance > 0:
            stats_lines.append(f"Total flight distance: {total_distance:.1f}m")

    if not mission_polygons.empty:
        total_area = mission_polygons.geometry.area.sum()
        stats_lines.append(f"Coverage area: {total_area:.0f} m²")

    stats_text = '\n'.join(stats_lines)

    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white",
                      edgecolor="#dddddd",
                      alpha=0.95),
            verticalalignment='bottom',
            zorder=10)

    # 9. Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels,
                  loc='upper left',
                  frameon=True,
                  framealpha=0.9,
                  facecolor='white',
                  edgecolor='#dddddd',
                  fontsize=10)

    # 10. Add north arrow if using geographic coordinates
    if (not mission_labeled_images.empty and
            mission_labeled_images.crs and
            mission_labeled_images.crs.is_geographic):
        arrow_x, arrow_y = 0.95, 0.85
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.05),
                    xycoords='axes fraction', textcoords='axes fraction',
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='#444444',
                    arrowprops=dict(arrowstyle='->', lw=3, color='#444444'),
                    zorder=10)

        # Add circular background for north arrow
        from matplotlib.patches import Circle
        circle = Circle((arrow_x, arrow_y - 0.025), 0.03,
                        transform=ax.transAxes, facecolor='white',
                        edgecolor='#dddddd', alpha=0.9, zorder=9)
        ax.add_patch(circle)

    # 11. Add watermark
    fig.text(0.99, 0.01, "Generated with Python • Galápagos Marine Iguana Project",
             ha='right', va='bottom', fontsize=8, color='#cccccc', style='italic')

    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved geospatial flight visualization to {output_path}")

    return fig, ax


CRS_utm_zone_15 = "32715"
EPSG_WGS84 = "4326"

flight_database_path= Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")
flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=EPSG_WGS84)

full_hasty_annotation_file_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")
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
                             "island":"island_new_name",
                             "site_code": "site_code_new_name",
                             "mission_folder": "mission_folder_new_name",
                             "datetime_digitized": "datetime_digitized_new_name"}, inplace=True)

gdf_hasty_images_merged = gdf_hasty_images.merge(df_flight_database_new_names,
                                                left_on="image_hash",
                                                right_on="image_hash",
                                                how="inner")

# filter the flight_database for the images that are in the hasty images
flight_database_filtered = flight_database[flight_database["image_hash"].isin(gdf_hasty_images_merged["image_hash"])]
flight_database_filtered.to_file("labelled_hasty_images.geojson", driver="GeoJSON")
# get the full mission
flight_database_full_missions_filtered = flight_database[flight_database["mission_folder"].isin(flight_database_filtered["mission_folder"])]

# create a polygon from each mission
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import LineString

mission_polygons = []
mission_lines = []

buffer_radius = 10  # meters

for mission_folder, group in flight_database_full_missions_filtered.groupby("mission_folder"):
    group = group.dropna(subset=["geometry"])
    group = group.to_crs(epsg=CRS_utm_zone_15) # TODO get the right project for each zone
    if group.empty:
        continue

    # Sort for LineString
    group_sorted = group.sort_values("timestamp") if "timestamp" in group.columns else group.sort_index()
    coords = group_sorted.geometry.apply(lambda p: (p.x, p.y)).tolist()

    if len(coords) >= 2:
        flight_path = LineString(coords)
        mission_lines.append({
            "mission_folder": mission_folder,
            "geometry": flight_path
        })

    # Create buffer union (tight polygon)
    buffered_points = group.geometry.buffer(buffer_radius).to_crs(flight_database_full_missions_filtered.crs)
    merged = unary_union(buffered_points)

    mission_polygons.append({
        "mission_folder": mission_folder,
        "geometry": merged
    })

gdf_mission_lines = gpd.GeoDataFrame(mission_lines, crs=flight_database_full_missions_filtered.crs)
gdf_mission_polygons = gpd.GeoDataFrame(mission_polygons, crs=flight_database_full_missions_filtered.crs)

gdf_mission_polygons.to_file("labelled_hasty_mission_polygons.geojson", driver="GeoJSON")

mission_names_filter = [
    # Floreana, clockwise order
    {"island_name": "Floreana", "missions": ["FLMO04_03022021", "FLMO05_03022021", "FLMO06_03022021"]},
    {"island_name": "Floreana", "missions": ["FLMO01_02022021", "FLMO02_02022021", "FLMO03_02022021"]},
    {"island_name": "Floreana", "missions": ["FLMO02_28012023"]},
    {"island_name": "Floreana", "missions": ["FLBB01_28012023"]},  # intersection with annotated raster
    {"island_name": "Floreana", "missions": ["FLPC07_22012021"]},  # intersection with annotated raster
    {"island_name": "Floreana", "missions": ["FLPA03_21012021"]},
    {"island_name": "Floreana", "missions": ["FLSCA02_23012021"]},

    # Genovesa, clockwise order
    {"island_name": "Genovesa", "missions": ["GES06_04122021", "GES07_04122021"]},  # intersection with annotated raster
    {"island_name": "Genovesa", "missions": ["GES13_05122021"]},  # intersection with annotated raster

    # Santiago
    {"island_name": "Santiago", "missions": ["STJB01_10012023"]},

    # Fernandina
    {"island_name": "Fernandina", "missions": ["FCD01_20122021", "FCD02_20122021", "FCD03_20122021"]},
    {"island_name": "Fernandina", "missions": ["FPE01_18122021"]},
    {"island_name": "Fernandina", "missions": ["FEA01_18122021"]},  # accidentally assigned to Floreana in Hasty
]

islands_gdf = get_islands()
output_dir = "."
output_path = f"{output_dir}/hasty_single_image_training_data_overview.png"





output_dir = "."
for i, mission_names_set in enumerate(mission_names_filter):
    # Filter the DataFrame for the specific mission names
    filtered_df = gdf_mission_polygons[gdf_mission_polygons["mission_folder"].isin(mission_names_set)]
    island_name = mission_names_set["island_name"]
    missions = mission_names_set["missions"]
    # visualise the filtered DataFrame
    # TODO implement the visualisation

    output_path = f"{output_dir}/hasty_single_image_training_data_{island_name}_set_{i + 1}.png"

    fig, ax = visualise_flight_path_geospatial(
        mission_names_set=missions,
        filtered_df=flight_database_filtered,
        gdf_mission_polygons=gdf_mission_polygons,
        flight_database_full=flight_database,
        islands_gdf=islands_gdf[islands_gdf["nombre"] == island_name],
        output_path=output_path
    )

    plt.show()
    plt.close(fig)







# # Get unique pairs/combinations of both columns
# unique_combinations = gdf_hasty_images_merged[["mission_folder_new_name", "mission_folder"]].drop_duplicates()
#
#
# # ==========
#
# # TODO annotated hasty images
# hA = HastyAnnotationV2.from_file(full_hasty_annotation_file_path)
#
# # TODO rename the images in the hasty annotation
# # TODO rename the images and folders
#
# # TODO create a LaTeX table with metadata from the flights
#
# raster_mask_dd_path = Path("/Volumes/2TB/SamplingIssues/RasterMask/DD/")
# raster_mask_ms_path = Path("/Volumes/2TB/SamplingIssues/RasterMask/MS/")
#
#
#
# annotations_path = Path("/Volumes/2TB/Manual_Counting/Geospatial_Annotations/")
#
# # orthomosaic_shapefile_mapping_path = Path(
# #     "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/enriched_GIS_progress_report_with_stats.csv")
# # df_orthomosaic_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)
# #
# # df_orthomosaic_mapping