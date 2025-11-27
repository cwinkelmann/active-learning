"""
plot the individual training data on each island to illustrate what is annotated and what is not

1. get the database from all images every taken and the hasty annotateed images. The latter are still in the old name schema

There are two types: the orthomosaics and the direct drone shots



"""

import typing
from loguru import logger
from matplotlib_map_utils import inset_map, indicate_extent

from active_learning.config.mapping import mission_names_filter
from active_learning.types.image_metadata import list_images
from active_learning.util.drone_flight_check import get_analysis_ready_image_metadata
from active_learning.util.mapping.helper import get_islands, find_closest_island
from active_learning.util.visualisation.drone_flights import visualise_flights







import pandas as pd
import geopandas as gpd
from pathlib import Path

from active_learning.database import images_data_extraction, derive_image_metadata, create_image_db
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
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import LineString

def visualise_flight_path_geospatial(mission_names_set: list,
                                     filtered_df: gpd.GeoDataFrame,
                                     gdf_mission_polygons: gpd.GeoDataFrame,
                                     flight_database_full: gpd.GeoDataFrame = None,
                                     islands_gdf: gpd.GeoDataFrame = None,
                                     orthomosaics: typing.Optional[typing.List[Path]] = None,
                                     output_path: str = None,
                                     title: str = None,
                                        inset_map: bool = True,
                                     hA: HastyAnnotationV2 | None = None) -> typing.Tuple[plt.Figure, plt.Axes]:
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

    # gdf_mission_polygons = gdf_mission_polygons.to_crs(epsg=islands_gdf.crs.to_epsg())
    # filtered_df = filtered_df.to_crs(epsg=islands_gdf.crs.to_epsg())
    # flight_database_full = flight_database_full.to_crs(epsg=islands_gdf.crs.to_epsg())
    
    
    islands_gdf = islands_gdf.to_crs(epsg=gdf_mission_polygons.crs.to_epsg())
    
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
        buffer_x = (maxx - minx) * 0.1
        buffer_y = (maxy - miny) * 0.1

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

                    # # Add start and end markers
                    # ax.scatter(x[0], y[0], s=200, marker='o',
                    #            color='green', edgecolor='white', linewidth=3,
                    #            label=f'Start: {mission_name}', zorder=7)
                    # ax.scatter(x[-1], y[-1], s=200, marker='s',
                    #            color='red', edgecolor='white', linewidth=3,
                    #            label=f'End: {mission_name}', zorder=7)

    # 5. Plot labeled/hasty images
    if not mission_labeled_images.empty:
        # Plot all labeled images as red dots
        mission_labeled_images.plot(ax=ax,
                                    marker='o',
                                    color='red',
                                    markersize=40,
                                    alpha=0.8,
                                    label=f'Labeled Images ({len(mission_labeled_images)})',
                                    zorder=8)

        # Add image count annotations for each mission
        for mission_name in mission_names_set:
            mission_images = mission_labeled_images[mission_labeled_images['mission_folder'] == mission_name]
            if not mission_images.empty:
                # Get centroid of labeled images for this mission
                centroid = mission_images.geometry.unary_union.centroid
                ax.annotate(f'{mission_name}: images: {len(mission_images)}',
                            (centroid.x + 20, centroid.y),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor='white',
                                      edgecolor='red',
                                      alpha=0.9),
                            fontsize=12,
                            fontweight='bold',
                            zorder=9)
    else:
        logger.warning("No labeled images found for the specified missions.")

    # # 6. Add colorbar for flight paths if there are any
    # if not mission_flight_data.empty and 'height' in mission_flight_data.columns:
    #     # Create a dummy scatter for the colorbar
    #     scatter = ax.scatter([], [], c=[], cmap='viridis', s=0)
    #     scatter.set_array(mission_flight_data['height'])
    #     cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046, aspect=30)
    #     cbar.set_label('Flight Height (m)', fontsize=12, color='#555555')
    #     cbar.ax.tick_params(colors='#666666')

    if inset_map:
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

    if not title:
        title = f"Flight Analysis: {island_name}"
    ax.set_title(title,
                 fontsize=16, fontweight='bold', color='#333333', pad=20)

    ax.set_xlabel('Easting', fontsize=12, color='#666666')
    ax.set_ylabel('Northing', fontsize=12, color='#666666')

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
        stats_lines.append(f"Coverage area: {total_area / 100000:.5f} km²")

    stats_lines.append(f"Data CRS: {gdf_mission_polygons.crs.to_string()}")

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

    # # # 9. Add legend
    # handles, labels = ax.get_legend_handles_labels()
    # if handles:
    #     ax.legend(handles=handles, labels=labels,
    #               loc='upper left',
    #               frameon=True,
    #               framealpha=0.9,
    #               facecolor='white',
    #               edgecolor='#dddddd',
    #               fontsize=10)

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


    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved geospatial flight visualization to {output_path}")

    return fig, ax



if __name__ == "__main__":

    # import dataset_configs_hasty_point_iguanas as dataset_configs
    import scripts.training_data_preparation.dataset_configs_hasty_point_iguanas as dataset_configs
    base_path = Path("/raid/cwinkelmann/work/active_learning/mapping/database/")
    base_path_mapping = base_path / "mapping"
    base_path_mapping.mkdir(parents=True, exist_ok=True)


    for dataset in dataset_configs.datasets:
        logger.info(f"Starting Dataset {dataset.dataset_name}, split: {dataset.dset}")

        hasty_dataset_path = Path(
            # "/home/cwinkelmann/work/Herdnet/data/2025_09_28_orthomosaic_data/") / dataset.dataset_name / dataset.dset
            # "/raid/cwinkelmann/training_data/iguana/2025_08_10_endgame/") / dataset.dataset_name / dataset.dset
            "/raid/cwinkelmann/training_data/iguana/2025_10_11") / dataset.dataset_name / dataset.dset

        dataset_dict = dataset.model_dump()

        CRS_utm_zone_15 = "32715"
        EPSG_WGS84 = "4326"

        flight_database_path= Path(base_path / "2020_2021_2022_2023_2024_database_analysis_ready.parquet")
        flight_database = gpd.read_parquet(flight_database_path)

        missions = gdf = gpd.read_file('/raid/cwinkelmann/work/active_learning/mapping/database/mapping/Iguanas_From_Above_all_data.gpkg', layer='iguana_missions')

        output_dir = base_path_mapping / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        # flight_database.to_file(output_dir / f"full_flight_database_{dataset.dataset_name}_analysis_read.geojson",
        #                         driver="GeoJSON")
        flight_database = flight_database.to_crs(epsg=CRS_utm_zone_15)
        # read the right database
        #flight_database = get_analysis_ready_image_metadata(flight_database)
        #flight_database.to_parquet("/raid/cwinkelmann/work/active_learning/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")

        full_hasty_annotation_file_path = hasty_dataset_path / "hasty_format_full_size.json"
        hA = HastyAnnotationV2.from_file(full_hasty_annotation_file_path)

        hasty_images_path = hasty_dataset_path / "Default"
        #images_list = list_images(hasty_images_path, extension="JPG", recursive=True)
        #gdf_hasty_image_metadata = images_data_extraction(images_list)
        
        try:
            gdf_hasty_images = create_image_db(hasty_images_path)


        except Exception as e:
            logger.error(f"Problem with HASTY images database: {e}")
            continue

        site_code_filter = ["FCD", "FEA", "FPE", "FPM", "FWK", "FNA", "FEF", "FNJ", "FNI", "FND"
                                                                                           "FLMO", "FLBB", "FLPC",
                            "FLSCA", "FLPA",
                            "GES",
                            ]

        # filter the flight_database for the images that are in the hasty images
        flight_database_filtered = flight_database[flight_database["image_hash"].isin(gdf_hasty_images["image_hash"])]
        # missions_filtered = missions[missions["mission_folder"].isin(flight_database_filtered["mission_folder"])]
        missions_filtered = missions[missions["site_code"].isin(site_code_filter)]


        # # the the image_name mapping
        # a = flight_database[["image_hash", "image_name", "geometry"]]
        # b = gdf_hasty_images[["image_hash", "image_name"]]
        # mapping = pd.merge(a, b, on="image_hash", suffixes=("_new", "_old")).to_dict(orient="records")
        # gdf_mapping = gpd.GeoDataFrame(mapping, geometry="geometry")
        # gdf_mapping.to_file(base_path_mapping / f"image_name_mapping_{dataset.dataset_name}.geojson", driver="GeoJSON")

        # gdf_mapping = gpd.read_file("/raid/cwinkelmann/work/active_learning/mapping/database/mapping/All_detection_mapping.geojson")


        flight_database_filtered.to_file(output_dir / f"labelled_hasty_images_{dataset.dataset_name}_{dataset.dset}.geojson", driver="GeoJSON")
        # get the full mission
        flight_database_full_missions_filtered = flight_database[flight_database["mission_folder"].isin(flight_database_filtered["mission_folder"])]

        # get the mission statistics.
        missions_filtered.to_file(output_dir / f"labelled_missions_filtered_{dataset.dataset_name}_{dataset.dset}.geojson", driver="GeoJSON")

        # create a polygon from each mission


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

        gdf_mission_polygons.to_file(output_dir / f"labelled_hasty_mission_polygons_{dataset.dataset_name}_{dataset.dset}.geojson", driver="GeoJSON")
        gdf_mission_lines.to_file(output_dir / f"labelled_hasty_mission_lines_{dataset.dataset_name}_{dataset.dset}.geojson", driver="GeoJSON")

        islands_gdf = get_islands(gpkg_path=base_path / "sampling_issues.gpkg",
                                  fligth_database_path=flight_database_path,)



        # for i, mission_names_set in enumerate(mission_names_filter):
        missions = list(gdf_mission_polygons.mission_folder.unique())
        # Filter the DataFrame for the specific mission names

        # filtered_df = gdf_mission_polygons[gdf_mission_polygons["mission_folder"].isin(dataset.dataset_filter)]
        filtered_df = gdf_mission_polygons

        closest_island = find_closest_island(point_geometry=gdf_mission_polygons.centroid,
                            islands_gdf=islands_gdf, name_col = 'nombre')

        # island_name = mission_names_set["island_name"]
        # missions = dataset.dataset_filter
        # visualise the filtered DataFrame
        # TODO implement the visualisation

        island_name = list(flight_database_filtered.island.unique())[0]
        if flight_database_filtered.island.nunique() > 1:
            logger.warning(f"Multiple islands found in the filtered data: {island_name}, using the first one: {island_name[0]}")

        output_path = f"{output_dir}/hasty_single_image_training_data_{island_name}_set_{dataset.dataset_name}_{dataset.dset}.png"

        fig, ax = visualise_flight_path_geospatial(
            mission_names_set=missions,
            filtered_df=flight_database_filtered,
            gdf_mission_polygons=gdf_mission_polygons,
            flight_database_full=flight_database,
            # islands_gdf=islands_gdf[islands_gdf["nombre"] == island_name],
             islands_gdf=islands_gdf,
            output_path=output_path,
            title=f"Annotated images {dataset.dataset_name}, {dataset.dset}",
            inset_map = False,
            hA = hA
        )

        plt.show()
        plt.close(fig)



