"""
Creates an overall Galápagos Islands map, then add the training data of orhomosaic origin

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
    draw_accurate_scalebar, get_geographic_ticks
from active_learning.util.mapping.plots import plot_orthomomsaic_training_data
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.status import ImageFormat
from geospatial_transformations import get_geotiff_compression, get_gsd


def geospatial_training_data(annotations_file: Path,
                             orthomosaic_path: Path,
                             vis_output_dir: Path,
                             ) -> typing.Tuple[Path, gpd.GeoDataFrame, gpd.GeoDataFrame, float]:
    """
    Convert geospatial annotations to create training data for herdnet out of geospatial dots
    :param annotations_file:
    :param orthomosaic_path:
    :param island_code:
    :param tile_folder_name:
    :param output_dir:
    :param output_empty_dir:
    :param tile_size:
    :param vis_output_dir:
    :param visualise_crops:
    :param format:
    :return:
    """


    gdf_points = gpd.read_file(annotations_file)
    gdf_points["image_name"] = orthomosaic_path.name



    # incase the orthomosaic has a different CRS than the annotations # TODO check if I really want to do this here
    gdf_points = project_gdfcrs(gdf_points, orthomosaic_path)
    # project the global coordinates to the local coordinates of the orthomosaic


    # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
    cog_compression = get_geotiff_compression(orthomosaic_path)
    logger.info(f"COG compression: {cog_compression}")
    gsd_x, gsd_y = get_gsd(orthomosaic_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning(
            "You are either a precise pilot or you wasted quality by using 'DroneDeploy', which caps Orthophoto GSD at about 0.93cm/px, compresses images a lot and throws away details")

    # TODO make sure the CRS is the for both

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))
    raster_mask_path = vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson"
    grid_manager.gdf_raster_mask.to_file(filename=raster_mask_path , driver='GeoJSON')


    return raster_mask_path, grid_manager.gdf_raster_mask, gdf_points, gsd_x



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
    gdf_flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=EPSG_WGS84)

    # How many unique flights (mission_folder) are in the flight database?
    amount_missions = gdf_flight_database["mission_folder"].nunique()
    logger.info(f"Amount of missions: {amount_missions}")
    stats_collection["amount_missions"] = amount_missions


    # See 043_reorganise_shapefiles for the creation of this file
    orthomosaic_shapefile_mapping_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Geospatial_Annotations/enriched_GIS_progress_report_with_stats.csv")
    df_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)

    df_mapping = df_mapping[df_mapping["terrain"] == "flat"]
    df_mapping["parsed_date"] = pd.to_datetime(df_mapping["Date"], format='%d.%m.%Y', errors='coerce')

    stats_collection["supposedly_labeled_orthomsaics"] = len(df_mapping)
    # which user counted on how many orthomosaics?
    stats_collection["count_by_user"] = df_mapping["Expert"].value_counts().to_dict()
    stats_collection["count_by_stitching_software"] = df_mapping["Orthophoto/Panorama"].value_counts().to_dict()



    for index, row in df_mapping.iterrows():
        try:
            quality = row["Orthophoto/Panorama quality"]
            if quality == "Bad":
                logger.warning(f"This orthomosaic is of bad quality: {row['Orthophoto/Panorama name']}")

            HasAgisoftOrthomosaic = row["HasAgisoftOrthomosaic"]
            HasDroneDeployOrthomosaic = row["HasDroneDeployOrthomosaic"]
            HasShapefile = row["HasShapefile"]
            annotations_file = row["shp_file_path"]


            if HasShapefile:
                try:
                    # replace base path with the new path
                    annotations_file = annotations_file.replace(
                        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations",
                        "/Volumes/2TB/Manual_Counting/Geospatial_Annotations")
                    annotations_file = Path(annotations_file)

                    row["shp_file_path"] = annotations_file


                except Exception as e:
                    raise AnnotationFileNotSetError(f"Could not set annotations file: {annotations_file}")
            else:
                raise AnnotationFileNotSetError(f"Could not set annotations file, because it is None")

            if HasAgisoftOrthomosaic or HasDroneDeployOrthomosaic:
                orthomosaic_path = row["images_path"]
                # raplace base path with the new path
                orthomosaic_path = orthomosaic_path.replace(
                    "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/",
                    "/Volumes/2TB/Manual_Counting/")
                orthomosaic_path = Path(orthomosaic_path)

                row["images_path"] = orthomosaic_path
            else:
                raise OrthomosaicNotSetError(f"No Orthomosaic found for {row['Orthophoto/Panorama name']}")

            island_code = row["island_code"]
            logger.info(f"Processing {orthomosaic_path.name}")

            # if not orthomosaic_path.name == "Esp_EGB02_12012021.tif":
            #     continue

            # island_code = orthomosaic_path.parts[-2]
            tile_folder_name = orthomosaic_path.stem


            raster_mask_path, gdfraster_mask, gdf_annotations, gsd = geospatial_training_data(annotations_file=annotations_file,
                                                                                              orthomosaic_path=orthomosaic_path,
                                                                                              vis_output_dir=vis_output_dir,
                                                                                              )

            gdfraster_mask = gdfraster_mask.to_crs(epsg=EPSG_WGS84)
            gdf_annotations = gdf_annotations.to_crs(epsg=EPSG_WGS84)

            usable_training_data_row = row.copy()
            usable_training_data_row["orthomosaic_path"] = orthomosaic_path


            usable_training_data_row["annotations_file"] = annotations_file


            usable_training_data_row["raster_mask"] = unary_union(gdfraster_mask.geometry).iloc[0]
            usable_training_data_row["area"] = gdfraster_mask.area.sum()
            usable_training_data_row["gsd"] = gsd
            usable_training_data_row["source_crs"] = gdfraster_mask.crs.to_epsg()

            if not gdf_annotations.empty:
                gdf_annotations = gdf_annotations.assign(**usable_training_data_row[["Orthophoto/Panorama name","gsd"]].to_dict())
                usable_training_annotations.append(gdf_annotations)

            if int(row["Number of iguanas"]) != int(row["number_of_iguanas_shp"]):
                raise NotEnoughLabelsError(f"Not enough labels found in {annotations_file}")

            usable_training_data.append(usable_training_data_row)

        except ProjectionError:
            row["reason"] = "ProjectionError"
            logger.error(f"ProjectionError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)
        except KeyError:
            row["reason"] = "KeyError"
            logger.error(f"KeyError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)
        except NoLabelsError:
            row["reason"] = "NoLabelsError"
            logger.error(f"NoLabelsError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)
        except NotEnoughLabelsError:
            row["reason"] = "NotEnoughLabelsError"
            logger.error(f"NotEnoughLabelsError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)
        except AnnotationFileNotSetError:
            row["reason"] = "AnnotationFileNotSetError"
            logger.error(f"AnnotationFileNotSetError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)
        except OrthomosaicNotSetError:
            row["reason"] = "OrthomosaicNotSetError"
            logger.error(f"OrthomosaicNotSetError: {row['Orthophoto/Panorama name']}")
            problematic_data_pairs.append(row)


    pd.DataFrame([dict(record) for record in problematic_data_pairs]).to_csv("problematic_data_pairs.csv", index=False)
    df_usable_training_data = pd.DataFrame(usable_training_data)
    df_usable_training_data.rename(columns={"raster_mask": "geometry"}, inplace=True)
    df_usable_training_annotations = pd.concat(usable_training_annotations, axis=0, ignore_index=True)

    df_usable_training_data.to_csv("usable_training_data.csv", index=False)

    stats_collection["usable_training_data"] = len(df_usable_training_data)

    gdf_usable_training_data_raster_mask = gpd.GeoDataFrame(df_usable_training_data, geometry="geometry", crs=EPSG_WGS84)
    gdf_usable_training_annotations = gpd.GeoDataFrame(df_usable_training_annotations, geometry="geometry", crs=EPSG_WGS84)

    gdf_usable_training_data_raster_mask.to_file(filename="usable_training_data_raster_mask.geojson", driver="GeoJSON")
    gdf_usable_training_annotations.to_file(filename="usable_training_data_annotations.geojson", driver="GeoJSON")



    print(f"stats_collection: {stats_collection}")
    with open("stats_collection.json", "w") as f:
        import json
        json.dump(stats_collection, f, indent=4)

    # Visualize the groups
    import matplotlib.pyplot as plt


