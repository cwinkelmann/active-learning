"""
Creates an overall Galápagos Islands map add the training data of orhomosaic origin

See 044_prepare_orthomosaic_classifiation for the complete pipeline
"""
import geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from matplotlib_map_utils.core.north_arrow import NorthArrow
from pathlib import Path
from shapely.geometry import box

from active_learning.database import images_data_extraction, derive_image_metadata
from active_learning.types.Exceptions import NoLabelsError, AnnotationFileNotSetError, ProjectionError, \
    OrthomosaicNotSetError
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid
from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.converter.HastyConverter import ImageFormat
from geospatial_transformations import get_geotiff_compression, get_gsd

web_mercator_projection_epsg = 3857

def geospation_training_data(annotations_file: Path,
                                                    orthomosaic_path: Path,
                                                    vis_output_dir: Path,
                                                    ):
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

    if len(gdf_points) == 0:
        raise NoLabelsError(f"No labels found in {annotations_file}")


    # incase the orthomosaic has a different CRS than the annotations # TODO check if I really want to do this here
    gdf_points = project_gdfcrs(gdf_points, orthomosaic_path)
    # project the global coordinates to the local coordinates of the orthomosaic


    # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
    cog_compression = get_geotiff_compression(orthomosaic_path)
    logger.info(f"COG compression: {cog_compression}")
    gsd_x, gsd_y = get_gsd(orthomosaic_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning(
            "You are either a precise pilot or you wasted quality by using drone deploy, which caps images at about 0.93cm/px, compresses images a lot throws away details")

    # TODO make sure the CRS is the for both

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))
    raster_mask_path = vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson"
    grid_manager.gdf_raster_mask.to_file(filename=raster_mask_path , driver='GeoJSON')


    return grid_manager.gdf_raster_mask, gdf_points, gsd_x



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

    # See 043_reorganise_shapefiles for the creation of this file
    orthomosaic_shapefile_mapping_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Geospatial_Annotations/enriched_GIS_progress_report_with_stats.csv")
    df_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)


    analysis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_analysis/")
    vis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/visualisation")

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    herdnet_annotations = []
    problematic_data_pairs = []

    for index, row in df_mapping.iterrows():
        print(f"Processing {index}")
        try:
            quality = row["Orthophoto/Panorama quality"]
            if quality == "Bad":
                logger.warning(f"This orthomosaic is of bad quality: {row}")

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
            else:
                raise OrthomosaicNotSetError(f"No Orthomosaic found for {row['Orthophoto/Panorama name']}")

            island_code = row["island_code"]
            logger.info(f"Processing {orthomosaic_path.name}")

            # if not orthomosaic_path.name == "Esp_EGB02_12012021.tif":
            #     continue

            # island_code = orthomosaic_path.parts[-2]
            tile_folder_name = orthomosaic_path.stem


            raster_mask, annotations, gsd = geospation_training_data(annotations_file=annotations_file,
                                                                                 orthomosaic_path=orthomosaic_path,
                                                                                 vis_output_dir=vis_output_dir,
                                                                                 )
        except ProjectionError:
            row["reason"] = "ProjectionError"
            problematic_data_pairs.append(row)
            logger.error(f"ProjectionError: {row}")
        except KeyError:
            row["reason"] = "KeyError"
            logger.error(f"KeyError: {row}")
            problematic_data_pairs.append(row)
        except NoLabelsError:
            row["reason"] = "NoLabelsError"
            logger.error(f"KeyError: {row}")
            problematic_data_pairs.append(row)
        except AnnotationFileNotSetError:
            row["reason"] = "AnnotationFileNotSetError"
            logger.error(f"AnnotationFileNotSetError: {row}")
            problematic_data_pairs.append(row)
        except OrthomosaicNotSetError:
            row["reason"] = "OrthomosaicNotSetError"
            logger.error(f"OrthomosaicNotSetError: {row}")
            problematic_data_pairs.append(row)

    # TODO check how many of the rasters contain points



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