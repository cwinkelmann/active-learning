"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
import gc

import copy
import csv
import geopandas as gpd
import pandas as pd
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.types.Exceptions import ProjectionError, NoLabelsError, AnnotationFileNotSetError, \
    OrthomosaicNotSetError, LabelsOverlapError
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.image_manipulation import convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords, project_gdfcrs
from active_learning.util.super_resolution import super_resolve, SuperResolution
from com.biospheredata.converter.HastyConverter import ImageFormat
from geospatial_transformations import get_gsd, get_geotiff_compression
from util.util import visualise_image, visualise_polygons

from osgeo import gdal
from collections import defaultdict
gdal.UseExceptions()

# Herdnet model


def save_tiles_to_csv(tiles, output_csv: Path, species="iguana", label=1):
    """
    Save raster tile filenames to a CSV file with additional metadata.

    Parameters:
        tiles (list): List of tile filenames (Paths or strings).
        output_csv (Path): Path to the output CSV file.
        species (str): Default species name (default: "iguana").
        label (int): Default label (default: 1).

    Returns:
        Path: Path to the saved CSV file.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Open CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["images", "x", "y", "species", "labels"])

        # Iterate through tiles and write rows
        for tile in tiles:
            filename = Path(tile).name  # Extract just the filename
            x, y = 0, 0  # Default placeholders (modify if coordinates exist)
            writer.writerow([filename, x, y, species, label])

    print(f"CSV saved to: {output_csv}")
    return output_csv


def geospatial_data_to_detection_training_data(annotations_file: Path,
                                               orthomosaic_path: Path,
                                               island_code: str,
                                               tile_folder_name: str,
                                               output_obj_dir: Path,
                                               output_empty_dir: Path,
                                               tile_output_dir: Path,
                                               tile_size: int,
                                               vis_output_dir: Path,
                                               visualise_crops: bool,
                                               format: ImageFormat,
                                                OBJECT_CENTERED= False
                                               ):
    """
    Convert geospatial annotations to create training data for herdnet out of geospatial dots
    :param annotations_file:
    :param orthomosaic_path:
    :param island_code:
    :param tile_folder_name:
    :param output_obj_dir:
    :param output_empty_dir:
    :param tile_size:
    :param vis_output_dir:
    :param visualise_crops:
    :param format:
    :return:
    """


    gdf_points = gpd.read_file(annotations_file)
    gdf_points["image_name"] = orthomosaic_path.name
    gdf_points.to_file(vis_output_dir / f"annotations_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

    if len(gdf_points) == 0:
        raise NoLabelsError(f"No labels found in {annotations_file}")


    # incase the orthomosaic has a different CRS than the annotations # TODO check if I really want to do this here
    gdf_points = project_gdfcrs(gdf_points, orthomosaic_path)
    # project the global coordinates to the local coordinates of the orthomosaic



    # TODO clean this up or make very ordinary hasty annotations out of it. This way the rest of the code would be kind of unnecessary
    gdf_local = convert_gdf_to_jpeg_coords(gdf_points, orthomosaic_path)
    # create an ImageCollection of the annotations

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

    grid_manager.gdf_raster_mask.to_file(filename= vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson", driver='GeoJSON')

    logger.info(f"Partitioning annotations into a balanced dataset")
    grid_gdf, gdf_empty_cells = grid_manager.create_balanced_dataset_grids(points_gdf=gdf_points,
                                                                           box_size_x=tile_size,
                                                                           box_size_y=tile_size,
                                                                           num_empty_samples=len(gdf_points),
                                                                           object_centered=OBJECT_CENTERED,
                                                                           min_distance_pixels=700,
                                                                           overlap_ratio=0.0)

    grid_gdf.to_file(vis_output_dir / f"grid_all_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

    # grid = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0.0)
    # grid_gdf, gdf_empty_cells = grid_manager.filter_by_points(gdf_points)

    # grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(orthomosaic_path),
    #                                                  x_size=tile_size,
    #                                                  y_size=tile_size,
    #                                                  overlap_ratio=0.0)
    logger.info(f"Done creating polygons for cutouts")
    gdf_empty_cells.to_file(vis_output_dir / f"grid_empty_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)
    # remove grid cells which don't contain points which are saved in gdf
    grid_gdf_filtered = grid_gdf[grid_gdf.geometry.apply(lambda poly: gdf_points.geometry.within(poly).any())]
    try:
        grid_gdf_filtered.to_file(vis_output_dir / f"grid_occupied_{orthomosaic_path.stem}.geojson", driver='GeoJSON')
    except Exception:
        pass

    # grid_gdf_filtered = grid_gdf_filtered[:1] ### TODO remove this

    logger.info(f"start slicing: {orthomosaic_path.stem}")
    slicer_occupied = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf_filtered,
                       output_dir=tile_output_dir)

    gdf_sliced_points = slicer_occupied.slice_annotations_regular_grid(gdf_points, grid_gdf_filtered)
    # for tile_name, gdf_group in gdf_sliced_points.groupby(by="tile_name"):
    #     gdf_group.to_file(vis_output_dir / f"{tile_name}.geojson", driver='GeoJSON')

    slicer_empty = GeoSlicer(base_path=orthomosaic_path.parent,
                             image_name=orthomosaic_path.name,
                             grid=gdf_empty_cells,
                             output_dir=tile_output_dir)

    logger.info(f"cut empty tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    slicer_empty.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=3)
    logger.info(f"cut occupied tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=3)

    converted_empty_tiles = convert_tiles_to(tiles=list(slicer_empty.gdf_slices.slice_path),
                                             format=format,
                                             output_dir=output_empty_dir)

    converted_tiles = convert_tiles_to(tiles=list(slicer_occupied.gdf_slices.slice_path), format=format,
                                       output_dir=output_obj_dir)

    converted_tiles = [a for a in converted_tiles]
    converted_empty_tiles = [a for a in converted_empty_tiles]

    # ### TODO create herdnet annotations for each tile

    vis_path = vis_output_dir / f"visualisations"
    vis_path.mkdir(exist_ok=True, parents=True)
    l_herdnet = []


    for tile_name, gdf_group in gdf_sliced_points.groupby(by="tile_name"):
        # just to check if that actually works we take the file not the gdf
        annotations_file = output_object_dir / f"{tile_name}.csv"
        df_herdnet = copy.copy(gdf_group[["tile_name", "local_pixel_x", "local_pixel_y"]])
        df_herdnet.loc[:, "species"] = "iguana"
        df_herdnet.loc[:, "labels"] = 1

        tile_image_name = f"{tile_name}.jpg"
        assert output_object_dir.joinpath(
            tile_image_name).exists(), f"{tile_image_name} does not exist"
        filename = output_obj_dir / f"{tile_name}.jpg"
        df_herdnet.loc[:, "tile_name"] = filename
        l_herdnet.append(df_herdnet)



        if visualise_crops:
            vis_filename = vis_output_dir / f"{tile_name}.jpg"
            logger.info(f"Visualising {tile_name}")
            ax_s = visualise_image(image_path=output_object_dir / tile_image_name, show=False,
                                   title=f"Visualisation of {len(df_herdnet)} labels in {tile_image_name}")
            visualise_polygons(
                points=[shapely.Point(x, y) for x, y in zip(df_herdnet.local_pixel_x, df_herdnet.local_pixel_y)],
                labels=df_herdnet["species"], ax=ax_s, show=False, linewidth=6, markersize=10,
                filename=vis_filename)

            plt.close()

    df_herdnet = pd.concat(l_herdnet, axis=0)
    return df_herdnet



# TODO maybe use the group_nearby_polygons_simple function to get the groups

def get_training_data_stats(gdf_mapping):
    stats = {}

    stats["total_orthomosaics"] = len(gdf_mapping)
    stats["total_labels"] = gdf_mapping["number_of_iguanas_shp"].sum()
    stats["labels_per_island"] = gdf_mapping.groupby('island')['number_of_iguanas_shp'].sum()
    stats["area_per_island"] = gdf_mapping.groupby('island')['area'].sum()
    stats["gsd_per_island"] = gdf_mapping.groupby('island')['gsd'].mean()

    print(f"stats: {stats}")

    return stats


def check_merged_split_overlaps(gdf_train, gdf_val, gdf_test):
    """
    Merge geometries per split and check for overlaps
    """
    # Merge all geometries per split into single polygons
    train_merged = gdf_train.geometry.unary_union
    val_merged = gdf_val.geometry.unary_union
    test_merged = gdf_test.geometry.unary_union

    # Check overlaps
    train_val_overlap = train_merged.intersects(val_merged)
    train_test_overlap = train_merged.intersects(test_merged)
    val_test_overlap = val_merged.intersects(test_merged)

    # Print results
    if any([train_val_overlap, train_test_overlap, val_test_overlap]):
        logger.error("❌ OVERLAPS DETECTED:")
        if train_val_overlap:
            overlap_area = train_merged.intersection(val_merged).area
            logger.error(f"  - Train vs Val: {overlap_area:.2f} area units")
        if train_test_overlap:
            overlap_area = train_merged.intersection(test_merged).area
            logger.error(f"  - Train vs Test: {overlap_area:.2f} area units")
        if val_test_overlap:
            overlap_area = val_merged.intersection(test_merged).area
            logger.error(f"  - Val vs Test: {overlap_area:.2f} area units")
        raise LabelsOverlapError
    else:
        logger.info("✓ No overlaps detected between splits")
        return True




if __name__ == "__main__":

    # train_orthophotos = ["Scris_SRL12_10012021"]
    # val_orthophotos = ["Mar_MBBE05_09122021"]
    # test_orthophotos = ["Fer_FNE01_19122021"]
    #
    #
    # train_orthophotos = [, ]
    # val_orthophotos = ["Flo_FLPC05_22012021",]
    train_orthophotos = ["Flo_FLPC07_22012021", "Flo_FLPC06_22012021", "Flo_FLPC05_22012021", "Flo_FLPC04_22012021", "Flo_FLPC01_22012021", "Flo_FLPC03_22012021"]




    data_splits = {
        "train": train_orthophotos,
        # "val": val_orthophotos,
        # "test": test_orthophotos
    }
    orthomosaic_to_split = {
        orthomosaic: split_name
        for split_name, orthomosaics in data_splits.items()
        for orthomosaic in orthomosaics
    }

    resolution = 512
    scale_factor = 1
    visualise_crops = True
    OBJECT_CENTERED = False  # If True, the crops are centered around the object, otherwise they are centered around the tile
    problematic_data_pairs = []
    herdnet_annotations = defaultdict(list)
    format = ImageFormat.JPG

    # See 043_reorganise_shapefiles for the creation of this file
    usable_training_data_raster_mask = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/usable_training_data_raster_mask_with_group.geojson")

    gdf_mapping = gpd.read_file(usable_training_data_raster_mask)

    get_training_data_stats(gdf_mapping)

    # add column split to dataframe
    gdf_mapping["split"] = gdf_mapping["Orthophoto/Panorama name"].map(orthomosaic_to_split)

    # sanity check if any split overlaps another split
    gdf_train = gdf_mapping[gdf_mapping["split"] == "train"]
    gdf_val = gdf_mapping[gdf_mapping["split"] == "val"]
    gdf_test = gdf_mapping[gdf_mapping["split"] == "test"]
    # Run the check
    no_overlaps = check_merged_split_overlaps(gdf_train, gdf_val, gdf_test)


    tile_size = resolution // scale_factor


    base_path = Path(f"/Volumes/2TB/DD_MS_COG_Prepared_Training_2025_06_04_{tile_size}_obcj_{OBJECT_CENTERED}")

    for index, row in gdf_mapping.iterrows():
        print(f"Processing idx {index} out of {len(gdf_mapping)}")

        annotations_file = Path(row["shp_file_path"])
        orthomosaic_path = Path(row["images_path"])

        if not row["Orthophoto/Panorama name"] in orthomosaic_to_split:
            logger.info(f"Skipping {orthomosaic_path.name} as it is not in the orthomosaic_to_split mapping")
            continue

        # get the split
        split = row["split"]

        output_object_dir = base_path / "herdnet" / f"{split}/iguana"
        output_empty_dir = base_path / "herdnet" / f"{split}/empty"


        vis_output_dir = base_path / "visualisation" / f"{split}"
        tile_output_dir = base_path / "tiff_tiles" / f"{split}"

        output_object_dir.mkdir(parents=True, exist_ok=True)

        output_empty_dir.mkdir(parents=True, exist_ok=True)
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        tile_output_dir.mkdir(parents=True, exist_ok=True)

        if row["Orthophoto/Panorama quality"] == "Bad":
            logger.warning(f"This orthomosaic is of bad quality: {row['Orthophoto/Panorama name']}")


        island_code = row["island_code"]
        logger.info(f"Processing {orthomosaic_path.name}")

        tile_folder_name = orthomosaic_path.stem

        herdnet_annotation = geospatial_data_to_detection_training_data(annotations_file=annotations_file,
                                                                        orthomosaic_path=orthomosaic_path,
                                                                        island_code=island_code,
                                                                        tile_folder_name=tile_folder_name,
                                                                        output_obj_dir=output_object_dir,
                                                                        output_empty_dir=output_empty_dir,
                                                                        vis_output_dir=vis_output_dir,
                                                                        tile_output_dir=tile_output_dir,
                                                                        tile_size=tile_size,
                                                                        visualise_crops=visualise_crops,
                                                                        format=format,
                                                                        OBJECT_CENTERED=OBJECT_CENTERED
                                                                        )

        logger.info(f"Done with {orthomosaic_path.name}")


        herdnet_annotations[split].append(herdnet_annotation)

        gc.collect()




    for split, annotations in herdnet_annotations.items():
        # Save the herdnet annotations to a CSV filep
        combined_df = pd.concat(herdnet_annotations[split], ignore_index=True)

        output_annotation_dir = base_path / "herdnet" / f"{split}"
        combined_df.to_csv(
            output_annotation_dir / f"herdnet_annotations_{split}.csv", index=False)


