"""
Use this code to convert geospatial data into training data for object detection tasks, specifically for HerdNet.

"""
import copy
import pandas as pd
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from osgeo import gdal
from pathlib import Path

from active_learning.types.Exceptions import LabelsOverlapError
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.geospatial_transformations import get_geotiff_compression, get_gsd
from active_learning.util.image_manipulation import convert_tiles_to
from active_learning.util.projection import convert_gdf_to_jpeg_coords, project_gdfcrs
from com.biospheredata.types.status import ImageFormat
from image_template_search.util.util import visualise_image, visualise_polygons

gdal.UseExceptions()


def geospatial_data_to_detection_training_data(
        gdf_points,
        orthomosaic_path: Path,
        output_obj_dir: Path,
        output_empty_dir: Path,
        tile_output_dir: Path,
        tile_size: int,
        vis_output_dir: Path,
        visualise_crops: bool,
        format: ImageFormat,
        OBJECT_CENTERED=False,
        sample_fraction=1.0,
        overlap_ratio=0.0,
):
    """
    Convert geospatial annotations to create training data for herdnet out of geospatial dots
    :param sample_fraction:
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


    # gdf_points = gpd.read_file(annotations_file)
    # if len(gdf_points) == 0:
    #     raise NoLabelsError(f"No labels found in {annotations_file}")

    gdf_points["image_name"] = orthomosaic_path.name
    gdf_points.to_file(vis_output_dir / f"annotations_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

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

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_manager.gdf_raster_mask.to_file(filename= vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson", driver='GeoJSON')
    logger.info(f"Raster mask saved to {vis_output_dir / f'raster_mask_{orthomosaic_path.stem}.geojson'}")

    logger.info(f"Partitioning annotations into a balanced dataset")
    grid_gdf, gdf_empty_cells = grid_manager.create_balanced_dataset_grids(points_gdf=gdf_points,
                                                                           box_size_x=tile_size,
                                                                           box_size_y=tile_size,
                                                                           num_empty_samples=len(gdf_points) * sample_fraction,
                                                                           object_centered=OBJECT_CENTERED,
                                                                           min_distance_pixels=500,
                                                                           overlap_ratio=overlap_ratio)

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
    slicer_empty.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=5)
    logger.info(f"cut occupied tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=5)

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
        annotations_file = output_obj_dir / f"{tile_name}.csv"
        df_herdnet = copy.copy(gdf_group[["tile_name", "local_pixel_x", "local_pixel_y", "species", "labels"]])
        #df_herdnet.loc[:, "species"] = "iguana"
        #df_herdnet.loc[:, "labels"] = 1

        tile_image_name = f"{tile_name}.jpg"
        assert output_obj_dir.joinpath(
            tile_image_name).exists(), f"{tile_image_name} does not exist"
        filename = output_obj_dir / f"{tile_name}.jpg"
        df_herdnet.loc[:, "tile_name"] = filename
        l_herdnet.append(df_herdnet)



        if visualise_crops:
            vis_filename = vis_output_dir / f"{tile_name}.jpg"
            logger.info(f"Visualising {tile_name}")
            ax_s = visualise_image(image_path=output_obj_dir / tile_image_name, show=False,
                                   title=f"Visualisation of {len(df_herdnet)} labels in {tile_image_name}")
            visualise_polygons(
                points=[shapely.Point(x, y) for x, y in zip(df_herdnet.local_pixel_x, df_herdnet.local_pixel_y)],
                labels=df_herdnet["species"], ax=ax_s, show=False, linewidth=6, markersize=10,
                filename=vis_filename)

            plt.close()

    df_herdnet = pd.concat(l_herdnet, axis=0)


    return df_herdnet


def geospatial_data_to_detection_training_data_with_hard_neg(
        gdf_points_objects,
        gdf_points_hard_neg,
        orthomosaic_path: Path,
        output_obj_dir: Path,
        output_empty_dir: Path,
        tile_output_dir: Path,
        tile_size: int,
        vis_output_dir: Path,
        visualise_crops: bool,
        format: ImageFormat,
        OBJECT_CENTERED=False,
        sample_fraction=1.0,
        overlap_ratio=0.0,
):
    """
    Convert geospatial annotations to create training data for herdnet out of geospatial dots
    :param sample_fraction:
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


    # gdf_points = gpd.read_file(annotations_file)
    # if len(gdf_points) == 0:
    #     raise NoLabelsError(f"No labels found in {annotations_file}")

    gdf_points_objects["image_name"] = orthomosaic_path.name
    gdf_points_objects.to_file(vis_output_dir / f"annotations_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

    # incase the orthomosaic has a different CRS than the annotations # TODO check if I really want to do this here
    gdf_points_objects = project_gdfcrs(gdf_points_objects, orthomosaic_path)
    # project the global coordinates to the local coordinates of the orthomosaic

    # TODO clean this up or make very ordinary hasty annotations out of it. This way the rest of the code would be kind of unnecessary
    gdf_local = convert_gdf_to_jpeg_coords(gdf_points_objects, orthomosaic_path)
    # create an ImageCollection of the annotations

    # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
    cog_compression = get_geotiff_compression(orthomosaic_path)
    logger.info(f"COG compression: {cog_compression}")
    gsd_x, gsd_y = get_gsd(orthomosaic_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning(
            "You are either a precise pilot or you wasted quality by using 'DroneDeploy', which caps Orthophoto GSD at about 0.93cm/px, compresses images a lot and throws away details")

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_manager.gdf_raster_mask.to_file(filename= vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson", driver='GeoJSON')

    logger.info(f"Partitioning annotations into a balanced dataset")
    gdf_object_grid = grid_manager.object_centered_grid(points_gdf=gdf_points_objects,
                                                                           box_size_x=tile_size,
                                                                           box_size_y=tile_size,
                                                                           )

    gdf_empty_cells = grid_manager.object_centered_grid(points_gdf=gdf_points_hard_neg,
                                                                           box_size_x=tile_size,
                                                                           box_size_y=tile_size,
                                                                           )

    gdf_object_grid.to_file(vis_output_dir / f"grid_occupied_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)
    gdf_empty_cells.to_file(vis_output_dir / f"grid_hard_neg_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)


    # grid_gdf_filtered = grid_gdf_filtered[:1] ### TODO remove this

    logger.info(f"start slicing: {orthomosaic_path.stem}")
    slicer_occupied = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=gdf_object_grid,
                       output_dir=tile_output_dir)

    gdf_sliced_objects_points = slicer_occupied.slice_annotations_regular_grid(gdf_points_objects, gdf_object_grid)


    slicer_empty = GeoSlicer(base_path=orthomosaic_path.parent,
                             image_name=orthomosaic_path.name,
                             grid=gdf_empty_cells,
                             output_dir=tile_output_dir)

    logger.info(f"cut empty tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    slicer_empty.slice_very_big_raster(num_chunks=len(gdf_points_objects) // 20 + 1, num_workers=3)
    logger.info(f"cut occupied tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_points_objects) // 20 + 1, num_workers=3)

    converted_empty_tiles = convert_tiles_to(tiles=list(slicer_empty.gdf_slices.slice_path),
                                             format=format,
                                             output_dir=output_empty_dir)

    converted_tiles = convert_tiles_to(tiles=list(slicer_occupied.gdf_slices.slice_path),
                                       format=format,
                                       output_dir=output_obj_dir)

    converted_tiles = [a for a in converted_tiles]
    converted_empty_tiles = [a for a in converted_empty_tiles]

    # ### TODO create herdnet annotations for each tile

    vis_path = vis_output_dir / f"visualisations"
    vis_path.mkdir(exist_ok=True, parents=True)
    l_herdnet = []


    for tile_name, gdf_objects_group in gdf_sliced_objects_points.groupby(by="tile_name"):
        # just to check if that actually works we take the file not the gdf
        annotations_file = output_obj_dir / f"{tile_name}.csv"
        df_herdnet = copy.copy(gdf_objects_group[["tile_name", "local_pixel_x", "local_pixel_y", "species", "labels", "scores"]])


        tile_image_name = f"{tile_name}.jpg"
        assert output_obj_dir.joinpath(
            tile_image_name).exists(), f"{tile_image_name} does not exist"
        filename = output_obj_dir / f"{tile_name}.jpg"
        df_herdnet.loc[:, "tile_name"] = filename
        l_herdnet.append(df_herdnet)



        if visualise_crops:
            vis_filename = vis_output_dir / f"{tile_name}.jpg"
            logger.info(f"Visualising {tile_name}")
            ax_s = visualise_image(image_path=output_obj_dir / tile_image_name, show=False,
                                   title=f"Visualisation of {len(df_herdnet)} labels in {tile_image_name}")
            visualise_polygons(
                points=[shapely.Point(x, y) for x, y in zip(df_herdnet.local_pixel_x, df_herdnet.local_pixel_y)],
                labels=df_herdnet["species"], ax=ax_s, show=False, linewidth=6, markersize=10,
                filename=vis_filename)

            plt.close()

    # TODO check if everything is correct here
    #
    df_herdnet = pd.concat(l_herdnet, axis=0)
    df_herdnet.rename(columns={"local_pixel_x": "x",
                               "local_pixel_y": "y",
                               "tile_name": "images"}, inplace=True)

    # reduce the full path in "images" to just the filename
    df_herdnet["images"] = df_herdnet["images"].apply(lambda x: Path(x).name)


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