import fiftyone as fo
import geopandas as gpd
import hashlib
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_geospatial_single_image
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig, \
    GeospatialDatasetCorrectionConfigCollection
from active_learning.types.Exceptions import NoLabelsError, NullRow, ProjectionError, DatasetsExistsError
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.geospatial_transformations import get_geotiff_compression, get_gsd
from active_learning.util.image_manipulation import convert_tiles_to
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.status import ImageFormat
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def verfify_points(gdf_points, config: GeospatialDatasetCorrectionConfig):
    """
    Execute some sanity checks on the points and project them to the local CRS of the orthomosaic
    :param gdf_points:
    :param config:
    :return:
    """
    if len(gdf_points) > 0:

        gdf_points.drop(columns=["images", "x", "y"], inplace=True, errors='ignore')
        gdf_points = project_gdfcrs(gdf_points, config.image_path)
        # project the global coordinates to the local coordinates of the orthomosaic

        if "scores" not in gdf_points.columns:
            gdf_points["scores"] = None
        if "species" not in gdf_points.columns:
            gdf_points["species"] = "iguana_point"
        if "labels" not in gdf_points.columns:
            gdf_points["labels"] = 1

        # TODO clean this up or make very ordinary hasty annotations out of it. This way the rest of the code would be kind of unnecessary
        # gdf_local = convert_gdf_to_jpeg_coords(gdf_points, config.image_path)
        # create an ImageCollection of the annotations

        # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
        cog_compression = get_geotiff_compression(config.image_path)
        logger.info(f"COG compression: {cog_compression}")
        gsd_x, gsd_y = get_gsd(config.image_path)
        if round(gsd_x, 4) == 0.0093:
            logger.warning(
                "You are either a precise pilot or you wasted quality by using 'DroneDeploy', which caps Orthophoto GSD at about 0.93cm/px, compresses images a lot and throws away details")

        logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")

        # remove rows where species is empty or nan
        gdf_points = gdf_points[gdf_points['species'].notna() & (gdf_points['species'] != '')]

    return gdf_points


def batched_main(configs: GeospatialDatasetCorrectionConfigCollection,
                 output_dir: Path,
                 vis_output_dir: Path,
                 geospatial_flag=True,
                 submit_to_CVAT=False,
                 include_reference=False,
                 delete_dataset_if_exists=False,
                 radius=0.3,
                    cvat_upload = False
                 ):
    """
    Main function for geospatial human in the loop correction of batches
    :param configs:
    :param output_dir:
    :param vis_output_dir:
    :param geospatial_flag:
    :param submit_to_CVAT:
    :param include_reference:
    :return:
    """

    tile_output_dir = output_dir / "tiles"
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    format = ImageFormat.JPG


    for config in configs.configs:
        logger.info(f"Processing config: {config.image_path.name}")
        if submit_to_CVAT:
            try:
                # TODO manage the datasets better!
                if fo.dataset_exists(config.dataset_name) and delete_dataset_if_exists:
                    logger.warning(f"Deleting existing dataset {config.dataset_name}")
                    fo.delete_dataset(config.dataset_name)
                else:
                    logger.warning(f"FiftyOne Dataset already exists: {config.dataset_name}")
            except:
                logger.warning(f"Dataset {config.dataset_name} does not exist")

            try:
                # Create an empty dataset, TODO put this away so the dataset is just passed into this
                dataset = fo.Dataset(name=config.dataset_name)
                dataset.persistent = True
            except:
                logger.info(f"Dataet already exists, skipping thre rest")
                continue
        else:
            logger.info("FiftyOne and CVAT submission disabled, working locally only")

        assert isinstance(config, GeospatialDatasetCorrectionConfig)

        # # TODO extract this
        def preppare_online_checkup(config: GeospatialDatasetCorrectionConfig, species="iguana_point"):
            hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)
            config.image_tiles_path = tile_output_dir

            if geospatial_flag:
                gdf_prediction_points = gpd.read_file(config.geojson_prediction_path)

                gdf_prediction_points = verfify_points(gdf_prediction_points, config)
                gdf_prediction_points["species"] = species  # TODO that is a bit of a hack, but whatever

                gdf_prediction_points.to_file(
                    filename=vis_output_dir / f"predictions_a_{config.image_path.stem}.geojson",
                    driver='GeoJSON')

                # There are two modes:
                # 1. include the reference data into the evaluation, i.e. human vs ai or ai vs ai or even human vs human
                # 2. just submit the first prediction to CVAT
                if include_reference:
                    gdf_reference_points = gpd.read_file(config.geojson_reference_annotation_path)
                    gdf_reference_points = verfify_points(gdf_reference_points, config)
                    gdf_reference_points["species"] = species

                    gdf_reference_points.to_file(
                        filename=vis_output_dir / f"predictions_b_{config.image_path.stem}.geojson",
                        driver='GeoJSON')

                    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_geospatial_single_image(
                        gdf_detections=gdf_prediction_points,
                        gdf_ground_truth=gdf_reference_points,
                        radius_m=radius,
                    )
                    try:
                        gdf_prediction_points = pd.concat([df_false_positives, df_true_positives, df_false_negatives],
                                                      axis=0, ignore_index=True)
                    except ValueError:
                        raise ProjectionError(f"Projection error between {config.geojson_prediction_path} and {config.geojson_reference_annotation_path}, please check if both files are in the same projection")
                    assert isinstance(gdf_prediction_points, gpd.GeoDataFrame)

                grid_manager = GeoSpatialRasterGrid(Path(config.image_path))

                grid_manager.gdf_raster_mask.to_file(
                    filename=vis_output_dir / f"raster_mask_{config.image_path.stem}.geojson",
                    driver='GeoJSON')
                logger.info(f"Raster mask saved to {vis_output_dir / f'raster_mask_{config.image_path.stem}.geojson'}")

                grid_gdf = grid_manager.create_filtered_grid(points_gdf=gdf_prediction_points,
                                                             box_size_x=config.box_size_x,
                                                             box_size_y=config.box_size_y,
                                                             num_empty_samples=len(gdf_prediction_points) * 0.0,
                                                             object_centered=False,
                                                             min_distance_pixels=0,
                                                             overlap_ratio=0.0)

                grid_gdf_filtered = grid_gdf[
                    grid_gdf.geometry.apply(lambda poly: gdf_prediction_points.geometry.within(poly).any())]

                logger.info(f"start slicing: {config.image_path.stem}")
                slicer_occupied = GeoSlicer(base_path=config.image_path.parent,
                                            image_name=config.image_path.name,
                                            grid=grid_gdf_filtered,
                                            output_dir=tile_output_dir)

                grid_gdf_filtered.to_file(
                    filename=vis_output_dir / f"occupied_grid_{config.image_path.stem}.geojson",
                    driver='GeoJSON')


                gdf_sliced_points = slicer_occupied.slice_annotations_regular_grid(gdf_prediction_points,
                                                                                   grid_gdf_filtered)
                gdf_sliced_points
                occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_prediction_points) // 20 + 1,
                                                                       num_workers=5)

                converted_tile_output_dir = tile_output_dir / "converted_tiles"
                converted_tiles = convert_tiles_to(tiles=list(slicer_occupied.gdf_slices.slice_path),
                                                   format=format,
                                                   output_dir=converted_tile_output_dir, )
                converted_tiles = [a for a in converted_tiles]

                if submit_to_CVAT:
                    samples = [fo.Sample(filepath=path) for path in converted_tiles]
                    dataset.add_samples(samples)
                else:
                    logger.info("FiftyOne and CVAT submission disabled, working locally only")

                gdf_sliced_points["images"] = gdf_sliced_points["tile_name"].apply(
                    lambda x: f"{x}.{str(format.value)}"
                )
                gdf_sliced_points.rename(columns={"local_pixel_x": "x", "local_pixel_y": "y"}, inplace=True)
                gdf_sliced_points["image_id"] = gdf_sliced_points["tile_name"].apply(
                    lambda x: hashlib.md5(x.encode()).hexdigest()
                )
                predicted_images = herdnet_prediction_to_hasty(gdf_sliced_points,
                                                               hA_reference=None,
                                                               images_path=converted_tile_output_dir,
                                                               )
                logger.info(f"Converted {len(converted_tiles)} tiles to {format.name} format")

                hA_intermediate_path = configs.output_path / f"{config.dataset_name}_intermediate_hasty.json"
                config.hasty_intermediate_annotation_path = hA_intermediate_path

                # raise ValueError("TODO, it seems this is not persisted")

                hA_reference.images = predicted_images
                hA_reference.save(hA_intermediate_path)

                return predicted_images, config

            else:
                raise NotImplementedError(
                    "Non-geospatial correction not implemented yet. This should be easy to do though and is part of the of the other HIT_fp script")

        try:
            predicted_images, config = preppare_online_checkup(config)

            if submit_to_CVAT:
                dataset = submit_for_cvat_evaluation(dataset=dataset,
                                                     detections=predicted_images)

                # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
                dataset.annotate(
                    anno_key=config.dataset_name,
                    label_field=f"detection",
                    attributes=[],
                    launch_editor=True,
                    organization=configs.organization,
                    project_name=configs.project_name
                )


        except NoLabelsError as e:
            logger.warning(f"No labels found for dataset {config.dataset_name}, skipping...")

        except ProjectionError as e:
            logger.error(f"Projection error for dataset {config.dataset_name}: {e}")

        except NullRow as e:
            logger.error(f"Null rows found for dataset {config.dataset_name}: {e}")

        updated_config_path = configs.output_path / f"{config.dataset_name}_config.json"
        config.save(updated_config_path)
        logger.info(f"Updated config saved to {updated_config_path}")
