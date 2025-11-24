import typing
from time import sleep

import fiftyone as fo
import geopandas as gpd
import hashlib
import pandas as pd
import shutil
import typing
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from time import sleep
from typing import Optional

from active_learning.analyse_detections import analyse_point_detections_geospatial_single_image_hungarian
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfigCollection
from active_learning.reconstruct_hasty_annotation_cvat import download_cvat_annotations, foDataset2Hasty, \
    determine_changes
from active_learning.types.Exceptions import NoLabelsError
from active_learning.types.Exceptions import NullRow, ProjectionError
from active_learning.util.converter import hasty_to_shp
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.geospatial_transformations import get_geotiff_compression, get_gsd
from active_learning.util.image_manipulation import convert_tiles_to
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage, ImageLabel
from com.biospheredata.types.status import ImageFormat, LabelingStatus
from com.biospheredata.visualization.visualize_result import visualise_hasty_annotation


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


# # TODO extract this
def preppare_online_checkup(config: GeospatialDatasetCorrectionConfig,
                            tile_output_dir,
                            configs,
                            vis_output_dir,
                            include_reference,
                            submit_to_CVAT,
                            dataset,
                            species="iguana_point",
                            empties_fraction=0.0,
                            geospatial_flag=True,
                            radius_m=0.5,
                            format=ImageFormat.JPG
                            ):
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

            df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_geospatial_single_image_hungarian(
                gdf_detections=gdf_prediction_points,
                gdf_ground_truth=gdf_reference_points,
                radius_m=radius_m,
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
                                                     num_empty_samples=len(gdf_prediction_points) * empties_fraction,
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
        logger.info(f"Converting Tiles to {format.name} format")
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


def retried_ds_annotate(
        dataset,
        config,
        configs,
        max_retry_count=10,
        retry_wait=10, task_size=100):
    """
    Attempt to upload annotations to CVAT with retry logic.

    Returns:
        bool: True if successful, False otherwise
    """
    anno_key = config.dataset_name
    for retry_count in range(max_retry_count):
        try:
            if anno_key in dataset.list_annotation_runs():
                logger.warning(f"Annotation run '{anno_key}' already exists from previous attempt")

                try:
                    # Try to get info about existing run
                    anno_info = dataset.get_annotation_info(anno_key)
                    logger.info(f"Existing annotation info: {anno_info}")

                    # Delete the incomplete/failed annotation run
                    logger.info(f"Deleting incomplete annotation run '{anno_key}'...")
                    dataset.delete_annotation_run(anno_key)
                    logger.info(f"✓ Deleted annotation run '{anno_key}'")

                    # Wait a bit for CVAT to process the deletion
                    sleep(2)

                except Exception as delete_error:
                    logger.error(f"Failed to delete existing annotation run: {delete_error}")
                    # Use a different key if we can't delete

                # Attempt upload with timeout
            logger.info(f"Uploading to CVAT (attempt {retry_count + 1}/{max_retry_count})...")
            
            
            dataset.annotate(
                anno_key=config.dataset_name,
                label_field=f"detection",
                attributes=[],
                launch_editor=False,
                organization=configs.organization,
                project_name=configs.project_name,
                # Optional: Add these parameters for better control
                backend="cvat",
                task_size=task_size,  # Split into smaller tasks
                segment_size=100,  # Images per job
                # task_assignee=None,
                # job_assignees=None,
                # job_reviewers=None,
            )
            logger.info(f"Successfully uploaded annotations on attempt {retry_count + 1}")
            return True

        except Exception as e:
            logger.warning(f"Upload attempt {retry_count + 1}/{max_retry_count} failed: {e}")

            if retry_count < max_retry_count - 1:  # Don't sleep on last attempt
                logger.info(f"Retrying in {retry_wait} seconds...")
                sleep(retry_wait)
            else:
                logger.error(f"Failed to upload annotations after {max_retry_count} attempts")
                return False

    return False



def batched_geospatial_correction_upload(configs: GeospatialDatasetCorrectionConfigCollection,
                 output_dir: Path,
                 vis_output_dir: Path,
                 geospatial_flag=True,
                 submit_to_CVAT=False,
                 include_reference=False,
                 delete_dataset_if_exists=False,
                 radius=0.3,
                 ):
    """
    Main function for geospatial human in the loop correction of batches
    :param cvat_upload:
    :param radius:
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
                logger.info(f"Dataset already exists, skipping thre rest")
                continue
        else:
            logger.info("FiftyOne and CVAT submission disabled, working locally only")

        assert isinstance(config, GeospatialDatasetCorrectionConfig)


        try:
            predicted_images, config = preppare_online_checkup(config=config,
                                                               tile_output_dir=tile_output_dir,
                                                               configs=configs,
                                                               vis_output_dir=vis_output_dir,
                                                               include_reference=include_reference,
                                                               submit_to_CVAT= submit_to_CVAT,
                                                               dataset=dataset,
                                                                radius_m=radius,
                                                               )

            if submit_to_CVAT:
                dataset = submit_for_cvat_evaluation(dataset=dataset,
                                                     detections=predicted_images)

                logger.info(f"Submitting {len(predicted_images)} images to CVAT")

                # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
                retried_ds_annotate(dataset=dataset,
                                    config=config,
                                    configs=configs, task_size=60
                )


        except NoLabelsError as e:
            logger.warning(f"No labels found for dataset {config.dataset_name}, skipping...")

        except ProjectionError as e:
            logger.error(f"Projection error for dataset {config.dataset_name}: {e}")

        except NullRow as e:
            logger.error(f"Null rows found for dataset {config.dataset_name}: {e}")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            logger.error(f"This should never happen")

        updated_config_path = configs.output_path / f"{config.dataset_name}_config.json"
        config.save(updated_config_path)
        logger.info(f"Updated config saved to {updated_config_path}")




def get_point_offset(corrected_label: ImageLabel,
                     hA_prediction_tiled: HastyAnnotationV2) -> (int, int):
    """
    Get the offset for the corrected label by comparing its current position to the original position

    :param hA_prediction_tiled:
    :param hA_prediction_tiled_corrected:
    :return:
    """
    for i, annotated_image in enumerate(hA_prediction_tiled.images):
        # read the mapping file which global coordinate
        for l in annotated_image.labels:
            if l.id == corrected_label.id:
                x_offset = corrected_label.incenter_centroid.x - l.incenter_centroid.x
                y_offset = corrected_label.incenter_centroid.y - l.incenter_centroid.y
                logger.info(f"Label {l.id} was moved by {x_offset}, {y_offset}")
                return x_offset, y_offset


    raise ValueError("Label not found")


def shift_keypoint_label(corrected_label: ImageLabel, hA_prediction: HastyAnnotationV2,
                         x_offset: Optional[int] = None, y_offset: Optional[int] = None):
    """

    :param corrected_label:
    :param hA_prediction:
    :param x_offset: move the label by this offset or delete if None
    :param y_offset: move the label by this offset or delete if None
    :return:
    """
    for i, annotated_image in enumerate(hA_prediction.images):
        # read the mapping file which global coordinate
        for l in annotated_image.labels:
            if l.id == corrected_label.id:
                if x_offset is None and y_offset is None:
                    hA_prediction.images[i].labels.remove(l)
                elif x_offset != 0 or y_offset != 0:
                    for kp in l.keypoints:
                        kp.x += int(x_offset)
                        kp.y += int(y_offset)
                else:
                    logger.info(f"Label {l.id} was not moved")


def merged_corrected_annotations(config: GeospatialDatasetCorrectionConfig,
                                    configs_path: Path,
                                 target_images_path: Path,
                                 visualisation_path: Path | None = None ):
    """
    Merge the hasty prediction and the corrected hasty annotation into the reference annotation file
    :param config:
    :param target_images_path:
    :param visualisation_path:
    :return:
    """

    hA_prediction_path = config.hasty_intermediate_annotation_path
    if hA_prediction_path is None:
        hA_prediction_path = configs_path / f"{config.dataset_name}_intermediate_hasty.json"  # in the current folder
        if hA_prediction_path.exists():
            logger.info(f"Guessing {hA_prediction_path} as hasty intermediate annotation path successful")
        else:
            raise NoLabelsError("hA_prediction_path is None")

    hA_corrected_path = config.output_path / f"{config.dataset_name}_corrected_intermediate_hasty.json"

    if not hA_corrected_path.exists():
        raise NoLabelsError(f"hA_corrected_path {hA_corrected_path} does not exist")
    # we need the prediction for the list of images
    hA_prediction = HastyAnnotationV2.from_file(file_path=hA_prediction_path)
    hA_correction = HastyAnnotationV2.from_file(file_path=hA_corrected_path)
    hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)

    assert len(hA_prediction.images) == len(hA_correction.images)
    # remove "_counts" from end of dataset name if present
    dataset_name = config.dataset_name.replace("_counts", "")
    hA_reference.delete_dataset(dataset_name=dataset_name)
    hA_reference.delete_dataset(dataset_name=f"ha_corrected_{dataset_name}")
    hA_reference.delete_dataset(dataset_name=f"ha_{dataset_name}")

    images = []
    uncorrected_images = []
    total_labels = 0

    for pred_image in hA_prediction.images:
        # logger.info(f"Processing {pred_image.image_name}, dataset {dataset_name}")


        # be careful to match the image ids
        corr_image = hA_correction.get_image_by_id(pred_image.image_id)
        corr_image.image_status = LabelingStatus.COMPLETED
        try:
            # The image were not AnnotatedImage before but ImageLabel Collection
            pred_image = AnnotatedImage(**pred_image.model_dump(), dataset_name=dataset_name,
                                        image_status=LabelingStatus.COMPLETED)
        except:
            pred_image.dataset_name = dataset_name
            pred_image.image_status = LabelingStatus.COMPLETED
            # pred_image = AnnotatedImage(**pred_image.model_dump())

        if visualisation_path is not None and len(corr_image.labels) > 0:
            logger.info(f"visualising {pred_image.image_name}, dataset {dataset_name} to {visualisation_path}")
            visualise_hasty_annotation(image=corr_image, images_path=config.image_tiles_path / "converted_tiles",
                                       output_path=visualisation_path,
                                       # title = f"Prediction {dataset_name}, {pred_image.image_name}",
                                       show=False)
            plt.close()

        pred_image.dataset_name = f"ha_{dataset_name}"
        # uncorrected_images.append(pred_image)

        # hA_reference.images.append(pred_image)
        target_path = target_images_path / pred_image.dataset_name
        target_path.mkdir(parents=True, exist_ok=True)

        if not (target_path / pred_image.image_name).exists():
            shutil.copy(config.image_tiles_path / "converted_tiles" / pred_image.image_name,
                        target_path / pred_image.image_name)

        corr_image.dataset_name = f"ha_corrected_{dataset_name}"
        hA_reference.images.append(corr_image)
        total_labels += len(corr_image.labels)
        images.append(corr_image)

        target_path = target_images_path / corr_image.dataset_name
        target_path.mkdir(parents=True, exist_ok=True)

        if not (target_images_path / corr_image.dataset_name / corr_image.image_name).exists():
            shutil.copy(config.image_tiles_path / "converted_tiles" / corr_image.image_name,
                        target_images_path / corr_image.dataset_name / corr_image.image_name)

    # update the reference annotation file
    logger.info(f"Kept {sum([len(img.labels) for img in images])} labels including hard negatives")
    # hA_updated_path = config.hasty_reference_annotation_path.stem + f"_updated_with_{dataset_name}.json"
    logger.info(f"Updated config.hasty_reference_annotation_path to {config.hasty_reference_annotation_path}")

    hA_reference.save(config.hasty_reference_annotation_path)

    logger.info(hA_reference.dataset_statistics())

    return images, uncorrected_images


def batched_geospatial_correction_download(config: GeospatialDatasetCorrectionConfig,
                                           hA_to_update: Path,
                                           anno_field = "iguana"): # () configs_path: Path, ):
    """

    :param config:
    :return:
    """

    hA_prediction_path = config.hasty_intermediate_annotation_path
    if hA_prediction_path is None:
        raise NoLabelsError("hasty_intermediate_annotation_path is None but it is needed.")
    #     hA_prediction_path = configs_path / f"{config.dataset_name}_intermediate_hasty.json" # in the current folder
    #     if hA_prediction_path.exists():
    #         logger.info(f"Guessing {hA_prediction_path} as hasty intermediate annotation path sucessful")
    #     else:
    #         raise NoLabelsError("hA_prediction_path is None")
    hA_prediction = HastyAnnotationV2.from_file(file_path=hA_prediction_path)
    #
    # # the 256px crops
    # hA_prediction_tiled_path = output_path / f"{dataset_name}_tiled_hasty.json"
    # hA_prediction_tiled = HastyAnnotationV2.from_file(file_path=config.hA_prediction_tiled_path)

    if hA_to_update is None:
        hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)
    else:
        hA_reference = HastyAnnotationV2.from_file(hA_to_update)
    hA_reference_updated = hA_reference.copy(deep=True)
    view, dataset = download_cvat_annotations(dataset_name=config.dataset_name)

    hA_updated = foDataset2Hasty(hA_template=hA_prediction.copy(deep=True), dataset=dataset, anno_field=anno_field)
    for image in hA_updated.images:
        image.dataset_name = config.dataset_name

    new_boxes = 0
    new_points = 0
    modified_annotated_image_names = []

    changes = determine_changes(hA_prediction, hA_updated)

    logger.info(f"Found {changes['total_changes']} changes")


    ### TODO add removed points to Hard Negative Category.

    gdf_prediction = hasty_to_shp(tif_path=config.image_tiles_path, hA_reference=hA_prediction)
    gdf_annotation = hasty_to_shp(tif_path=config.image_tiles_path, hA_reference=hA_updated)


    gdf_false_positives, gdf_true_positives, gdf_false_negatives = analyse_point_detections_geospatial_single_image_hungarian(
        gdf_ground_truth=gdf_annotation,
        gdf_detections=gdf_prediction, radius_m=0.5,
        labels_column="label", scores_column="score",
    )

    corrected_path = config.output_path / f"{config.dataset_name}_corrected_annotation.geojson"
    hard_negatives_path = config.output_path / f"{config.dataset_name}_hard_negatives.geojson"
    gdf_annotation.to_file(filename=corrected_path, driver="GeoJSON")

    gdf_false_positives.drop(columns="buffer").to_file(filename=hard_negatives_path, driver="GeoJSON")
    gdf_true_positives.drop(columns="buffer").to_file(filename=config.output_path / f"{config.dataset_name}_true_positives.geojson", driver="GeoJSON")
    gdf_false_negatives.to_file(filename=config.output_path / f"{config.dataset_name}_false_negatives.geojson", driver="GeoJSON")
    logger.info(f"corrected file saved to : {corrected_path}")
    logger.info(f"hard negatives file saved to : {hard_negatives_path}")

    # TODO add the hard negatives to the hA_updated files
    def add_hard_negatives(config: GeospatialDatasetCorrectionConfig,
                           hA_updated: HastyAnnotationV2,
                           hA_prediction: HastyAnnotationV2,
                           label_ids: typing.List[str],
                           gdf_false_positives: gpd.GeoDataFrame):



        for id, row in gdf_false_positives.iterrows():
            if row.img_id == "ec1eb174-6979-485b-b0c9-aecef26c3a47":
                pass
            hard_negative = hA_prediction.get_label_by_id(row.label_id)
            hard_negative.class_name = "hard_negative"
            hA_updated.add_labels_to_image(image_id=row.img_id, label=hard_negative)

        return hA_updated

    hA_updated = add_hard_negatives(config,
                                    hA_updated=hA_updated,
                                    hA_prediction=hA_prediction,
                                    gdf_false_positives=gdf_false_positives,
                                    label_ids = list(gdf_false_positives.label_id))

    summary = hA_updated.get_flat_df()[["class_name", "image_id"]].groupby("class_name").agg("count").rename(columns={"image_id": "count"})
    logger.info(f"summary: {summary}")

    # create a new annotations from the changes and save everything
    hA_updated.save(config.output_path / f"{config.dataset_name}_corrected_intermediate_hasty.json")

    config.corrected_path = corrected_path
    report_path = corrected_path.parent / f"{corrected_path.stem}_correction_config.json"
    config.save(report_path)
    logger.info(f"Saved report config to {report_path}")

    return report_path