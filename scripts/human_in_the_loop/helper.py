from datetime import date

import copy

from typing import List, Optional

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.config.mapping import keypoint_id_mapping
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.Annotation import project_point_to_crop, project_label_to_crop
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object, create_regular_raster_grid, \
    RasterCropperPoints, crop_by_regular_grid, pad_to_multiple, RasterCropperPointsMatching
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file, \
    ImageLabelCollection

import fiftyone as fo
from loguru import logger
import shapely
from loguru import logger
from pathlib import Path
from typing import Optional

from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty, download_cvat_annotations, foDataset2Hasty, \
    determine_changes
from active_learning.types.Exceptions import TooManyLabelsError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabel
from com.biospheredata.types.status import AnnotationType
from com.biospheredata.visualization.visualize_result import visualise_hasty_annotation


def hit_fp_cvat_upload(config: DatasetCorrectionConfig, report_path: Path, CONFIDENCE_THRESHOLD = 0.9):
    """
    Upload false positives one annotation at a time to CVAT for human correction
    :param config:
    :param report_path:
    :param CONFIDENCE_THRESHOLD: scores
    :return:
    """

    config.output_path.mkdir(exist_ok=True)
    config_path = report_path / f"{config.dataset_name}_config.json"
    config.save(config_path)
    report_config = DatasetCorrectionReportConfig(**config.model_dump())
    report_config.report_path = report_path

    # loaded_config = DatasetCorrectionConfig.load(config_path)

    df_ground_truth = pd.read_csv(config.subset_base_path / config.herdnet_annotation_name)

    hA_ground_truth = HastyAnnotationV2.from_file(config.subset_base_path / config.hasty_ground_truth_annotation_name)

    images = list(config.images_path.glob(f"*.{config.suffix}"))
    images_list = [i.image_name for i in hA_ground_truth.images]
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {config.images_path}")

    df_detections = pd.read_csv(config.detections_path)
    # df_detections.species = "iguana_point" # if that does not show up in CVAT you will have to create it manually
    df_detections = df_detections[df_detections.scores > CONFIDENCE_THRESHOLD]

    # df_detections = df_detections[:15] # TODO debugging
    # df_detections = df_detections[df_detections.images == "delplanque_train___012e884435eef1b4503af40f1cca54ee9aa2c2f3.JPG"] # TODO debugging

    df_detections[["images", "labels", "scores"]].merge(df_ground_truth[["images", "labels"]], on=["images", "labels"], how="left", indicator=True)

    df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth_all = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=config.radius,
        image_list=images_list,
    )

    logger.info(f"Found {len(df_false_positives)} false positives, {len(df_true_positives)} true positives and {len(df_false_negatives)} false negatives in the detections for a Ground Truth of {len(df_ground_truth)} detections.")
    logger.info(f"This means that our error is: {(len(df_false_positives) + len(df_true_positives)) / len(df_ground_truth)} .")

    if len(df_false_positives) == 0:
        raise ValueError(f"No false positives found in the detections. Normally that would be good. But here we want to correct false positives in the ground truth. ")

    IL_false_positive_detections = herdnet_prediction_to_hasty(df_false_positives, config.images_path, dataset_name=config.dataset_name, hA_reference=hA_ground_truth)

    # getting this reference just for the label classes
    hA_reference = HastyAnnotationV2.from_file(config.reference_base_path / config.hasty_reference_annotation_name)
    hA_prediction = HastyAnnotationV2(
        project_name="false_positive_correction",
        images=IL_false_positive_detections,
        export_format_version="1.1",
        label_classes=hA_reference.label_classes
    )

    hA_prediction_path = config.corrected_path / f"{config.dataset_name}_predictions_hasty.json"
    report_config.hA_prediction_path = hA_prediction_path
    hA_prediction.save(hA_prediction_path)
    hA_tiled_prediction = copy.deepcopy(hA_prediction)
    hA_tiled_prediction.images = []
    all_image_mappings: List[ImageCropMetadata] = []

    # delete the dataset if it exists
    try:
        # fo.dataset_exists(config.dataset_name)
        fo.delete_dataset(config.dataset_name)
        pass
    except:
        logger.warning(f"Dataset {config.dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(name=config.dataset_name)
        dataset.persistent = True

    # create crops for each of the detections
    for i in IL_false_positive_detections:
        im = PIL.Image.open(config.images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")

        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          width=512,
                                                                                          height=512,
                                                                                          im=im,
                                                                                          output_path=config.output_path,
                                                                                          )

        gt_image = hA_ground_truth.get_image_by_name(i.image_name, dataset_name=config.dataset_name)

        for im, ci in zip(cropped_annotated_images, image_mappings):
            projected_points = []
            for p in gt_image.labels:

                if p.incenter_centroid.within(ci.bbox_polygon):

                    # projected_points.append( project_point_to_crop(p.incenter_centroid, ci.bbox_polygon) )
                    projected_points.append(project_label_to_crop(p, ci.bbox_polygon))
            im.labels.extend(projected_points)

        # get the original annotations for the cropped image
        hA_tiled_prediction.images.extend(cropped_annotated_images)

        samples = [fo.Sample(filepath=path) for path in images_set]
        dataset.add_samples(samples)

        # Save image mappings
        # TODO save this at the end
        for image_mapping in image_mappings:
            image_mapping.save(
                config.output_path / f"{image_mapping.parent_label_id}_metadata.json")

        all_image_mappings.extend(image_mappings)

        for cropped_annotated_image in cropped_annotated_images:
            cropped_annotated_image.save(config.output_path / f"{cropped_annotated_image.image_name}_labels.json")

        # create a polygon around each Detection
        # TODO visualise where the crops happened

        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                             detections=cropped_annotated_images)

    # report_config.image_mappings = image_mappings
    report_config.hA_prediction_tiled_path = config.corrected_path / f"{config.dataset_name}_tiled_hasty.json"

    hA_tiled_prediction.save(report_config.hA_prediction_tiled_path)
    report_config.save(report_path / "report.json")

    # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
    dataset.annotate(
        anno_key=config.dataset_name,
        label_field=f"detection",
        attributes=[],
        launch_editor=True,
        organization="IguanasFromAbove",
        project_name="Single_Image_FP_correction"
    )

    logger.info(f"Correct the false positives in the CVAT interface and then save the annotations. ")
    logger.info(f"After saving the annotations, you can run the next script to apply the corrections to the ground truth annotations with {report_path / 'report.json'}")





def hit_fp_gt_cvat_upload(config: DatasetCorrectionConfig, report_path: Path, CONFIDENCE_THRESHOLD = 0.9, grid_size = 512):
    """
    Upload false positives and ground truth annoations to CVAT for human correction
    :param config:
    :param report_path:
    :param CONFIDENCE_THRESHOLD: scores
    :return:
    """

    config.output_path.mkdir(exist_ok=True)
    config_path = report_path / f"{config.dataset_name}_config.json"
    config.save(config_path)
    report_config = DatasetCorrectionReportConfig(**config.model_dump())
    report_config.report_path = report_path

    # loaded_config = DatasetCorrectionConfig.load(config_path)

    df_ground_truth = pd.read_csv(config.subset_base_path / config.herdnet_annotation_name)
    hA_ground_truth = HastyAnnotationV2.from_file(config.subset_base_path / config.hasty_ground_truth_annotation_name)

    #  in order for this to work we need to convert all box annotations to points
    for i in hA_ground_truth.images:
        for l in i.labels:
                kp = Keypoint(
                    id=l.id,
                    x=int(l.incenter_centroid.x),
                    y=int(l.incenter_centroid.y),
                    keypoint_class_id =keypoint_id_mapping.get(l.class_name.lower(), "body")
                )
                l.keypoints = [kp]


    images = list(config.images_path.glob(f"*.{config.suffix}"))
    images_list = [i.image_name for i in hA_ground_truth.images]
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {config.images_path}")

    df_detections = pd.read_csv(config.detections_path)
    # df_detections.species = "iguana_point" # if that does not show up in CVAT you will have to create it manually
    df_detections = df_detections[df_detections.scores > CONFIDENCE_THRESHOLD]


    # df_detections = df_detections[:15] # TODO debugging
    # df_detections = df_detections[df_detections.images == "delplanque_train___012e884435eef1b4503af40f1cca54ee9aa2c2f3.JPG"] # TODO debugging

    df_detections[["images", "labels", "scores"]].merge(df_ground_truth[["images", "labels"]], on=["images", "labels"], how="left", indicator=True)

    df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth_all = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=config.radius,
        image_list=images_list,
    )

    logger.info(f"Found {len(df_false_positives)} false positives, {len(df_true_positives)} true positives and {len(df_false_negatives)} false negatives in the detections for a Ground Truth of {len(df_ground_truth)} detections.")
    logger.info(f"This means that our error is: {(len(df_false_positives) + len(df_true_positives)) / len(df_ground_truth)} .")

    if len(df_false_positives) == 0:
        raise ValueError(f"No false positives found in the detections. Normally that would be good. But here we want to correct false positives in the ground truth. ")


    # TODO rename false positive class names
    # Rename false positive class names
    df_false_positives = df_false_positives.rename(columns={"species": "original_class_name"})
    df_false_positives["species"] = df_false_positives["original_class_name"] + "_prediction"
    IL_false_positive_detections = herdnet_prediction_to_hasty(df_false_positives, config.images_path, dataset_name=config.dataset_name, hA_reference=hA_ground_truth)

    # getting this reference just for the label classes
    hA_reference = HastyAnnotationV2.from_file(config.reference_base_path / config.hasty_reference_annotation_name)
    hA_prediction = HastyAnnotationV2(
        project_name="false_positive_correction",
        images=IL_false_positive_detections,
        export_format_version="1.1",
        label_classes=hA_reference.label_classes
    )

    hA_prediction_path = config.corrected_path / f"{config.dataset_name}_predictions_hasty.json"
    report_config.hA_prediction_path = hA_prediction_path
    hA_prediction.save(hA_prediction_path)
    hA_tiled_prediction = copy.deepcopy(hA_prediction)
    hA_tiled_prediction.images = []
    all_image_mappings: List[ImageCropMetadata] = []

    # delete the dataset if it exists
    try:
        # fo.dataset_exists(config.dataset_name)
        fo.delete_dataset(config.dataset_name)
        pass
    except:
        logger.warning(f"Dataset {config.dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(name=config.dataset_name)
        # TODO set these from the reference
        dataset.default_classes = [
            # "Buffalo",
            # "Elephant",
            # "Kob",
            # "Topi",
            # "Warthog",
            # "Waterbuck",
            "hard_negative",  # Class that will be added in CVAT even if it does not exist yet in the data
        ]

        dataset.persistent = True


    # different to hit_fp_cvat_upload , we group by image

    # create crops for each of the detections
    for i in IL_false_positive_detections:
        im = PIL.Image.open(config.images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")
        full_images_path_padded = config.output_path / "padded_images"
        full_images_path_padded.mkdir(exist_ok=True, parents=True)



        padded_image_path = full_images_path_padded / i.image_name
        new_width, new_height = pad_to_multiple(config.images_path / i.image_name,
                                                padded_image_path,
                                                grid_size,
                                                grid_size,
                                                0)
        logger.info(f"Padded {i.image_name} to {new_width}x{new_height}")

        grid, _ = create_regular_raster_grid(max_x=new_width, max_y=new_height,
                                             slice_height=grid_size, slice_width=grid_size,
                                             overlap=0)

        # which grid cell is occupied with a prediction label
        occupied_rasters = {}
        for raster_id, g in grid.items():
            contains_label = False
            for p in i.labels:
                if p.incenter_centroid.within(g):
                    contains_label = True
                    occupied_rasters[raster_id] = g
                    break
            if not contains_label:
                continue

        gt_image = hA_ground_truth.get_image_by_name(i.image_name, dataset_name=config.dataset_name)
        gt_labels = gt_image.labels
        i.labels.extend(gt_labels)

        # get grid cells which contain at least one pred label



        cropper_points = RasterCropperPointsMatching(hi=i,
                                             rasters=occupied_rasters,
                                             full_image_path=full_images_path_padded / i.image_name,
                                             output_path=config.output_path,
                                             dataset_name=i.dataset_name)

        cropped_annotated_images, images_set = cropper_points.crop_out_images()
        image_mappings = cropper_points.image_mappings

        hA_tiled_prediction.images.extend(cropped_annotated_images)

        samples = [fo.Sample(filepath=path) for path in images_set]
        dataset.add_samples(samples)

        # Save image mappings
        # TODO save this at the end
        for image_mapping in image_mappings:
            image_mapping.save(
                config.output_path / f"{image_mapping.cropped_image_id}_ci_metadata.json")

        all_image_mappings.extend(image_mappings)

        for cropped_annotated_image in cropped_annotated_images:
            cropped_annotated_image.save(config.output_path / f"{cropped_annotated_image.image_name}_labels.json")


        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                             detections=cropped_annotated_images)

    # report_config.image_mappings = image_mappings
    report_config.hA_prediction_tiled_path = config.corrected_path / f"{config.dataset_name}_tiled_hasty.json"

    hA_tiled_prediction.save(report_config.hA_prediction_tiled_path)
    report_config.save(report_path / "report.json")

    # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
    dataset.annotate(
        anno_key=config.dataset_name,
        label_field=f"detection",
        attributes=[],
        launch_editor=True,
        organization="IguanasFromAbove",
        project_name="Single_Image_HNPM_correction"
    )

    logger.info(f"Correct the false positives in the CVAT interface and then save the annotations. ")
    logger.info(f"After saving the annotations, you can run the next script to apply the corrections to the ground truth annotations with {report_path / 'report.json'}")



def get_point_offset(labels_1: ImageLabel,
                     labels_2: ImageLabel) -> (int, int):
    """
    Get the offset for the corrected label by comparing its current position to the original position

    :param hA_prediction_tiled:
    :param hA_prediction_tiled_corrected:
    :return:
    """

    x_offset = labels_1.incenter_centroid.x - labels_2.incenter_centroid.x
    y_offset = labels_1.incenter_centroid.y - labels_2.incenter_centroid.y
    logger.info(f"Label {labels_2.id} was moved by {x_offset}, {y_offset}")
    return x_offset, y_offset


    raise ValueError("Label not found")


def shift_keypoint_label(corrected_label: ImageLabel,
                         hA_prediction: HastyAnnotationV2,
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



def hit_cvat_download(report_path: Path):


    config = DatasetCorrectionReportConfig.load(report_path)

    # analysis_date = "2025_02_22"
    # base_path = Path(f'/Users/christian/data/training_data/2025_02_22_HIT/FMO02_full_orthophoto_tiles')
    #
    #
    # images_path = base_path
    # dataset_name = f"eal_{analysis_date}_review"
    #
    # images_path = base_path
    #
    # output_path = base_path / "object_crops"
    #
    # # original predictions
    hA_prediction_path = config.hA_prediction_path
    hA_prediction = HastyAnnotationV2.from_file(file_path=hA_prediction_path)
    #
    # # the 256px crops
    # hA_prediction_tiled_path = output_path / f"{dataset_name}_tiled_hasty.json"
    hA_prediction_tiled = HastyAnnotationV2.from_file(file_path=config.hA_prediction_tiled_path)

    hA_reference = HastyAnnotationV2.from_file(config.reference_base_path / config.hasty_reference_annotation_name)

    #  in order for this to work we need to convert all box annotations to points
    for i in hA_reference.images:
        for l in i.labels:

            if not l.keypoints or len(l.keypoints) == 0:

                kp = Keypoint(
                    id=l.id,
                    x=int(l.incenter_centroid.x),
                    y=int(l.incenter_centroid.y),
                    keypoint_class_id=keypoint_id_mapping.get(l.class_name.lower(), "body")
                )
                l.keypoints = [kp]


    hA_reference_updated = hA_reference.copy(deep=True)
    view, dataset = download_cvat_annotations(dataset_name=config.dataset_name)

    hA_updated_tiled = foDataset2Hasty(hA_template=hA_prediction_tiled.copy(deep=True), dataset=dataset, anno_field="iguana")

    hA_ground_truth = HastyAnnotationV2.from_file(config.subset_base_path / config.hasty_ground_truth_annotation_name)



    changes = determine_changes(hA_prediction_tiled, hA_updated_tiled)

    new_boxes = 0
    new_points = 0
    moved_points = 0
    modified_annotated_image_names = []

    hA_updated_tiled_flat = hA_updated_tiled.get_flat_df()
    hA_updated_tiled_flat.groupby("class_name").size()

    # Now we can add every annotation to the original hasty data
    for i, updated_tiled_image in enumerate(hA_updated_tiled.images):
        point_added = False
        point_existing = False
        point_moved = False
        for j, corrected_label in enumerate(updated_tiled_image.labels):
            original_point_shift = False
            metadata_mapping_file_name = f"{updated_tiled_image.image_id}_ci_metadata.json"
            icm = ImageCropMetadata.load(config.output_path / metadata_mapping_file_name)
            # check if the label was present before
            combined_subsets_tiled_label = hA_prediction_tiled.get_label_by_id(corrected_label.id)

            # it should not be in the reference data
            ref_original_label = hA_reference_updated.get_label_by_id(corrected_label.id)

            # label on the big image
            pred_original_label = hA_prediction.get_label_by_id(corrected_label.id)
            # point not in the ground truth but in the predicitons
            if not ref_original_label and pred_original_label:
                logger.info(f"New point detected: {corrected_label.id} in image {updated_tiled_image.image_name}")

                # project the point back to the original image coordinates
                for keypoint in corrected_label.keypoints:
                    # convert the keypoint to the original image coordinates
                    keypoint.x += icm.bbox[0]
                    keypoint.y += icm.bbox[1]
                image_id = icm.parent_image_id

                # upsert the corrected label into the original image
                hA_reference_updated.add_labels_to_image(image_id=icm.parent_image_id, label=corrected_label)
            
                # # we move the original point just by how much we moved the point in the cropped image
                # if corrected_label.incenter_centroid != combined_subsets_tiled_label.incenter_centroid:
                #     point_moved = True
                #     # how much was the point moved in the tiled image
                #     x_offset, y_offset = get_point_offset(corrected_label, hA_prediction_tiled)
                #     # move the point in the original image by the same amount
                #     shift_keypoint_label(pred_original_label, x_offset, y_offset)
                #
                #     metadata_mapping_file_name = f"{updated_tiled_image.image_id}_ci_metadata.json"
                #
                # if corrected_label.class_name != pred_original_label.class_name:
                #     # point renamed.
                #     pred_original_label.class_name = corrected_label.class_name
                #
                # # upsert the corrected label into the original image
                # hA_reference_updated.add_labels_to_image(image_id=image_id, label=corrected_label)

            elif ref_original_label and not pred_original_label:
                original_point_shift = True
                """
                The point was in the original reference data but now it is not in the predictions. """
                logger.info(f"original point, not in predictions")
                if corrected_label.incenter_centroid != combined_subsets_tiled_label.incenter_centroid:
                    x_offset, y_offset = get_point_offset(corrected_label, combined_subsets_tiled_label)
                    ref_original_label.shift_label(x=x_offset, y=y_offset)
                    pass

            elif not ref_original_label and not pred_original_label:
                logger.warning(f"Entirely new label which was not in the predictions before: {corrected_label.id} in image {updated_tiled_image.image_name}")
                # entirely new label
                for keypoint in corrected_label.keypoints:
                    # convert the keypoint to the original image coordinates
                    keypoint.x += icm.bbox[0]
                    keypoint.y += icm.bbox[1]
                image_id = icm.parent_image_id

                # upsert the corrected label into the original image
                hA_reference_updated.add_labels_to_image(image_id=icm.parent_image_id, label=corrected_label)

        hA_reference_updated_flat = hA_reference_updated.get_flat_df()
        hA_reference_updated_flat.groupby("class_name").size()


        # visualise all corrections
        grid_size = 512
        original_image_path = Path("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/delplanque_train")
        i = hA_reference_updated.get_image_by_id(icm.parent_image_id)
        visualise_hasty_annotation(i, images_path=original_image_path,
                                   show=False, output_path=config.output_path / "corrected", #  / f"{i.image_name}_corrected.png",
                                   figsize=(8,8), dpi=300)


        # im = PIL.Image.open(original_image_path / i.image_name)
        # # convert to RGB
        # if im.mode != "RGB":
        #     im = im.convert("RGB")
        full_images_path_padded = config.output_path / "padded_images_hit_2"
        full_images_path_padded.mkdir(exist_ok=True, parents=True)

        padded_image_path = full_images_path_padded / i.image_name
        new_width, new_height = pad_to_multiple(original_image_path / i.image_name,
                                                padded_image_path,
                                                grid_size,
                                                grid_size,
                                                0)
        logger.info(f"Padded {i.image_name} to {new_width}x{new_height}")

        grid, _ = create_regular_raster_grid(max_x=new_width, max_y=new_height,
                                             slice_height=grid_size, slice_width=grid_size,
                                             overlap=0)

        # which grid cell is occupied with a prediction label
        occupied_rasters = {}
        for raster_id, g in grid.items():
            contains_label = False
            for p in i.labels:
                if p.incenter_centroid.within(g):
                    contains_label = True
                    occupied_rasters[raster_id] = g
                    break
            if not contains_label:
                continue


        # get grid cells which contain at least one pred label

        cropper_points = RasterCropperPointsMatching(hi=i,
                                                     rasters=occupied_rasters,
                                                     full_image_path=full_images_path_padded / i.image_name,
                                                     output_path=config.output_path,
                                                     dataset_name=i.dataset_name)

        cropped_annotated_images, images_set = cropper_points.crop_out_images()


        for ci, imp in zip(cropped_annotated_images, images_set):
            # TODO visualise
            visualise_hasty_annotation(ci, images_path=config.output_path,
                                       show=False, output_path=config.output_path / "corrected",  #/ f"{i.image_name}_{ci.image_id}_corrected.png",
                                       figsize=(4, 4), dpi=300, show_axis=False)
            logger.info(f"Corrected {i.image_name}_{ci.image_id} saved to {config.output_path / 'corrected'}")
            pass

        pass
            # # if the label is new, so it can't be moved in the old image
            # else:
            #     metadata_mapping_file_name = f"{updated_tiled_image.image_id}_ci_metadata.json"
            # # project the label back to the original image coordinates
            #     try:
            #         icm = ImageCropMetadata.load(config.output_path / metadata_mapping_file_name)
            #     except FileNotFoundError:
            #         logger.error(f"Metadata file not found: {config.output_path / metadata_mapping_file_name}")
            #
            #
            #     # project the point back to the original image coordinates
            #     for keypoint in corrected_label.keypoints:
            #         # convert the keypoint to the original image coordinates
            #         keypoint.x += icm.bbox[0]
            #         keypoint.y += icm.bbox[1]
            #     image_id = icm.parent_image_id
            #
            #     # upsert the corrected label into the original image
            #     hA_reference_updated.add_labels_to_image(image_id=image_id, label=corrected_label)

        # TODO handle new labels



        # TODO: REMOVE a label in the reference data - labels in hA_prediction_tiled but not in hA_updated_tiled
        # TODO: to do that, we need to check if there is an object in either of the list
        # logger.info(f"Looking for deleted labels in image {i} ")
        # for i, ha_image in enumerate(hA_prediction_tiled.images):
        #     for j, preexisting_label in enumerate(ha_image.labels):
        #         updated_label = hA_updated_tiled.get_label_by_id(preexisting_label.id)
        #
        #         if updated_label is None:
        #             # label was deleted
        #             hA_reference_updated.remove_label(preexisting_label)





    # TODO evaluate how many labels were added. After this it makes no sense to look for additonal missing labels because we assument the model would have found all of them
    # This would be tested by training the model from scratch with the new labels and then evaluating the model on the training dataset. The recall should be 100% if the model is good enough
    logger.info(f"Found {new_boxes} new boxes and {new_points} new points in the hasty annotation")

    curr_date = date.today().strftime("%Y_%m_%d")
    changes = determine_changes(hA_reference, hA_reference_updated)
    logger.info(f"Updated {config.reference_base_path / config.hasty_reference_annotation_name}")
    corrected_full_path = config.reference_base_path / f"{config.dataset_name}_hasty_corrected_{curr_date}.json"
    hA_reference_updated.save(corrected_full_path)
    logger.info(f"Saved corrected hasty annotation to {corrected_full_path}")

    config.corrected_path = corrected_full_path

    config.save(report_path)
    logger.info(f"Saved report config to {report_path}")

    return hA_reference_updated