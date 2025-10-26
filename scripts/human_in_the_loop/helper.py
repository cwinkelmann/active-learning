import copy

from typing import List, Optional

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file

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



def hit_fp_cvat_upload(config: DatasetCorrectionConfig, report_path: Path, CONFIDENCE_THRESHOLD = 0.9):


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

        # TODO it produces the same image multiple times, tile_22.tif_ba2ec74e-76b8-4a26-ae48-dc969f9a4dd7 tile_22.tif_96eb06c5-e0b8-4ea6-96db-138b010fdee0.jpg
        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          width=512,
                                                                                          height=512,
                                                                                          im=im,
                                                                                          output_path=config.output_path,
                                                                                          )

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
    hA_reference_updated = hA_reference.copy(deep=True)
    view, dataset = download_cvat_annotations(dataset_name=config.dataset_name)

    hA_updated = foDataset2Hasty(hA_template=hA_prediction_tiled, dataset=dataset, anno_field="iguana")

    new_boxes = 0
    new_points = 0
    modified_annotated_image_names = []

    # Now we can add every annotation to the original hasty data
    for i, annotated_image in enumerate(hA_updated.images):
        point_existing = False

        for j, corrected_label in enumerate(annotated_image.labels):

            # if corrected_label.class_name == "iguana_point":
                 
            point_existing = True
            metadata_mapping_file_name = f"{corrected_label.id}_metadata.json"
            # project the label back to the original image coordinates
            icm = ImageCropMetadata.load(config.output_path / metadata_mapping_file_name)
            for keypoint in corrected_label.keypoints:
                # convert the keypoint to the original image coordinates
                keypoint.x += icm.bbox[0]
                keypoint.y += icm.bbox[1]
            image_id = icm.parent_image_id
            new_points += 1
            hA_reference_updated.add_labels_to_image(image_id, corrected_label)
        # Every picture contains either a single point OR a point + bbox

        # new_boxes
        for j, corrected_label in enumerate(annotated_image.labels):
            if corrected_label.class_name == "iguana" and not point_existing:
                corrected_label.bbox[0] += icm.bbox[0]
                corrected_label.bbox[1] += icm.bbox[1]
                corrected_label.bbox[2] += icm.bbox[0]
                corrected_label.bbox[3] += icm.bbox[1]

                image_id = icm.parent_image_id
                new_boxes += 1
                hA_reference_updated.add_labels_to_image(image_id, corrected_label)


    # TODO evaluate how many labels were added. After this it makes no sense to look for additonal missing labels because we assument the model would have found all of them
    # This would be tested by training the model from scratch with the new labels and then evaluating the model on the training dataset. The recall should be 100% if the model is good enough
    logger.info(f"Found {new_boxes} new boxes and {new_points} new points in the hasty annotation")


    changes = determine_changes(hA_reference, hA_reference_updated)
    logger.info(f"Updated {config.reference_base_path / config.hasty_reference_annotation_name}")
    corrected_full_path = config.reference_base_path / f"{config.dataset_name}_hasty_corrected_1.json"
    hA_reference_updated.save(corrected_full_path)
    logger.info(f"Saved corrected hasty annotation to {corrected_full_path}")

    config.corrected_path = corrected_full_path

    config.save(report_path)
    logger.info(f"Saved report config to {report_path}")