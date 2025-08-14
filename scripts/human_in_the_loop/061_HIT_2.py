"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
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




def main():
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report.json")
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    report_path_1: Path = Path("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    report_path_2: Path = Path("/Users/christian/data/training_data/2025_07_10_refined/Floreana_detection_classic/train/report/report.json")
    report_path_3: Path = Path("/Users/christian/data/training_data/2025_08_10_endgame/Fernandina_s_detection/train/report/report.json")

    report_path = report_path_3

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

            if corrected_label.class_name == "iguana_point":
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
            if corrected_label.class_name == "iguana" and point_existing:
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

if __name__ == "__main__":
    main()