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
    config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report.json")

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

    view, dataset = download_cvat_annotations(dataset_name=config.dataset_name)

    hA_updated = foDataset2Hasty(hA_template=hA_prediction_tiled, dataset=dataset)
    changes = determine_changes(hA_reference, hA_updated)
    # Now we can add every annotation to the original hasty data

    config.reference_base_path / config.hasty_reference_annotation_name

    config.dataset_name

    # df_stats, hA_prediction_tiled_corrected, iCLdl_deletions = cvat2hasty(hA_prediction_tiled, config.dataset_name)

    ## TODO outsource this to a function
    """
    Retrieve the original untiled label, the tiled label and the corrected label
    with the tiled version get the offset or if the label was deleted
    with that offset move it withing the original image
    """
    for i, corrected_annotated_image in enumerate(hA_prediction_tiled_corrected.images):
        if len(corrected_annotated_image.labels) > 1:
            raise TooManyLabelsError("Only one label is supported")

        elif len(corrected_annotated_image.labels) == 1:
            # "In this mode only the center label is supported, otherwise correction get complicated"
            corrected_label = corrected_annotated_image.labels[0]

            if all(isinstance(kp, (Keypoint)) for kp in corrected_label.keypoints) is True:
                if len(corrected_label.keypoints) != 1:
                    raise ValueError("Only one keypoint is supported")
                else:
                    x_offset, y_offset = get_point_offset(corrected_label, hA_prediction_tiled)
                    if x_offset > 0 or y_offset > 0:
                        logger.info(f"Label {corrected_label.id} was moved by {x_offset}, {y_offset}")
                    ## Now move the original label by the offset
                    shift_keypoint_label(corrected_label, hA_prediction, x_offset, y_offset)

            elif isinstance(corrected_label.bbox_polygon, shapely.Polygon):
                raise NotImplementedError("Not yet implemented")
            else:
                raise ValueError("Unknown label type")

    # remove deleted labels, TODO this could be more efficient if throug a mapping/trick the parent images would be retrieved
    for iL in iCLdl_deletions:
        for i, annotated_image in enumerate(hA_prediction.images):
            for l in annotated_image.labels:
                if l.id == iL.id:
                    hA_prediction.images[i].labels.remove(l)
                    logger.info(f"Deleted label {l.id}")


    hA_prediction.save(config.corrected_path / f"{config.dataset_name}_hasty_corrected.json")



if __name__ == "__main__":
    main()