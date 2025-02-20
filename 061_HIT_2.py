"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
import shapely
from typing import List, Optional

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detection_correction, analyse_point_detections_greedy
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty
from active_learning.types.Exceptions import TooManyLabelsError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabelCollection, ImageLabel
from examples.review_annotations import debug_hasty_fiftyone_v2
import fiftyone as fo
from loguru import logger


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
    analysis_date = "2024_12_09"
    type = "points"
    dataset_name = f"eal_{analysis_date}_review"
    base_path = Path(f'/Users/christian/data/orthomosaics/tiles')

    images_path = base_path

    output_path = base_path / "object_crops"

    # original predictions
    hA_prediction_path = output_path / f"{dataset_name}_hasty.json"
    hA_prediction = hA_from_file(file_path=hA_prediction_path)

    # the 256px crops
    hA_prediction_tiled_path = output_path / f"{dataset_name}_tiled_hasty.json"
    hA_prediction_tiled = hA_from_file(file_path=hA_prediction_tiled_path)

    df_stats, hA_prediction_tiled_corrected, iCLdl_deletions = cvat2hasty(hA_prediction_tiled, dataset_name)

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
                    kp = corrected_label.keypoints[0]

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


    hA_prediction.save(output_path / f"{dataset_name}_hasty_corrected.json")



if __name__ == "__main__":
    main()