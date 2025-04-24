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
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty, cvat2hasty_v2, download_cvat_annotations, \
    foDataset2Hasty, determine_changes
from active_learning.types.Exceptions import TooManyLabelsError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabelCollection, ImageLabel, KeypointSchema
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




def main(dataset_name, base_path: Path, hA_reference: HastyAnnotationV2):

    images_path = base_path


    images_path = base_path

    output_path = base_path / "object_crops"

    # original predictions
    # hA_prediction_path = output_path / f"{dataset_name}_hasty.json"
    # hA_prediction = hA_from_file(file_path=hA_prediction_path)

    # the 256px crops
    # hA_prediction_tiled_path = output_path / f"{dataset_name}_tiled_hasty.json"
    # hA_prediction_tiled = hA_from_file(file_path=hA_prediction_tiled_path)

    view, dataset = download_cvat_annotations(dataset_name=dataset_name)

    hA_updated = foDataset2Hasty(hA_template=hA_reference, dataset=dataset)

    # TODO implement this analysis
    changes = determine_changes(hA_reference, hA_updated)

    hA_updated.save(output_path / f"{dataset_name}_hasty_corrected.json")

    # create a shapefile from this



if __name__ == "__main__":
    df_detections = pd.read_csv(
        '/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/detections.csv')

    # TODO get this template too to persist the converted images later
    base_path = Path(f'/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/Fer_FCD01-02-03_tiles')

    analysis_date = "2025_04_23"
    output_path = Path(base_path) / "output"
    dataset_name = f"eal_{analysis_date}_{base_path.name}_review"
    # replace '-' with '_'
    dataset_name = dataset_name.replace("-", "_")

    images_path = base_path
    images_jpg_path = images_path / "jpg"
    images_jpg_path.mkdir(parents=True, exist_ok=True)


    hA_reference = hA_from_file(output_path / f"{dataset_name}_tiled_hasty.json")
    # hA_reference.images = hA_reference.images[:1]
    # TODO add the two extra keypoint Classes

    hA_reference

    main(dataset_name=dataset_name, base_path=base_path, hA_reference=hA_reference)