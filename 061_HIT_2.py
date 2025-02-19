"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
from typing import List

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detection_correction
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage
from examples.review_annotations import debug_hasty_fiftyone_v2
import fiftyone as fo
from loguru import logger



def main():
    analysis_date = "2024_12_09"
    type = "points"
    dataset_name = f"eal_{analysis_date}_review"
    base_path = Path(f'/Users/christian/data/orthomosaics/tiles')

    images_path = base_path

    output_path = base_path / "object_crops"
    hA_before_path = output_path / f"{dataset_name}_hasty.json"

    stats_path, downloaded_corrections = cvat2hasty(hA_before_path, dataset_name)

    ## TODO and now project the points back to the original image
    for i, correction in enumerate(downloaded_corrections.images):
        # crop_out_individual_object(correction, images_path, output_path)
        pass
    analyse_point_detection_correction



if __name__ == "__main__":
    main()