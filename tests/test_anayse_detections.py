"""
Take Detections from a model and
"""
from pathlib import Path

# import pytest

from active_learning.analyse_detections_yolo import analyse_point_detections_
from active_learning.pipelines.yolo_predict import YoloPredictor
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file
from examples.review_annotations import debug_hasty_fiftyone_v3

# TODO implement this


"""
testing various data preparation steps
"""
import json

import tempfile

import pytest
import numpy as np
from pathlib import Path



@pytest.fixture
def iSAID_annotations_path():
    return Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/iSAID/train/Annotations/iSAID_train.json")


@pytest.fixture
def herdnet_point_detections():
    p = Path("/home/christian/hnee/HerdNet/data_iguana/val/20240824_HerdNet_results/20240824_detections.csv")


@pytest.fixture
def herdnet_ground_truth():
    p = Path("/home/christian/hnee/HerdNet/data_iguana/val/herdnet_format.csv")





def test_analyse_predictions():
    """
    test the analysis of predictions
    :return: 
    """

    # TODO take predictions

    # filter for false positives, high confidence by using a simple dataframe exported by the intermediary annotations
    df_predictions = [
        {"image_name": "image1.jpg", "x": 100, "y": 100, "confidence": 0.9, "label": "iguana"},
    ]