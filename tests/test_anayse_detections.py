"""
Take Detections from a model and
"""
from pathlib import Path

# import pytest

from active_learning.analyse_detections_yolo import analyse_point_detections_
from active_learning.pipelines.predict import YoloPredictor
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file
from examples.review_annotations import debug_hasty_fiftyone_v3






