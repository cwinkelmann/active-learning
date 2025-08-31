import json

import gc

import shutil
import yaml
from loguru import logger
from pathlib import Path
from matplotlib import pyplot as plt

from active_learning.config.dataset_filter import DatasetFilterConfig, DataPrepReport
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.util.visualisation.annotation_vis import visualise_points_only
from com.biospheredata.converter.HastyConverter import AnnotationType, LabelingStatus
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import (visualise_image, visualise_polygons)

## Meeting presentation
base_path = Path("/raid/cwinkelmann/training_data/AED")
# hasty_annotations_labels_zipped = "aed_hasty.zip"
images_path = base_path / "hasty_style"
labels_name = base_path / Path("aed_hasty.json")
hasty_annotations_images_zipped = "hasty_style.zip"
annotation_types = [AnnotationType.KEYPOINT]
class_filter = ["Elephant"]
label_mapping = {"Elephant": 1}
crop_size = 512
empty_fraction = 0.0
overlap = 0
VISUALISE_FLAG = False
use_multiprocessing = True
edge_black_out = False # each box which is on the edge will be marked black
num = None # Amount of image to take

labels_path = base_path / f"AED_{crop_size}_overlap_{overlap}"
flatten = True
unpack = False

train_aed = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "aed_train",
    "dataset_filter": ["aed_train"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
test_aed = DatasetFilterConfig(**{
    "dset": "test",
    "dataset_name": "aed_test",
    "dataset_filter": ["aed_test"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

datasets = [train_aed, test_aed]
