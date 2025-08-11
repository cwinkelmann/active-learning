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
from image_template_search.util.util import (visualise_image, visualise_polygons)

## Meeting presentation
base_path = Path("/Volumes/G-DRIVE/Datasets/africa_elephants_uliege/general_dataset/hasty_style")
# hasty_annotations_labels_zipped = "delplanque_hasty.zip"
images_path = base_path
labels_name = base_path / Path("delplanque_hasty.json")
hasty_annotations_images_zipped = "hasty_style.zip"
annotation_types = [AnnotationType.BOUNDING_BOX]
class_filter = ["Alcelaphinae", "Buffalo", "Elephant", "Kob", "Warthog", "Waterbuck", "Elephant"]

crop_size = 518
empty_fraction = 0.0
overlap = 0
VISUALISE_FLAG = False
use_multiprocessing = True
edge_black_out = True
num = None

labels_path = base_path / f"labels_{crop_size}_overlap_{overlap}"

train_delplanque = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "delplanque_train",
    "dataset_filter": ["delplanque_train"],
    #"images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_delplanque = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "delplanque_val",
    "dataset_filter": ["delplanque_val"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "num": 10
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
test_delplanque = DatasetFilterConfig(**{
    "dset": "test",
    "dataset_name": "delplanque_test",
    "dataset_filter": ["delplanque_test"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "num": 10
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

datasets = [train_delplanque, val_delplanque, test_delplanque]
