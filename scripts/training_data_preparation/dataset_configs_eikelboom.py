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
from com.biospheredata.types.status import LabelingStatus, AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import (visualise_image, visualise_polygons)

base_path = Path("/raid/cwinkelmann/training_data/eikelboom2019")

labels_name = base_path / "eikelboom_hasty.json"
images_path = base_path
annotation_types = [AnnotationType.BOUNDING_BOX]
class_filter = ["Giraffe", "Elephant", "Zebra"]
label_mapping = {"Giraffe":1, "Elephant":2, "Zebra":3}
crop_size = 512
empty_fraction = 0.0
overlap = 0
VISUALISE_FLAG = False
use_multiprocessing = True
edge_black_out = False
num = None

labels_path = base_path / f"eikelboom_{crop_size}_overlap_{overlap}_eb{edge_black_out}"
flatten = True
unpack = False

train_eikelboom = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "eikelboom_train",
    "dataset_filter": ["eikelboom_train"],
    #"images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_eikelboom = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "eikelboom_val",
    "dataset_filter": ["eikelboom_val"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "num": 10
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
test_eikelboom = DatasetFilterConfig(**{
    "dset": "test",
    "dataset_name": "eikelboom_test",
    "dataset_filter": ["eikelboom_test"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "num": 10
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

datasets = [train_eikelboom, val_eikelboom, test_eikelboom]
# datasets = [train_eikelboom]