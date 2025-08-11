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

base_path = Path("/Volumes/G-DRIVE/Datasets/deep_forest_birds/hasty_style")
labels_name = base_path / Path("weinstein_birds_hasty.json")
images_path = base_path

annotation_types = [AnnotationType.BOUNDING_BOX]
class_filter = ["Bird"]

crop_size = 1024
empty_fraction = 0.0
overlap = 0
VISUALISE_FLAG = False
use_multiprocessing = True
edge_black_out = True
num = 10

labels_path = base_path / f"labels_{crop_size}_overlap_{overlap}"

michigan_train = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "michigan_birds_edge_blackout_train",
    # "dataset_filter": ["everglades_train", "hayes_train", "mckellar_train", "michigan_train"],
    "dataset_filter": ["michigan_train"],
    # "images_filter": ["C3_L10_F589_T20200925_132438_149_27.png"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})
michigan_val = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "michigan_birds_edge_blackout_val",
    # "dataset_filter": ["everglades_test", "hayes_test", "mckellar_test", "michigan_test"],
    "dataset_filter": ["michigan_test"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})

everglades_train = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "everglades_train",
    # "dataset_filter": ["everglades_train", "hayes_train", "mckellar_train", "michigan_train"],
    "dataset_filter": ["everglades_train"],
    # "images_filter": ["C3_L10_F589_T20200925_132438_149_27.png"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})
everglades_val = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "birds_edge_blackout_val",
    # "dataset_filter": ["everglades_test", "hayes_test", "mckellar_test", "michigan_test"],
    "dataset_filter": ["michigan_test"],
    # "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})



dataset_names = set(ai.dataset_name for ai in HastyAnnotationV2.from_file(labels_name).images)
dataset_names_train = [dset for dset in dataset_names if "train" in dset]
dataset_names_val = [dset for dset in dataset_names if "test" in dset]

# everglades_cross_validation
dataset_everglades_cv_names_train = [dset for dset in dataset_names if "train" in dset and "everglades_train" and dset != "everglades_train"]
dataset_everglades_cv_names_val = [dset for dset in dataset_names if "test" in dset and dset == "everglades_test"]

all_train = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "all_birds_train",
    # "dataset_filter": ["everglades_train", "hayes_train", "mckellar_train", "michigan_train"],
    "dataset_filter": dataset_names_train,
    # "images_filter": ["C3_L10_F589_T20200925_132438_149_27.png"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})

# all birds validation
all_val = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "all_birds_val",
    # "dataset_filter": ["everglades_test", "hayes_test", "mckellar_test", "michigan_test"],
    "dataset_filter": dataset_names_val,
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})

# all birds train but everglades
everglades_cv_train = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "all_birds_train_but_everglades",
    "dataset_filter": dataset_everglades_cv_names_train,
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})

# train with only 1000 everglades birds
everglades_cv_train_local_1000 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "birds_everglades_1000_train",
    "dataset_filter": ["everglades_train"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "num_labels": 1000,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": True,
    "remove_padding": True
})

# everglades all validationds
everglades_cv_val = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "birds_everglades_val",
    "dataset_filter": ["everglades_test"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "num": num,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})


train_everglades_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Everglades_il_{x}",
    "dataset_filter": ["everglades_train"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": x,
    "remove_Default": True,
    "remove_padding": True
}) for x in range(15, 300, 10)]

train_michigan_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Michigan_il_{x}",
    "dataset_filter": ["michigan_train"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": x,
    "remove_Default": True,
    "remove_padding": True
}) for x in range(15, 300, 10)]

# datasets = [all_train, all_val]
# datasets = [michigan_train, michigan_val]
# datasets = [all_val]
datasets = [michigan_train, michigan_val, ]
datasets = [
    # michigan_train, michigan_val,
    # michigan_train, michigan_val,
    # all_train, all_val,
    # everglades_cv_train_local_1000,
    #
    # everglades_cv_val,


]
datasets = datasets + train_everglades_increasing_length + train_michigan_increasing_length