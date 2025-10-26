from pathlib import Path

from active_learning.config.dataset_filter import DatasetFilterConfig
from com.biospheredata.types.status import LabelingStatus, AnnotationType

base_path = Path("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style")
images_path = base_path
labels_name = base_path / Path("delplanque_hasty.json")
hasty_annotations_images_zipped = "hasty_style.zip"
annotation_types = [AnnotationType.BOUNDING_BOX]
class_filter = ["Alcelaphinae", "Buffalo", "Kob", "Warthog", "Waterbuck", "Elephant"]
label_mapping = {"Alcelaphinae": 1, "Buffalo": 2, "Kob": 3, "Warthog": 4, "Waterbuck": 5, "Elephant": 6}

crop_size = 512
empty_fraction = 0.0
overlap = 160
VISUALISE_FLAG = False
use_multiprocessing = True
edge_black_out = False # each box which is on the edge will be marked black
num = None # Amount of image to take

labels_path = base_path / f"Delplanque2022_{crop_size}_overlap_{overlap}_eb{edge_black_out}"
flatten = True
unpack = False

train_delplanque = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "delplanque_train",
    "dataset_filter": ["delplanque_train"],
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
