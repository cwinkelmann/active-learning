"""
/Volumes/G-DRIVE/Datasets/africa_elephants_uliege/general_dataset
 THe dataset is broken in part. THe training_images.csv contains image ids which are in the test set.

"""
import json

import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from active_learning.util.visualisation.annotation_vis import create_simple_histograms, \
    visualise_hasty_annotation_statistics
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass, \
    Keypoint

from loguru import logger
from pathlib import Path

from scripts.dataset_conversion.sanity_check import get_dataset_stats

keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"
dataset_base_path = Path("/raid/cwinkelmann/training_data/AED")
destination_base_path = dataset_base_path / "hasty_style"
annotations_path = dataset_base_path

train_big_size = annotations_path / "training_elephants.csv"
test_big_size = annotations_path / "test_elephants.csv"

train_images_path = dataset_base_path / "all_images"
test_images_path = dataset_base_path / "all_images"


data_splits = [
    {"base_path": train_images_path, "annotations": train_big_size, "split": "train", "images_metadata": dataset_base_path / "training_images.csv"},
    {"base_path": test_images_path, "annotations": test_big_size, "split": "test", "images_metadata": dataset_base_path / "test_images.csv"},
]
annotations = dataset_base_path / 'annotations_images.csv'

annotated_images = []
label_classes =["Elephant"]


for item in data_splits:

    split = item["split"]
    base_path = item["base_path"]
    images_metadata = item["images_metadata"]
    annotations = item["annotations"]

    dataset_name = f"aed_{split}"
    destination_base_path.joinpath(dataset_name).mkdir(parents=True, exist_ok=True)
    logger.info(f"Working on dataset: {dataset_name} with split: {split}")

    df_image_metadata = pd.read_csv(
        images_metadata,
        names=[
            'image_name',
            'sortie_id',
            'image_width',
            'image_height',
            'gsd',
            'measured_altitude',
            'terrain_altitude',
            'gps_altitude'
        ],
        header=None  # Add this if the CSV doesn't have a header row
    )

    df_annotations = pd.read_csv(annotations, names=['image_id', "x", "y"])

    df_merged = df_annotations.merge(df_image_metadata, left_on='image_id', right_on='image_name', how='left')

    for image_id, df_image in df_merged.groupby("image_name"):

        labels = []
        df_image
        for _, row in df_image.iterrows():
            hkp = Keypoint(
                x=int(row.x),
                y=int(row.y),
                norder=0,
                keypoint_class_id=keypoint_class_id,
            )

            il = ImageLabel(
                id=str(uuid.uuid4()),
                class_name="Elephant",
                keypoints=[hkp],
            )

            labels.append(il)
        full_image_name = f"{image_id}.jpg"
        image_path = base_path / full_image_name

        width, height = row.image_width, row.image_height

        annotated_image = AnnotatedImage(
            image_id=str(image_id),
            image_name=full_image_name,
            dataset_name=dataset_name,
            ds_image_name=None,
            width=width,
            height=height,
            image_status="COMPLETED",
            tags=[],
            image_mode=None,
            labels=labels
        )

        annotated_images.append(annotated_image)

        if not (destination_base_path / dataset_name / full_image_name).exists():
            shutil.copy(base_path / full_image_name, destination_base_path / dataset_name / full_image_name)

    obj_lc = [
        LabelClass(
            class_name=lc,
            class_id=str(uuid.uuid4()),
            color=f"#{hash(lc) & 0xFFFFFF:06x}",
            class_type="object",
            norder=i
        )
        for i, lc in enumerate(label_classes)
    ]

hA = HastyAnnotationV2(
    project_name="AED",
    images=annotated_images,
    export_format_version="1.1",
    label_classes=obj_lc
)



hA.save(dataset_base_path / "aed_hasty.json")

logger.info(f"Saved Hasty Annotation to {dataset_base_path / 'aed_hasty.json'}")

logger.info(get_dataset_stats(dataset_base_path / 'aed_hasty.json'))