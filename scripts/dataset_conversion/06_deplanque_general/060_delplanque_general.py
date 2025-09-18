"""
/Volumes/G-DRIVE/Datasets/africa_elephants_uliege/general_dataset


"""
import json

import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from active_learning.util.visualisation.annotation_vis import create_simple_histograms, \
    visualise_hasty_annotation_statistics
from com.biospheredata.types.status import LabelingStatus
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger
from pathlib import Path

from scripts.dataset_conversion.sanity_check import get_dataset_stats

base_path = Path("/raid/cwinkelmann/training_data/delplanque/general_dataset")
destination_base_path = base_path / "hasty_style"
annotations_path = base_path / "groundtruth/json/big_size/"

train_big_size = annotations_path / "train_big_size_A_B_E_K_WH_WB.json"
val_big_size = annotations_path / "val_big_size_A_B_E_K_WH_WB.json"
test_big_size = annotations_path / "test_big_size_A_B_E_K_WH_WB.json"

train_images_path = base_path / "train"
val_images_path = base_path / "val"
test_images_path = base_path / "test"


data_splits = [
    {"base_path": train_images_path, "annotations": train_big_size, "split": "train"},
    {"base_path": val_images_path, "annotations": val_big_size, "split": "val"},
    {"base_path": test_images_path, "annotations": test_big_size, "split": "test"},
]
annotations = base_path / 'annotations_images.csv'

annotated_images = []
label_classes = set()


def coco_to_hasty_label(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert a COCO-style annotation file to HastyAnnotationV2 format.
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    df_images = pd.DataFrame(images)
    df_annotations = pd.DataFrame(annotations)
    df_categories = pd.DataFrame(categories)

    return df_images, df_annotations, df_categories


for item in data_splits:

    split = item["split"]
    base_path = item["base_path"]
    annotations = item["annotations"]

    dataset_name = f"delplanque_{split}"

    logger.info(f"Working on dataset: {dataset_name} with split: {split}")

    (destination_base_path / dataset_name).mkdir(parents=True, exist_ok=True)
    loaded_train_images = [i for i in (base_path / split).glob('*.JPG') if not i.name.startswith('._')]

    df_images, df_annotations, df_categories = coco_to_hasty_label(annotations)
    mapping = df_categories.set_index('id')['name'].to_dict()

    for _, l in df_images.iterrows():
        df_image_annotations = df_annotations[df_annotations['image_id'] == l.get('id')]
        logger.info(f"Processing image: {l.get('file_name')}, id: {l.get('id')} with {len(df_image_annotations)} annotations")
        labels = []

        for _, row in df_image_annotations.iterrows():
            x1, y1, w, h = row["bbox"]
            x2 = x1 + w
            y2 = y1 + h
            species = mapping[row["category_id"]]
            il = ImageLabel(
                id=str(row.id),
                class_name=species,
                bbox=[x1,y1,x2,y2]
            )

            labels.append(il)

        image_id = get_image_id(base_path / l['file_name'])
        width, height = l['width'], l['height']

        annotated_image = AnnotatedImage(
                    image_id=image_id,
                    image_name=l['file_name'],
                    dataset_name=dataset_name,
                    ds_image_name=None,
                    width=width,
                    height=height,
                    image_status=LabelingStatus.COMPLETED,
                    tags=[],
                    image_mode=None,
                    labels=labels
                )

        annotated_images.append(annotated_image)

        if not (destination_base_path / dataset_name / l['file_name']).exists():
            shutil.copy(base_path/ l['file_name'], destination_base_path / dataset_name / l['file_name'])

    obj_lc = [
        LabelClass(
            class_name=lc,
            class_id=str(i),
            color=f"#{hash(lc) & 0xFFFFFF:06x}",
            class_type="object",
            norder=i
        )
        for i, lc in mapping.items()
    ]

hA = HastyAnnotationV2(
        project_name="eikelboom2019",
        images=annotated_images,
        export_format_version="1.1",
        label_classes=obj_lc
    )


# create plots for the dataset
create_simple_histograms(annotated_images, dataset_name="Delplanque General Dataset")
visualise_hasty_annotation_statistics(annotated_images)

dataset_path = destination_base_path / "delplanque_hasty.json"
hA.save(dataset_path)

logger.info(f"Saved Hasty Annotation to {dataset_path}")

logger.info(get_dataset_stats(dataset_path))