import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger

base_path = Path('/Volumes/G-DRIVE/Datasets/deep_forest_birds')
destination_base_path = Path('/Volumes/G-DRIVE/Datasets/deep_forest_birds/hasty_style')

everglades_subset = base_path / 'everglades'
everlades_train_annotations = everglades_subset / "everglades_train.csv"
everlades_test_annotations = everglades_subset / "everglades_test.csv"



hayes_subset = base_path / 'hayes'
pfeifer_subset = base_path / 'pfeifer'
tern_subset = base_path / 'terns'

hayes_train_annotations = hayes_subset / "hayes_train.csv"
hayes_test_annotations = hayes_subset / "hayes_test.csv"

pfeifer_train_annotations = pfeifer_subset / "pfeifer_train.csv"
pfeifer_test_annotations = pfeifer_subset / "pfeifer_test.csv"

tern_train_annotations = tern_subset / "terns_train.csv"
tern_test_annotations = tern_subset / "terns_test.csv"

subset_dict = [
    {"base_path": everglades_subset, "annotations": everlades_train_annotations, "split": "train"},
    {"base_path": everglades_subset, "annotations": everlades_test_annotations, "split": "test"},

    {"base_path": hayes_subset, "annotations": hayes_train_annotations, "split": "train"},
    {"base_path": hayes_subset, "annotations": hayes_test_annotations, "split": "test"},

    {"base_path": pfeifer_subset, "annotations": pfeifer_train_annotations, "split": "train"},
    {"base_path": pfeifer_subset, "annotations": pfeifer_test_annotations, "split": "test"},

    {"base_path": tern_subset, "annotations": tern_train_annotations, "split": "train"},
    {"base_path": tern_subset, "annotations": tern_test_annotations, "split": "test"}
]


annotated_images = []
label_classes = set()

for split in subset_dict:
    logger.info(f"Processing split: {split}")
    loaded_train_images = [i for i in split["base_path"].glob('*.png') if not i.name.startswith('._')]
    annotations_path = split["annotations"]
    split_name = split["split"]
    base_path = split["base_path"]

    df_annotations = pd.read_csv(annotations_path)

    for image_name in df_annotations["image_path"].unique():
        dataset_name = f"{base_path.name}_{split_name}"
        (destination_base_path / dataset_name).mkdir(parents=True, exist_ok=True)

        df_image_annotations = df_annotations[df_annotations['image_path'] == image_name]
        labels = []
        for _, row in df_image_annotations.iterrows():
            il = ImageLabel(
                id=str(uuid.uuid4()),
                class_name=row.label,
                bbox=[row.xmin, row.ymin, row.xmax, row.ymax],
            )
            label_classes.add(row.label)
            labels.append(il)

        image_id = get_image_id(base_path / image_name)
        width, height = get_image_dimensions(base_path / image_name)

        annotated_image = AnnotatedImage(
                    image_id=image_id,
                    image_name=image_name,
                    dataset_name=dataset_name,
                    ds_image_name=None,
                    width=width,
                    height=height,
                    image_status="COMPLETED",
                    tags=[],
                    image_mode=None,
                    labels=labels
                )
        if not (destination_base_path / dataset_name / image_name).exists():
            shutil.copy(base_path / image_name, destination_base_path / dataset_name / image_name)

        annotated_images.append(annotated_image)

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
        project_name="weinstein_birds",
        images=annotated_images,
        export_format_version="1.1",
        label_classes=obj_lc
    )

hA.save(base_path / "weinstein_birds_hasty.json")