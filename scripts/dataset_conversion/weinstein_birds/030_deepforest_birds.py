import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from active_learning.util.visualisation.annotation_vis import plot_bbox_sizes, create_simple_histograms, \
    visualise_hasty_annotation_statistics
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

    {"base_path": base_path / "mckellar", "annotations": base_path / "mckellar" / "mckellar_train.csv",
     "split": "train"},
    {"base_path": base_path / "mckellar", "annotations": base_path / "mckellar" / "mckellar_test.csv", "split": "test"},

    {"base_path": base_path / "michigan", "annotations": base_path / "michigan" / "michigan_train.csv",
     "split": "train"},
    {"base_path": base_path / "michigan", "annotations": base_path / "michigan" / "michigan_test.csv", "split": "test"},

    {"base_path": base_path / "neill", "annotations": base_path / "neill" / "neill_train.csv",
     "split": "train"},
    {"base_path": base_path / "neill", "annotations": base_path / "neill" / "neill_test.csv", "split": "test"},

    {"base_path": base_path / "newmexico", "annotations": base_path / "newmexico" / "newmexico_train.csv",
     "split": "train"},
    {"base_path": base_path / "newmexico", "annotations": base_path / "newmexico" / "newmexico_test.csv", "split": "test"},

    {"base_path": base_path / "palmyra", "annotations": base_path / "palmyra" / "palmyra_train.csv",
     "split": "train"},
    {"base_path": base_path / "palmyra", "annotations": base_path / "palmyra" / "palmyra_test.csv", "split": "test"},

    {"base_path": base_path / "penguins", "annotations": base_path / "penguins" / "penguins_train.csv",
     "split": "train"},
    {"base_path": base_path / "penguins", "annotations": base_path / "penguins" / "penguins_test.csv", "split": "test"},

    {"base_path": pfeifer_subset, "annotations": pfeifer_train_annotations, "split": "train"},
    {"base_path": pfeifer_subset, "annotations": pfeifer_test_annotations, "split": "test"},

    {"base_path": tern_subset, "annotations": tern_train_annotations, "split": "train"},
    {"base_path": tern_subset, "annotations": tern_test_annotations, "split": "test"}
]


annotated_images = []
label_classes = set()

all_annotations = []

for split in subset_dict:
    logger.info(f"Processing split: {split}")
    loaded_train_images = [i for i in split["base_path"].glob('*.png') if not i.name.startswith('._')]
    annotations_path = split["annotations"]
    split_name = split["split"]
    split_base_path = split["base_path"]

    df_annotations = pd.read_csv(annotations_path)
    df_annotations_split = df_annotations.copy()
    df_annotations_split['split'] = split_name
    all_annotations.append(df_annotations_split)

    for image_name in df_annotations["image_path"].unique():
        dataset_name = f"{split_base_path.name}_{split_name}"
        (destination_base_path / dataset_name).mkdir(parents=True, exist_ok=True)

        df_image_annotations = df_annotations[df_annotations['image_path'] == image_name]
        labels = []
        for _, row in df_image_annotations.iterrows():

            if row.xmax - row.xmin == 0 or row.ymax - row.ymin == 0:
                logger.warning(f"Skipping annotation for {image_name} with zero area bbox: {row}")
                continue


            il = ImageLabel(
                id=str(uuid.uuid4()),
                class_name=row.label,
                bbox=[row.xmin, row.ymin, row.xmax, row.ymax],
            )
            label_classes.add(row.label)
            labels.append(il)

        image_id = get_image_id(split_base_path / image_name)
        width, height = get_image_dimensions(split_base_path / image_name)

        logger.info(f"Image {image_name}, size: {width} x {height}")

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
            shutil.copy(split_base_path / image_name, destination_base_path / dataset_name / image_name)

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

df_all_annotations = pd.concat(all_annotations)
for split_name, df_annotations_group in df_all_annotations.groupby("split"):

    df_annotations_group.to_csv(destination_base_path / f"{split_name}_annotations.csv", index=False)


hA.save(destination_base_path / "weinstein_birds_hasty.json")



dataset_names = set(ai.dataset_name for ai in annotated_images)

# for split in dataset_names:
#
#     annotated_images_split = [ai for ai in annotated_images if ai.dataset_name == split]
#     # create_simple_histograms(annotated_images_split)
#     plot_bbox_sizes(annotated_images_split, dataset_name=split, plot_name = f"box_sizes_{split}.png")
#
#     # create plots for the dataset
#     create_simple_histograms(annotated_images_split, dataset_name=split)
#     visualise_hasty_annotation_statistics(annotated_images_split)