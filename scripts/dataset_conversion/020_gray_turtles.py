
"""
process the grey turtles dataset for hasty

The dataset is actually a classification where each tile is either a certain turtle or numbers between 1 and 5 which is not yet explained.

"""

import uuid

import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

base_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/GreyTurtles/')

annotations = base_path / 'turtle_image_metadata.csv'
df_annotations = pd.read_csv(annotations)

df_certain_turtles = df_annotations[df_annotations.label == "Certain Turtle"]
df_images_unique = df_certain_turtles['file_location'].unique()

# TODO they have only class labels for the crops and it seems there are only 11 images with certain turtles


for split, images in data_splits.items():

    loaded_train_images = images_train.glob('*.JPG')

    for l in loaded_train_images:
        logger.info(f"Images: {l.name}")

        df_image_annotations = df_annotations[df_annotations['FILE'] == l.name]
        labels = []
        for _, row in df_image_annotations.iterrows():
            il = ImageLabel(
                id=str(uuid.uuid4()),
                class_name=row.SPECIES,
                bbox=[row.x1, row.y1, row.x2, row.y2],
            )
            label_classes.add(row.SPECIES)
            labels.append(il)

        image_id = get_image_id(l)
        width, height = get_image_dimensions(l)

        annotated_image = AnnotatedImage(
                    image_id=image_id,
                    image_name=l.name,
                    dataset_name=f"eikelboom_{split}",
                    ds_image_name=None,
                    width=width,
                    height=height,
                    image_status="Done",
                    tags=[],
                    image_mode=None,
                    labels=labels
                )

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
        project_name="eikelboom2019",
        images=annotated_images,
        export_format_version="1.1",
        label_classes=obj_lc
    )

hA.save(base_path / "eikelboom_hasty.json")