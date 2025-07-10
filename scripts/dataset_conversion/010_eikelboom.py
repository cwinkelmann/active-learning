import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger

base_path = Path('/Volumes/G-DRIVE/Datasets/ImprovingPrecisionAccuracy_Eikelboom2019data/Improving the precision and accuracy of animal population estimates with aerial image object detection_1_all')

images_train = base_path / 'train'
images_val = base_path / 'val'
images_test = base_path / 'test'

data_splits = {
    'train': images_train,
    'val': images_val,
    'test': images_test
}

annotations = base_path / 'annotations_images.csv'

# annotations which are based on images which are not in the folder
annotations_train = base_path / 'annotations_train.csv'
annotations_val = base_path / 'annotations_val.csv'
annotations_test = base_path / 'annotations_test.csv'

annotated_images = []
label_classes = set()

for split, images in data_splits.items():

    loaded_train_images = [i for i in images_train.glob('*.JPG') if not i.name.startswith('._')]

    df_annotations = pd.read_csv(annotations)

    for l in loaded_train_images:
        logger.info(f"Processing image: {l.name}")
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