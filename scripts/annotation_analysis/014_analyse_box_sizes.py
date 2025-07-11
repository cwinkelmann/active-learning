"""
Analyse the box sizes of a Hasty Annotation dataset.


"""
import json

import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from active_learning.util.visualisation.annotation_vis import create_simple_histograms, \
    visualise_hasty_annotation_statistics, plot_bbox_sizes
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger
from pathlib import Path

base_path = Path("/Users/christian/data/training_data/2025_07_10_final_analysis/unzipped_hasty_annotation")


hA = HastyAnnotationV2.from_file(base_path / "2025_07_10_labels_final.json")
annotated_images = hA.images

annotated_images = [ai for ai in annotated_images if ai.dataset_name not in [
    "Zooniverse_expert_phase_3", "Zooniverse_expert_phase_2"]]

# keep only "iguana"
annotated_images = [
    AnnotatedImage(
        **{**ai.__dict__, 'labels': [label for label in ai.labels if label.class_name == "iguana"]}
    )
    for ai in annotated_images
]
# create plots for the dataset
create_simple_histograms(annotated_images)
visualise_hasty_annotation_statistics(annotated_images)

dataset_names = set(ai.dataset_name for ai in annotated_images)

for split in dataset_names:

    annotated_images_split = [ai for ai in annotated_images if ai.dataset_name == split]
    # create_simple_histograms(annotated_images_split)
    plot_bbox_sizes(annotated_images_split, dataset_name=split, plot_name = f"box_sizes_{split}.png")
    # visualise_hasty_annotation_statistics(annotated_images_split)