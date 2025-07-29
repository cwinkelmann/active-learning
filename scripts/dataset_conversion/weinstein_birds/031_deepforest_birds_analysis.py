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


hA = HastyAnnotationV2.from_file(destination_base_path / "weinstein_birds_hasty.json")


dataset_names = set(ai.dataset_name for ai in hA.images)

for split in dataset_names:

    annotated_images_split = [ai for ai in hA.images if ai.dataset_name == split]
    # create_simple_histograms(annotated_images_split)
    plot_bbox_sizes(annotated_images_split, dataset_name=split, plot_name = f"box_sizes_{split}.png")

    # create plots for the dataset
    create_simple_histograms(annotated_images_split, dataset_name=split)
    visualise_hasty_annotation_statistics(annotated_images_split)