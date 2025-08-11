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
    visualise_hasty_annotation_statistics, plot_bbox_sizes
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger
from pathlib import Path

base_path = Path("/Volumes/G-DRIVE/Datasets/africa_elephants_uliege/general_dataset/")
destination_base_path = base_path / "hasty_style"

hA = HastyAnnotationV2.from_file(destination_base_path / "delplanque_hasty.json")
annotated_images = hA.images

# create plots for the dataset
create_simple_histograms(annotated_images)
visualise_hasty_annotation_statistics(annotated_images)


for split in ["delplanque_train", "delplanque_val", "delplanque_test"]:

    annotated_images_split = [ai for ai in annotated_images if ai.dataset_name == split]
    create_simple_histograms(annotated_images_split)
    plot_bbox_sizes(annotated_images_split, suffix=f"Delplanque {split}", plot_name=f"box_sizes_Delplanque_{split}.png")
    # visualise_hasty_annotation_statistics(annotated_images_split)