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

base_path = Path("/Volumes/G-DRIVE/Datasets/ImprovingPrecisionAccuracy_Eikelboom2019data/Improving the precision and accuracy of animal population estimates with aerial image object detection_1_all/hasty_style")


hA = HastyAnnotationV2.from_file(base_path / "eikelboom_hasty.json")
annotated_images = hA.images

# create plots for the dataset
create_simple_histograms(annotated_images)
visualise_hasty_annotation_statistics(annotated_images)



dataset_names = set(ai.dataset_name for ai in annotated_images)

for split in dataset_names:
    annotated_images_split = [ai for ai in annotated_images if ai.dataset_name == split]
    # create_simple_histograms(annotated_images_split)
    plot_bbox_sizes(annotated_images_split, dataset_name=split, plot_name=f"box_sizes_{split}.png")
    # visualise_hasty_annotation_statistics(annotated_images_split)