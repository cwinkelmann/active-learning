"""
HUMAN IN THE LOOP correction of existing annotations in Hasty format and not geospatial format.

Mainly correct for false positives in the prediction which could be false_negatives in the ground truth in point detection annotations.

These false positives can be used for hard negative mining in the next training round.

"""
import copy

from typing import List, Optional

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file

import fiftyone as fo
from loguru import logger

from scripts.human_in_the_loop.helper import hit_fp_cvat_upload

if __name__ == "__main__":
    # fernandina_base_path = Path('/Users/christian/data/training_data/2025_08_10_endgame/Fernandina_s_detection/train')
    #
    # fernandina_report_path = fernandina_base_path / "report"
    # config = DatasetCorrectionConfig(
    #     analysis_date="correcting_fernandina_s",
    #     dataset_name=f"fernandina_s_correction",
    #     type="points",
    #     subset_base_path=fernandina_base_path,
    #     reference_base_path=Path(
    #         '/Users/christian/data/training_data/2025_08_10_label_correction'),
    #     hasty_reference_annotation_name='full_label_correction_floreana_2025_07_10_train_correction_hasty_corrected_1.json',
    #
    #     # the full dataset annotations
    #     hasty_ground_truth_annotation_name='hasty_format_full_size.json',
    #     # the hasty annotations ground truth we created detections for
    #     herdnet_annotation_name="herdnet_format_points.csv", # should be in base_path
    #     detections_path=Path('/Volumes/2TB/work/training_data_sync/herdnet/outputs/2025-08-11/09-38-28/detections.csv'),
    #     correct_fp_gt=True,
    #     box_size=350,
    #     radius=150,
    #     images_path=fernandina_base_path / "Default",
    #     output_path=fernandina_base_path / "output",
    #     corrected_path=fernandina_base_path / "corrections"
    # )


    fernandina_base_path = Path('/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/Delplanque2022_512_overlap_160_ebFalse/delplanque_train/train')

    delplanqetrain__report_path = fernandina_base_path / "report"
    delplanqetrain = DatasetCorrectionConfig(
        analysis_date="delplanque_train_correction",
        dataset_name=f"Default",
        type="points",
        subset_base_path=fernandina_base_path,
        reference_base_path=Path(
            '/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/'),
        hasty_reference_annotation_name='delplanque_hasty.json',

        # the full dataset annotations
        hasty_ground_truth_annotation_name='hasty_format_full_size.json',
        # the hasty annotations ground truth we created detections for
        herdnet_annotation_name="herdnet_format.csv", # should be in base_path
        detections_path=Path('/raid/cwinkelmann/training_data/delplanque/general_dataset/2025-10-25_delplanque/10-11-12_delplanque_train/detections.csv'),
        correct_fp_gt=True,
        box_size=350,
        radius=150,
        images_path=fernandina_base_path / "Default",
        output_path=fernandina_base_path / "output",
        corrected_path=fernandina_base_path / "corrections"
    )

    hit_fp_cvat_upload(config=delplanqetrain, report_path=delplanqetrain__report_path)
