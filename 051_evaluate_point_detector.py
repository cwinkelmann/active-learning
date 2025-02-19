"""
Take Detections from a model, compare them with ground truth and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
import typing

import PIL.Image
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
# import pytest

from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import evaluate_in_fifty_one
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file


if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2024_12_09"
    # lcrop_size = 640
    num = 56
    type = "points"

    # test dataset
    base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    images_path = base_path / "Default"

    # IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)
    hA_ground_truth_path = base_path / 'hasty_format_iguana_point.json'

    df_ground_truth = pd.read_csv(base_path / 'herdnet_format.csv')

    # # val dataset
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/19-17-14/detections.csv')
    # df_ground_truth = pd.read_csv('/Users/christian/data/training_data/2024_12_09_debug/val/herdnet_format.csv')
    # images_path = Path("/Users/christian/data/training_data/2024_12_09_debug/val/Default")

    hA_ground_truth = hA_from_file(hA_ground_truth_path)
    images = images_path.glob("*.JPG")

    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
        df_detections=df_detections, df_ground_truth=df_ground_truth, radius=150)

    df_concat = pd.concat([df_false_positives, df_true_positives, df_false_negatives])


    IL_all_detections = herdnet_prediction_to_hasty(df_concat, images_path)
    IL_fp_detections = herdnet_prediction_to_hasty(df_false_positives, images_path)
    IL_tp_detections = herdnet_prediction_to_hasty(df_true_positives, images_path)
    IL_fn_detections = herdnet_prediction_to_hasty(df_false_negatives, images_path)


    dataset_name = f"eal_{analysis_date}_review"

    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, "
                f"False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")

    box_size = 350

    for i in images:
        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]

        draw_thumbnail(df_fp, i, suffix="fp", images_path=images_path, box_size=box_size)
        draw_thumbnail(df_fn, i, suffix="fn", images_path=images_path, box_size=box_size)
        draw_thumbnail(df_tp, i, suffix="tp", images_path=images_path, box_size=box_size)

    # TODO draw a curve: x: confidence, y: precision, recall, f1, MAE, MSE

    images_set = [images_path / i.image_name for i in hA_ground_truth.images]

    evaluate_in_fifty_one(dataset_name,
                          images_set,
                          hA_ground_truth,
                          IL_fp_detections,
                          IL_fn_detections,
                          IL_tp_detections,
                          type="points")


