"""
Take Detections from a model, compare them with ground truth and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
import numpy as np
import typing

import PIL.Image
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
# import pytest

from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import evaluate_in_fifty_one, Evaluator
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file



if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2025_07_10"
    # lcrop_size = 640
    num = 56
    type = "points"


    base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_final_point_detection_edge_black/Floreana_detection/val')

    ## On full size original images
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47_full_Size/detections.csv')
    # hasty_annotation_name = 'hasty_format_full_size.json'
    # herdnet_annotation_name = 'herdnet_format.csv'
    # images_path = base_path / "Default"
    suffix = "JPG"

    ## On cropped images
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47/detections.csv')
    hasty_annotation_name = 'hasty_format_crops_512_0.json'
    herdnet_annotation_name = 'herdnet_format_512_0_crops.csv'
    images_path = base_path / "crops_512_numNone_overlap0"
    suffix = "jpg"

    box_size = 350
    radius = 150

    visualisations_path = base_path / "visualisations"
    visualisations_path.mkdir(exist_ok=True, parents=True)
    # IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)
    hA_ground_truth_path = base_path / hasty_annotation_name
    hA_ground_truth = hA_from_file(hA_ground_truth_path)

    df_ground_truth = pd.read_csv(base_path / herdnet_annotation_name)

    images = list(images_path.glob(f"*.{suffix}"))
    if len(images) == 0:
        raise FileNotFoundError("No images found in: " + images_path)

    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=radius
    )



    df_concat = pd.concat([df_false_positives, df_true_positives, df_false_negatives])

    IL_all_detections = herdnet_prediction_to_hasty(df_concat, images_path)
    IL_fp_detections = herdnet_prediction_to_hasty(df_false_positives, images_path)
    IL_tp_detections = herdnet_prediction_to_hasty(df_true_positives, images_path)
    IL_fn_detections = herdnet_prediction_to_hasty(df_false_negatives, images_path)


    dataset_name = f"eal_{analysis_date}_review"

    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, "
                f"False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")

    # TODO draw a curve: x: confidence, y: precision, recall, f1, MAE, MSE
    ev = Evaluator(df_detections=df_detections, df_ground_truth=df_ground_truth, radius=radius)
    ev.get_precition_recall_curve()
    precision, recall, f1 = ev.get_precision_recall_f1()
    logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    for i in images:
        if len(df_concat[df_concat.images == i.name]) > 0:
            logger.info(f"Drawing thumbnails for {i.name} with {len(df_concat[df_concat.images == i.name])} detections")
        else:
            logger.warning(f"Images not {i.name} found in detections/false negatives, skipping drawing thumbnails")

        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]

        draw_thumbnail(df_fp[df_fp.scores > 0.9], i, suffix="fp_hc", images_path=visualisations_path, box_size=box_size)

        draw_thumbnail(df_fp[df_fp.scores <= 0.9], i, suffix="fp_lc", images_path=visualisations_path, box_size=box_size)
        draw_thumbnail(df_fn, i, suffix="fn", images_path=visualisations_path, box_size=box_size)
        # draw_thumbnail(df_tp, i, suffix="tp", images_path=visualisations_path, box_size=box_size)



    images_set = [images_path / i.image_name for i in hA_ground_truth.images]

    # evaluate_in_fifty_one(dataset_name,
    #                       images_set,
    #                       hA_ground_truth,
    #                       IL_fp_detections,
    #                       IL_fn_detections,
    #                       IL_tp_detections,
    #                       type=AnnotationType.KEYPOINT,)


