"""
Take Detections from a model, compare them with ground truth and display them in a FiftyOne Dataset


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
from active_learning.util.evaluation.evaluation import evaluate_in_fifty_one, Evaluator, plot_confidence_density, \
    plot_error_curve
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.types.status import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats




if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2025_07_10"
    # lcrop_size = 640
    num = 56
    type = "points"

    # Path of the base directory where the images and annotations are stored which we want to correct
    base_path = Path(f'/Users/christian/data/training_data/2025_07_10_refined/Floreana_detection_corrected/train')

    ## On full size original images
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/label_correction/floreana_train_inference/detections.csv')
    hasty_annotation_name = 'hasty_format_full_size.json'
    herdnet_annotation_name = 'herdnet_format_points.csv'
    images_path = base_path / "Default"
    suffix = "JPG"

    hA_ground_truth_path = base_path / hasty_annotation_name
    hA_ground_truth = hA_from_file(hA_ground_truth_path)

    # ## On cropped images
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47/detections.csv')
    # hasty_annotation_name = 'hasty_format_crops_512_0.json'
    # herdnet_annotation_name = 'herdnet_format_512_0_crops.csv'
    # images_path = base_path / "crops_512_numNone_overlap0"
    # suffix = "jpg"

    box_size = 350
    radius = 150

    visualisations_path = base_path / "visualisations"
    visualisations_path.mkdir(exist_ok=True, parents=True)
    IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)


    df_ground_truth = pd.read_csv(base_path / herdnet_annotation_name)

    images = list(images_path.glob(f"*.{suffix}"))
    if len(images) == 0:
        raise FileNotFoundError("No images found in: " + images_path)

    df_detections = df_detections[df_detections.scores > 0.30]

    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=radius
    )

    fig, ax = plot_confidence_density(df_false_positives, title="Confidence Score Density Distribution of False Positves")
    plt.show()

    fig, ax = plot_confidence_density(df_true_positives, title="Confidence Score Density Distribution of True Positves")
    plt.show()

    # raise ValueError("Stop here to check the density plot of false positives")

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
    df_recall_curve = ev.get_precition_recall_curve(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                            #0.95,
                                                            #0.96, 0.97, 0.98, 0.99,
                                                            #0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
                                                            0.9999, 1.0])

    # Plot comprehensive view
    plot_error_curve(df_recall_curve, title="Performance Analysis")

    # TODO plot that QQ Plot here to see if there is a bias depending on animals in the images

    # Plot single metric
    plot_single_metric_curve(df_recall_curve, y_label='mean_error', title="Mean Error Analysis")

    # Plot all metrics separately
    plot_comprehensive_curves(df_recall_curve)

    logger.info(f"Precision: {ev.precision_all:.2f}, Recall: {ev.recall_all:.2f}, F1: {ev.f1_all:.2f}")

    for i in images:
        if len(df_concat[df_concat.images == i.name]) > 0:
            logger.info(f"Drawing thumbnails for {i.name} with {len(df_concat[df_concat.images == i.name])} detections")
        else:
            logger.warning(f"Images not {i.name} found in detections/false negatives, skipping drawing thumbnails")

        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]

        draw_thumbnail(df_fp[df_fp.scores > 0.9], i, suffix="fp_hc", images_path=visualisations_path, box_size=box_size)
        draw_thumbnail(df_fp[(df_fp.scores > 0.8) & (df_fp.scores < 0.9)], i, suffix="fp_hmc",
                       images_path=visualisations_path, box_size=box_size)
        # draw_thumbnail(df_fp[df_fp.scores <= 0.9], i, suffix="fp_lc", images_path=visualisations_path, box_size=box_size)
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


