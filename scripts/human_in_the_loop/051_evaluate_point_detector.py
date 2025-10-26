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
    plot_error_curve, plot_single_metric_curve, plot_comprehensive_curves, plot_species_detection_analysis
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.types.status import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def evaluate_point_detections(base_path, df_detections, herdnet_annotation_name,
                              images_path, suffix, radius, box_size, CONFIDENCE_THRESHOLD):
    visualisations_path = base_path / "visualisations"
    visualisations_path.mkdir(exist_ok=True, parents=True)
    IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    image_list_all = [i.image_name for i in IL_detections]

    df_ground_truth = pd.read_csv(base_path / herdnet_annotation_name)

    images = list(images_path.glob(f"*.{suffix}"))
    if len(images) == 0:
        raise FileNotFoundError("No images found in: " + images_path)

    df_detections = df_detections[df_detections.scores > CONFIDENCE_THRESHOLD]

    df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=radius,
        image_list=image_list_all
    )

    fig, ax = plot_confidence_density(df_false_positives,
                                      title="Confidence Score Density Distribution of False Positives",
                                      save_path=visualisations_path / "false_positives_confidence_density.png")
    plt.show()

    fig, ax = plot_confidence_density(df_true_positives,
                                      title="Confidence Score Density Distribution of True Positives",
                                      save_path=visualisations_path / "true_positives_confidence_density.png")
    plt.show()

    # raise ValueError("Stop here to check the density plot of false positives")

    df_concat = pd.concat([df_false_positives, df_true_positives, df_false_negatives])

    IL_all_detections = herdnet_prediction_to_hasty(df_concat, images_path)
    IL_fp_detections = herdnet_prediction_to_hasty(df_false_positives, images_path)
    IL_tp_detections = herdnet_prediction_to_hasty(df_true_positives, images_path)
    IL_fn_detections = herdnet_prediction_to_hasty(df_false_negatives, images_path)

    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, "
                f"False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")

    # TODO draw a curve: x: confidence, y: precision, recall, f1, MAE, MSE
    ev = Evaluator(df_detections=df_detections, df_ground_truth=df_ground_truth, radius=radius)

    df_recall_curve = ev.get_precition_recall_curve(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                            0.95,
                                                            0.96, 0.97, 0.98, 0.99,
                                                            0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998,
                                                            0.999,
                                                            0.9999, 1.0])

    # plot_species_detection_analysis(df_recall_curve, title="Performance Analysis", save_path=visualisations_path / "species_performance_analysis.png",)
    # Plot comprehensive view
    plot_error_curve(df_recall_curve, title="Performance Analysis",
                     save_path=visualisations_path / "performance_analysis.png", )

    # TODO plot that QQ Plot here to see if there is a bias depending on animals in the images

    # Plot single metric
    plot_single_metric_curve(df_recall_curve, y_label='mean_error', title="Mean Error Analysis",
                             save_path=visualisations_path / "mean_error_curve.png", )
    plot_single_metric_curve(df_recall_curve, x_label="recall", y_label='precision', title="Precision Recall Curve",
                             save_path=visualisations_path / "precision_recall_curve.png", )

    # Plot all metrics separately
    plot_comprehensive_curves(df_recall_curve,
                              save_path=visualisations_path / "comprehensive_performance_analysis.png", )

    logger.info(f"Precision: {ev.precision_all:.2f}, Recall: {ev.recall_all:.2f}, F1: {ev.f1_all:.2f}")

    for i in images:
        if len(df_concat[df_concat.images == i.name]) > 0:
            logger.info(f"Drawing thumbnails for {i.name} with {len(df_concat[df_concat.images == i.name])} detections")
        else:
            logger.warning(f"Images not {i.name} found in detections/false negatives, skipping drawing thumbnails")

        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]
        gdf_gt = gdf_ground_truth[gdf_ground_truth.images == i.name]

        logger.info(f"Drawing high confidence false positives for {i.name}")
        draw_thumbnail(df_fp[df_fp.scores > 0.9], i,
                       suffix="fp_hc",
                       images_path=visualisations_path, box_size=box_size,
                       df_gt=gdf_gt)

        df_fp_hc = df_fp[df_fp.scores > 0.9]

        logger.info(f"Drawing false positives to {visualisations_path} done.")
        # #
        # # high medium confidence false positives
        # draw_thumbnail(df_fp[(df_fp.scores > 0.8) & (df_fp.scores < 0.9)], i, suffix="fp_hmc",
        #                images_path=visualisations_path, box_size=box_size,
        #                df_gt=gdf_gt)
        #
        # # low confidence false positives
        # draw_thumbnail(df_fp[(df_fp.scores > 0.4) & (df_fp.scores < 0.7)], i, suffix="fp_lc",
        #                images_path=visualisations_path, box_size=box_size,
        #                df_gt=gdf_gt)
        # #
        # # low confidence false positives
        # # draw_thumbnail(df_fp[df_fp.scores <= 0.4], i, suffix="fp_lc", images_path=visualisations_path, box_size=box_size)
        #
        # # False negatives
        # if len(df_fn) > 0:
        #     draw_thumbnail(df_fn, i, suffix="fn",
        #                    images_path=visualisations_path, box_size=box_size,
        #                    DETECTECTED_COLOR="red",
        #                    GT_COLOR="blue",
        #                    df_gt=gdf_gt)

        # all true positives
        # draw_thumbnail(df_tp, i, suffix="tp", images_path=visualisations_path, box_size=box_size)
        #     logger.info(f"Drawing example negatives to {visualisations_path}")

    # images_set = [images_path / i.image_name for i in hA_ground_truth.images]

    # evaluate_in_fifty_one(dataset_name,
    #                       images_set,
    #                       hA_ground_truth,
    #                       IL_fp_detections,
    #                       IL_fn_detections,
    #                       IL_tp_detections,
    #                       type=AnnotationType.KEYPOINT,)


if __name__ == "__main__":
    ds= "Genovesa"
    if ds == "Eikelboom2019":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/eikelboom2019/eikelboom_512_overlap_0_ebFalse/eikelboom_test/test/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-05/10-14-15/detections.csv') # dla102

        rename_species = {
            "Buffalo": "Elephant",
            "Alcelaphinae": "Giraffe",
            "Kob": "Zebra"
        }
        df_detections['species'] = df_detections['species'].replace(rename_species)
        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"

    elif ds == "Genovesa":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Genovesa_detection/val/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/13-35-05/detections.csv') # dla102


        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
        
    elif ds == "Genovesa_crop":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Genovesa_detection/val/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/13-49-19/detections.csv') # dla102


        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "crops_512_numNone_overlap0"
        
    elif ds == "delplanque2023":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/delplanque2023/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/14-02-23/detections.csv')

        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
    
    else:
        raise ValueError("Unknown dataset: " + ds)
        
    suffix = "JPG"

    # hA_ground_truth_path = base_path / hasty_annotation_name
    # hA_ground_truth = HastyAnnotationV2.from_file(hA_ground_truth_path)

    # ## On cropped images
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47/detections.csv')
    # hasty_annotation_name = 'hasty_format_crops_512_0.json'
    # herdnet_annotation_name = 'herdnet_format_512_0_crops.csv'
    # images_path = base_path / "crops_512_numNone_overlap0"
    # suffix = "jpg"

    box_size = 350
    radius = 150
    CONFIDENCE_THRESHOLD = 0.0

    evaluate_point_detections(base_path, 
                              df_detections, 
                              herdnet_annotation_name, 
                              images_path,
                              suffix, radius, box_size, CONFIDENCE_THRESHOLD)


