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
from active_learning.util.evaluation.evaluation import evaluate_in_fifty_one
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Evaluator():

    def __init__(self, df_detections, df_ground_truth, radius=150):
        self.df_detections = df_detections
        self.df_ground_truth = df_ground_truth
        self.radius = radius

        self.df_false_positives, self.df_true_positives, self.df_false_negatives = analyse_point_detections_greedy(
            df_detections=self.df_detections,
            df_ground_truth=self.df_ground_truth,
            radius=self.radius
        )
        self.precision_all = self.precision(self.df_true_positives, self.df_false_positives)
        self.recall_all = self.recall(self.df_true_positives, self.df_false_negatives)
        self.f1_all = self.f1(self.precision_all, self.recall_all)

    def precision(self, df_true_positives, df_false_positives):
            if len(df_true_positives) + len(df_false_positives) == 0:
                return 0.0
            return len(df_true_positives) / (len(df_true_positives) + len(df_false_positives))

    def recall(self, df_true_positives, df_false_negatives):
        if len(df_true_positives) + len(df_false_negatives) == 0:
            return 0.0
        return len(df_true_positives) / (len(df_true_positives) + len(df_false_negatives))

    def f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


    def get_precision_recall_f1(self, df_detections):

        df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
            df_detections=df_detections,
            df_ground_truth=self.df_ground_truth,
            radius=self.radius
        )

        precision = self.precision(df_true_positives, df_false_positives)
        recall = self.recall(df_true_positives, df_false_negatives)
        f1 = self.f1(precision, recall)

        return precision, recall, f1


    def get_precition_recall_curve(self, values: typing.List[float] = None, range_start=0, range_end=1.0, step=0.05):
        results = []
        all_errors = []

        for confidence_threshold in values:
            df_detections = self.df_detections[self.df_detections.scores >= confidence_threshold]

            df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
                df_detections=df_detections,
                df_ground_truth=self.df_ground_truth,
                radius=self.radius
            )

            precision = self.precision(df_true_positives, df_false_positives)
            recall = self.recall(df_true_positives, df_false_negatives)
            f1 = self.f1(precision, recall)

            errors = self.calculate_error_metrics(df_true_positives, df_false_positives, df_false_negatives)

            all_errors.append(errors)
            d=[confidence_threshold, precision, recall, f1]
            results.append(d)

        df_results = pd.DataFrame(results, columns=["confidence_threshold", "precision", "recall", "f1"])
        df_errors = pd.DataFrame(all_errors)


        return pd.concat([df_results, df_errors], axis=1)

    def calculate_error_metrics(self, df_true_positives: pd.DataFrame,
                                df_false_positives: pd.DataFrame,
                                df_false_negatives: pd.DataFrame):

        # Get the counting errors (your existing function)
        diffs = self.get_counting_errors(df_true_positives, df_false_positives, df_false_negatives)
        errors = np.array(diffs)

        # Calculate all error metrics with numpy
        mean_error = np.mean(errors)  # Mean Error (ME)
        mean_absolute_error = np.mean(np.abs(errors))  # Mean Absolute Error (MAE)
        mean_squared_error = np.mean(errors ** 2)  # Mean Squared Error (MSE)
        root_mean_squared_error = np.sqrt(mean_squared_error)  # RMSE (bonus)

        return {
            'mean_error': mean_error,
            'mean_absolute_error': mean_absolute_error,
            'mean_squared_error': mean_squared_error,
            'root_mean_squared_error': root_mean_squared_error,
            'total_images': len(errors)
        }

    def get_counting_errors(self, df_true_positives: pd.DataFrame,
                            df_false_positives: pd.DataFrame, df_false_negatives: pd.DataFrame):

        image_list = self.df_ground_truth['images'].unique()

        tp_counts = df_true_positives['images'].value_counts().reindex(image_list, fill_value=0).values
        fp_counts = df_false_positives['images'].value_counts().reindex(image_list, fill_value=0).values
        fn_counts = df_false_negatives['images'].value_counts().reindex(image_list, fill_value=0).values
        gt_counts = self.df_ground_truth['images'].value_counts().reindex(image_list, fill_value=0).values

        total_counts = tp_counts + fp_counts + fn_counts
        mismatched = total_counts != gt_counts

        if np.any(mismatched):
            # Only loop for warnings (much fewer iterations)
            mismatched_images = image_list[mismatched]
            for i, image in enumerate(mismatched_images):
                idx = np.where(image_list == image)[0][0]
                # logger.warning(f"Counting error in image {image}: "
                #                f"TP: {tp_counts[idx]}, FP: {fp_counts[idx]}, FN: {fn_counts[idx]}, "
                #                f"GT: {gt_counts[idx]}")

            # Vectorized calculation of diffs
        diffs = tp_counts + fp_counts - fn_counts

        return diffs.tolist()





def plot_error_curve(df_recall_curve,
                     x_label="confidence_threshold",
                     y_label="mean_error",
                     title="Error Curve",
                     save_path=None):
    """
    Plot error metrics vs confidence threshold with multiple y-axes for different scales.

    Args:
        df_recall_curve: DataFrame with columns including confidence_threshold and error metrics
        x_label: Column name for x-axis (default: "confidence_threshold")
        y_label: Primary metric to plot (default: "mean_error")
        title: Plot title
        save_path: Optional path to save the plot
    """

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Top plot: Precision, Recall, F1
    ax1.plot(df_recall_curve[x_label], df_recall_curve['precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(df_recall_curve[x_label], df_recall_curve['recall'], 'r-', label='Recall', linewidth=2)
    ax1.plot(df_recall_curve[x_label], df_recall_curve['f1'], 'g-', label='F1 Score', linewidth=2)

    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1 Score vs Confidence Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)


    # Primary y-axis: Mean Error and MAE (similar scales)
    line1 = ax2.plot(df_recall_curve[x_label], df_recall_curve['mean_error'], 'purple',
                     label='Mean Error', linewidth=2, marker='o', markersize=4)
    line2 = ax2.plot(df_recall_curve[x_label], df_recall_curve['mean_absolute_error'], 'orange',
                     label='Mean Absolute Error', linewidth=2, marker='s', markersize=4)


    # Formatting
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Mean Error & MAE', color='black')
    ax2.set_title('Error Metrics vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')

    # Add zero line for mean error
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Zero Error')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_single_metric_curve(df_recall_curve,
                             x_label="confidence_threshold",
                             y_label="mean_error",
                             title="Error Curve",
                             save_path=None):
    """
    Plot a single metric vs confidence threshold.

    Args:
        df_recall_curve: DataFrame with the data
        x_label: Column name for x-axis
        y_label: Column name for y-axis
        title: Plot title
        save_path: Optional path to save the plot
    """

    plt.figure(figsize=(10, 6))

    plt.plot(df_recall_curve[x_label], df_recall_curve[y_label],
             'b-', linewidth=2, marker='o', markersize=6)

    plt.xlabel(x_label.replace('_', ' ').title())
    plt.ylabel(y_label.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add zero line if plotting error metrics
    if 'error' in y_label.lower():
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comprehensive_curves(df_recall_curve, save_path=None):
    """
    Create a comprehensive 2x2 plot showing all metrics.
    """

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Performance Analysis vs Confidence Threshold', fontsize=16, fontweight='bold')

    x = df_recall_curve['confidence_threshold']

    # Plot 1: Precision, Recall, F1
    ax1.plot(x, df_recall_curve['precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(x, df_recall_curve['recall'], 'r-', label='Recall', linewidth=2)
    ax1.plot(x, df_recall_curve['f1'], 'g-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Mean Error
    ax2.plot(x, df_recall_curve['mean_error'], 'purple', linewidth=2, marker='o')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Mean Error')
    ax2.set_title('Mean Error (Bias)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2025_07_10"
    # lcrop_size = 640
    num = 56
    type = "points"


    base_path = Path(f'/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_07_10_final_point_detection_edge_blackout')

    ## On full size original images
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/inference_2025-07-15_fullsize/08-19-14/detections.csv')
    hasty_annotation_name = 'hasty_format_full_size.json'
    herdnet_annotation_name = 'herdnet_format.csv'
    images_path = base_path / "Default"
    suffix = "JPG"

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
    df_recall_curve = ev.get_precition_recall_curve(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                            0.95,
                                                            0.96, 0.97, 0.98, 0.99,
                                                            0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
                                                            0.9999, 1.0])

    # Plot comprehensive view
    plot_error_curve(df_recall_curve, title="Performance Analysis")

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


