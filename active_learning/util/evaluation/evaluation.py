"""
Take Detections from a model and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
import numpy as np
import typing

import PIL.Image
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
# import pytest

from active_learning.util.converter import herdnet_prediction_to_hasty, _create_keypoints_s, _create_boxes_s, \
    _create_fake_boxes
from active_learning.util.image_manipulation import crop_out_images_v3
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.types.status import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, ImageLabelCollection, AnnotatedImage, \
    HastyAnnotationV2
import seaborn as sns

def evaluate_predictions(
        # images_set: typing.List[Path],
        sample: fo.Sample,
        # images_dir: Path,
        predictions: typing.List[ImageLabelCollection] = None,
        type: AnnotationType = AnnotationType.KEYPOINT,
        sample_field_name="prediction") -> fo.Sample:
    """
    Evaluate the predictions
    :return:
    """
    image_name = sample.filename

    # for image_path in images_set:
    filtered_predictions = [p for p in predictions if p.image_name == image_name]
    if len(filtered_predictions) == 1:

        hA_image_pred = filtered_predictions[0]

        if type == AnnotationType.KEYPOINT:
            # keypoints_pred = _create_fake_boxes(hA_image=hA_image_pred) # TODO this was a bit of hack to visualise better in fiftyone
            keypoints_pred = _create_keypoints_s(hA_image=hA_image_pred)
        elif type == AnnotationType.BOUNDING_BOX:
            boxes_pred = _create_boxes_s(hA_image=hA_image_pred)
        elif type == AnnotationType.POLYGON:
            raise NotImplementedError("Polygons are not yet implemented")
            #polygons = _create_polygons_s(hA_image=hA_image)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")
        # FIXME, if this function is called multiple times, the same image will be added multiple times


        if type == AnnotationType.KEYPOINT:
            # sample[sample_field_name] = fo.Detections(detections=keypoints_pred)
            sample[sample_field_name] = fo.Keypoints(keypoints=keypoints_pred)
        elif type == AnnotationType.BOUNDING_BOX:
            sample[sample_field_name] = fo.Detections(detections=boxes_pred)
        elif type == AnnotationType.POLYGON:
            raise NotImplementedError("Polygons are not yet implemented")
            # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")

        if hasattr(hA_image_pred, "tags"):
            sample.tags = hA_image_pred.tags
        sample["hasty_image_id"] = hA_image_pred.image_id
        sample["hasty_image_name"] = hA_image_pred.image_name

        # logger.info(f"Added {image_name} to the dataset")

        sample.save()
    else:
        logger.error(f"There should be one single image left, but {len(filtered_predictions)} are left.")

    return sample


def evaluate_ground_truth(
        sample: fo.Sample,
        ground_truth_labels: typing.List[AnnotatedImage] = None,
        type: AnnotationType = AnnotationType.KEYPOINT, sample_field_name="ground_truth", ):
    """
    add ground truth to fiftyOne sample    :return:
    """
    # dataset = fo.load_dataset(dataset_name) # loading a bad idea because the single source of truth is the hasty annotations
    image_name = sample.filename

    hA_gt_sample = [i for i in ground_truth_labels if i.image_name == sample.filename]
    assert len(hA_gt_sample) == 1, f"There should be one single image left, but {len(hA_gt_sample)} are left."

    hA_image = hA_gt_sample[0]

    if type == AnnotationType.KEYPOINT:
        keypoints = _create_keypoints_s(hA_image=hA_image)
    elif type == AnnotationType.BOUNDING_BOX:
        boxes = _create_boxes_s(hA_image=hA_image)
    elif type == AnnotationType.POLYGON:
        raise NotImplementedError("Polygons are not yet implemented")
        #polygons = _create_polygons_s(hA_image=hA_image)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    sample.tags=hA_image.tags
    sample["hasty_image_id"] = hA_image.image_id
    sample["hasty_image_name"] = hA_image.image_name

    if type == AnnotationType.KEYPOINT:
        sample["ground_truth"] = fo.Keypoints(keypoints=keypoints)
    elif type == AnnotationType.BOUNDING_BOX:
        sample["ground_truth"] = fo.Detections(detections=boxes)
    elif type == AnnotationType.POLYGON:
        raise NotImplementedError("Polygons are not yet implemented")
        # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    # logger.info(f"Added {image_name} to the dataset")

    sample.save()

    return sample


def evaluate_in_fifty_one(dataset_name: str, images_set: typing.List[Path],
                          hA_ground_truth: HastyAnnotationV2,
                          IL_fp_detections: typing.List[ImageLabelCollection],
                          IL_fn_detections: typing.List[ImageLabelCollection],
                          IL_tp_detections: typing.List[ImageLabelCollection],
                          type: AnnotationType =AnnotationType.KEYPOINT):
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True
        # fo.list_datasets()

    dataset = fo.Dataset.from_images([str(i) for i in images_set])
    dataset.persistent = True

    for sample in dataset:
        # create dot annotations
        sample = evaluate_ground_truth(
            ground_truth_labels=hA_ground_truth.images,
            sample=sample,
            sample_field_name="ground_truth_points",
            # images_set=images_set,
            type=type,
        )

        sample = evaluate_predictions(
            predictions=IL_fp_detections,
            sample=sample,
            sample_field_name="false_positives",
            # images_set=images_set,
            type=type,
        )
        sample = evaluate_predictions(
            predictions=IL_fn_detections,
            sample=sample,
            sample_field_name="false_negatives",
            type=type,
        )
        sample = evaluate_predictions(
            predictions=IL_tp_detections,
            sample=sample,
            sample_field_name="true_positives",
            type=type,
        )

        dataset.add_sample(sample)


    # ## TODO fix the dataset fields
    # evaluation_results = dataset.evaluate_detections(
    #     pred_field="true_positives",
    #     gt_field="ground_truth_points",
    #     eval_key="point_eval",
    #     eval_type="keypoint",
    #     distance=10.0,  # Adjust based on your specific requirements
    #     classes=None  # Specify classes if applicable
    # )
    # precision = evaluation_results.metrics()["precision"]
    # recall = evaluation_results.metrics()["recall"]
    # f1_score = evaluation_results.metrics()["f1"]
    #
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"F1 Score: {f1_score:.2f}")
    #
    # evaluation_results.print_report()

    session = fo.launch_app(dataset)
    session.wait()


def submit_for_cvat_evaluation(dataset: fo.Dataset,
                               # images_set: typing.List[Path],
                          detections: typing.List[ImageLabelCollection],
                          type=AnnotationType.KEYPOINT):
    """
    @Deprecated
    :param dataset_name:
    :param images_set:
    :param detections:
    :param type:
    :return:
    """


    for sample in dataset:
        # create dot annotations
        sample = evaluate_predictions(
            predictions=detections,
            sample=sample,
            sample_field_name="detection",
            # images_set=images_set,
            type=type,
        )
        sample.save()

    return dataset



def submit_for_roboflow_evaluation(dataset_name: str, images_set: typing.List[Path],
                          detections: typing.List[ImageLabelCollection],
                          type="points"):
    raise NotImplementedError("Roboflow is not yet implemented")


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
        diffs = fp_counts - fn_counts

        return diffs.tolist()


def plot_confidence_density(df: pd.DataFrame,
                            title="Confidence Score Density Distribution",
                            xlabel="Confidence Score",
                            ylabel="Density",
                            figsize=(12, 6),
                            color='blue',
                            fill=True,
                            show_stats=True,
                            show_histogram=True,
                            kde_bandwidth=None):
    """
    Create a density plot of confidence scores.

    :param df: DataFrame with a 'scores' column containing confidence scores
    :param title: Title for the plot
    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :param figsize: Figure size as tuple (width, height)
    :param color: Color for the density plot
    :param fill: Whether to fill the area under the curve
    :param show_stats: Whether to show mean, median lines
    :param show_histogram: Whether to show histogram behind density
    :param kde_bandwidth: Bandwidth for KDE (None for automatic)
    :return: matplotlib figure and axis objects
    """

    # Check if 'scores' column exists
    if 'scores' not in df.columns:
        raise ValueError("DataFrame must contain a 'scores' column")

    # Extract scores and remove NaN values
    scores = df['scores'].dropna()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram if requested (in background)
    if show_histogram:
        ax.hist(scores, bins=30, density=True, alpha=0.3, color='gray',
                edgecolor='black', label='Histogram')

    # Create density plot using seaborn
    if fill:
        sns.kdeplot(data=scores, ax=ax, color=color, fill=True, alpha=0.6,
                    label='Density', bw_adjust=kde_bandwidth if kde_bandwidth else 1.0)
    else:
        sns.kdeplot(data=scores, ax=ax, color=color, linewidth=2,
                    label='Density', bw_adjust=kde_bandwidth if kde_bandwidth else 1.0)

    # Add statistics lines if requested
    if show_stats:
        mean_val = scores.mean()
        median_val = scores.median()
        mode_val = scores.mode()[0] if len(scores.mode()) > 0 else None

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.3f}')
        if mode_val is not None:
            ax.axvline(mode_val, color='orange', linestyle='--', linewidth=2,
                       label=f'Mode: {mode_val:.3f}')

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set x-axis limits to [0, 1] for confidence scores
    ax.set_xlim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='best')

    # Add text box with statistics
    stats_text = f'Count: {len(scores)}\n'
    stats_text += f'Mean: {scores.mean():.3f}\n'
    stats_text += f'Std: {scores.std():.3f}\n'
    stats_text += f'Min: {scores.min():.3f}\n'
    stats_text += f'Max: {scores.max():.3f}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig, ax








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