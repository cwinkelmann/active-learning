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
    """
    Evaluator Abstraction to calculate precision, recall and f1 score
    """

    def __init__(self, df_detections, df_ground_truth, radius=150):
        """

        :param df_detections:
        :param df_ground_truth:
        :param radius: distance from point to another point
        """
        self.df_detections = df_detections
        self.df_ground_truth = df_ground_truth
        self.gdf_ground_truth = None
        self.radius = radius
        self.image_list = self.df_ground_truth['images'].unique()

        self.df_false_positives, self.df_true_positives, self.df_false_negatives, self.gdf_ground_truth = analyse_point_detections_greedy(
            df_detections=self.df_detections,
            df_ground_truth=self.df_ground_truth,
            radius=self.radius,
            image_list = self.image_list
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
            radius=self.radius,
            image_list = self.image_list
        )

        precision = self.precision(df_true_positives, df_false_positives)
        recall = self.recall(df_true_positives, df_false_negatives)
        f1 = self.f1(precision, recall)

        return precision, recall, f1


    def get_precition_recall_curve(self, values: typing.List[float] = None, range_start=0, range_end=1.0, step=0.05):
        """
        Get precision recall curve for a range of confidence thresholds
        :param values:
        :param range_start:
        :param range_end:
        :param step:
        :return:
        """
        results = []
        all_errors = []

        for confidence_threshold in values:
            df_detections = self.df_detections[self.df_detections.scores >= confidence_threshold]

            df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth = analyse_point_detections_greedy(
                df_detections=df_detections,
                df_ground_truth=self.df_ground_truth,
                radius=self.radius,
                image_list = self.image_list
            )


            recall = self.recall(df_true_positives, df_false_negatives)
            if recall == 0:
                precision = 1.0
                f1 = 0.0
            else:
                precision = self.precision(df_true_positives, df_false_positives)
                f1 = self.f1(precision, recall)

            num_fp = len(df_false_positives)
            num_tp = len(df_true_positives)
            num_fn = len(df_false_negatives)

            metrics = {
                'confidence_threshold': confidence_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_false_positives': num_fp,
                'num_true_positives': num_tp,
                'num_false_negatives': num_fn,
            }

            errors = self.calculate_error_metrics(df_true_positives, df_false_positives, df_false_negatives)

            all_errors.append(errors)
            # d=[confidence_threshold, precision, recall, f1]
            results.append(metrics)

        df_results = pd.DataFrame(results)
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
                            title_flag=False,
                            xlabel="Confidence Score",
                            ylabel="Density",
                            figsize=(12, 6),
                            color='blue',
                            fill=True,
                            show_stats=False,
                            show_histogram=False,
                            kde_bandwidth=None,
                            save_path=None):
    """
    Create a density plot of confidence scores.
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
    if title_flag:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Set x-axis limits to [0, 1] for confidence scores
    ax.set_xlim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='best')

    # # Add text box with statistics
    # stats_text = f'Count: {len(scores)}\n'
    # stats_text += f'Mean: {scores.mean():.3f}\n'
    # stats_text += f'Std: {scores.std():.3f}\n'
    #
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    #          )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_fp_tp_confidence_density(df_fp: pd.DataFrame,
                                  df_tp: pd.DataFrame,
                                  title="Confidence Score Density Distribution: True Positives vs False Positives",
                                  title_flag=False,
                                  xlabel="Confidence Score",
                                  ylabel="Density",
                                  figsize=(12, 6),
                                  fp_color='red',
                                  tp_color='blue',
                                  fill=True,
                                  show_stats=False,
                                  show_histogram=False,
                                  kde_bandwidth=None,
                                  save_path=None):
    """
    Create overlaid density plots of confidence scores for false positives and true positives.

    Args:
        df_fp: DataFrame containing false positive detections with 'scores' column
        df_tp: DataFrame containing true positive detections with 'scores' column
        title: Plot title
        title_flag: Whether to show the title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        fp_color: Color for false positives
        tp_color: Color for true positives
        fill: Whether to fill under the density curves
        show_stats: Whether to show mean/median lines
        show_histogram: Whether to show histograms in background
        kde_bandwidth: Bandwidth adjustment for KDE
        save_path: Path to save the figure
    """

    # Check if 'scores' column exists in both dataframes
    if 'scores' not in df_fp.columns or 'scores' not in df_tp.columns:
        raise ValueError("Both DataFrames must contain a 'scores' column")

    # Extract scores and remove NaN values
    fp_scores = df_fp['scores'].dropna()
    tp_scores = df_tp['scores'].dropna()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms if requested (in background)
    if show_histogram:
        ax.hist(fp_scores, bins=30, density=True, alpha=0.2, color=fp_color,
                edgecolor='darkred', label='FP Histogram')
        ax.hist(tp_scores, bins=30, density=True, alpha=0.2, color=tp_color,
                edgecolor='darkblue', label='TP Histogram')

    # Create density plots
    bw_adjust = kde_bandwidth if kde_bandwidth else 1.0

    if fill:
        sns.kdeplot(data=fp_scores, ax=ax, color=fp_color, fill=True, alpha=0.4,
                    label=f'False Positives (n={len(fp_scores)})', bw_adjust=bw_adjust)
        sns.kdeplot(data=tp_scores, ax=ax, color=tp_color, fill=True, alpha=0.4,
                    label=f'True Positives (n={len(tp_scores)})', bw_adjust=bw_adjust)
    else:
        sns.kdeplot(data=fp_scores, ax=ax, color=fp_color, linewidth=2,
                    label=f'False Positives (n={len(fp_scores)})', bw_adjust=bw_adjust)
        sns.kdeplot(data=tp_scores, ax=ax, color=tp_color, linewidth=2,
                    label=f'True Positives (n={len(tp_scores)})', bw_adjust=bw_adjust)

    # Add statistics lines if requested
    if show_stats:
        fp_mean = fp_scores.mean()
        tp_mean = tp_scores.mean()
        fp_median = fp_scores.median()
        tp_median = tp_scores.median()

        ax.axvline(fp_mean, color=fp_color, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'FP Mean: {fp_mean:.3f}')
        ax.axvline(tp_mean, color=tp_color, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'TP Mean: {tp_mean:.3f}')
        ax.axvline(fp_median, color=fp_color, linestyle=':', linewidth=2, alpha=0.7,
                   label=f'FP Median: {fp_median:.3f}')
        ax.axvline(tp_median, color=tp_color, linestyle=':', linewidth=2, alpha=0.7,
                   label=f'TP Median: {tp_median:.3f}')

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title_flag:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Set x-axis limits to [0, 1] for confidence scores
    ax.set_xlim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='best')

    # Optional: Add text box with comparative statistics
    if show_stats:
        stats_text = f'False Positives:\n'
        stats_text += f'  Mean: {fp_scores.mean():.3f}\n'
        stats_text += f'  Median: {fp_scores.median():.3f}\n'
        stats_text += f'  Std: {fp_scores.std():.3f}\n\n'
        stats_text += f'True Positives:\n'
        stats_text += f'  Mean: {tp_scores.mean():.3f}\n'
        stats_text += f'  Median: {tp_scores.median():.3f}\n'
        stats_text += f'  Std: {tp_scores.std():.3f}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax


def plot_fp_tp_confidence_histogram(df_fp: pd.DataFrame,
                                    df_tp: pd.DataFrame,
                                    title="Confidence Score Distribution: True Positives vs False Positives",
                                    title_flag=False,
                                    xlabel="Confidence Score",
                                    ylabel="Count",
                                    figsize=(12, 6),
                                    fp_color='red',
                                    tp_color='blue',
                                    bins=30,
                                    alpha=0.6,
                                    show_stats=False,
                                    density=False,
                                    stacked=False,
                                    save_path=None):
    """
    Create overlaid histograms of confidence scores for false positives and true positives.

    Args:
        df_fp: DataFrame containing false positive detections with 'scores' column
        df_tp: DataFrame containing true positive detections with 'scores' column
        title: Plot title
        title_flag: Whether to show the title
        xlabel: X-axis label
        ylabel: Y-axis label (automatically adjusted if density=True)
        figsize: Figure size tuple
        fp_color: Color for false positives
        tp_color: Color for true positives
        bins: Number of bins or bin edges
        alpha: Transparency of histogram bars
        show_stats: Whether to show mean/median lines
        density: If True, normalize to show probability density
        stacked: If True, stack histograms instead of overlaying
        save_path: Path to save the figure
    """

    # Check if 'scores' column exists in both dataframes
    if 'scores' not in df_fp.columns or 'scores' not in df_tp.columns:
        raise ValueError("Both DataFrames must contain a 'scores' column")

    # Extract scores and remove NaN values
    fp_scores = df_fp['scores'].dropna()
    tp_scores = df_tp['scores'].dropna()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    if stacked:
        ax.hist([tp_scores, fp_scores], bins=bins,
                color=[tp_color, fp_color],
                label=[f'True Positives (n={len(tp_scores)})',
                       f'False Positives (n={len(fp_scores)})'],
                alpha=alpha, edgecolor='black', density=density, stacked=True)
    else:
        ax.hist(fp_scores, bins=bins, color=fp_color, alpha=alpha,
                label=f'FP (n={len(fp_scores)})',
                edgecolor='darkred', density=density)
        ax.hist(tp_scores, bins=bins, color=tp_color, alpha=alpha,
                label=f'TP (n={len(tp_scores)})',
                edgecolor='darkblue', density=density)

    # Add statistics lines if requested
    # if show_stats:
    # fp_mean = fp_scores.mean()
    # tp_mean = tp_scores.mean()
    fp_median = fp_scores.median()
    tp_median = tp_scores.median()

    # Get y-limit for vertical lines
    y_max = ax.get_ylim()[1]

    # ax.axvline(fp_mean, color='darkred', linestyle='--', linewidth=2, alpha=0.8,
    #            label=f'FP Mean: {fp_mean:.3f}')
    # ax.axvline(tp_mean, color='darkblue', linestyle='--', linewidth=2, alpha=0.8,
    #            label=f'TP Mean: {tp_mean:.3f}')
    ax.axvline(fp_median, color='darkred', linestyle=':', linewidth=2, alpha=0.8,
               label=f'Confidence FP Median: {fp_median:.3f}')
    ax.axvline(tp_median, color='darkblue', linestyle=':', linewidth=2, alpha=0.8,
               label=f'Confidence TP Median: {tp_median:.3f}')

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    if density:
        ax.set_ylabel('Density' if ylabel == "Count" else ylabel, fontsize=12)
    else:
        ax.set_ylabel(ylabel, fontsize=12)

    if title_flag:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Set x-axis limits to [0, 1] for confidence scores
    ax.set_xlim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    ax.legend(loc='best')

    # Optional: Add text box with comparative statistics
    if show_stats:
        stats_text = f'False Positives:\n'
        stats_text += f'  Count: {len(fp_scores)}\n'
        stats_text += f'  Mean: {fp_scores.mean():.3f}\n'
        stats_text += f'  Median: {fp_scores.median():.3f}\n'
        stats_text += f'  Std: {fp_scores.std():.3f}\n\n'
        stats_text += f'True Positives:\n'
        stats_text += f'  Count: {len(tp_scores)}\n'
        stats_text += f'  Mean: {tp_scores.mean():.3f}\n'
        stats_text += f'  Median: {tp_scores.median():.3f}\n'
        stats_text += f'  Std: {tp_scores.std():.3f}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax

def plot_species_detection_analysis(df_recall_curve: dict,
                                    save_path=None,
                                    title="Marine Iguana Detection Analysis - Galápagos Islands",
                                    figsize=(16, 12)):
    """
    Create a comprehensive 2x2 plot showing detection metrics for multiple species.

    Args:
        species_data: Dictionary where keys are species names and values are dictionaries containing:
            - 'df_recall_curve': DataFrame with confidence_threshold, precision, recall
            - 'df_true_positives': DataFrame with true positive detections and scores
            - 'df_false_positives': DataFrame with false positive detections and scores
        save_path: Optional path to save the plot
        title: Overall plot title
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Set up color palette for species
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_data)))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Precision-Recall Curve (top left)
    for idx, (species_name, data) in enumerate(species_data.items()):
        df = data['df_recall_curve']
        ax1.plot(df['recall'], df['precision'],
                 label=species_name,
                 linewidth=2,
                 color=colors[idx])

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Add mAP annotation if needed
    # You can calculate mAP here if you have the data

    # Plot 2: Confidence-Recall Curve (top right)
    for idx, (species_name, data) in enumerate(species_data.items()):
        df = data['df_recall_curve']
        ax2.plot(df['confidence_threshold'], df['recall'],
                 label=species_name,
                 linewidth=2,
                 color=colors[idx])

    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Confidence-Recall Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Plot 3: Confidence-Precision Curve (bottom left)
    for idx, (species_name, data) in enumerate(species_data.items()):
        df = data['df_recall_curve']
        ax3.plot(df['confidence_threshold'], df['precision'],
                 label=species_name,
                 linewidth=2,
                 color=colors[idx])

    ax3.set_xlabel('Confidence Threshold', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Confidence-Precision Curve', fontsize=13, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Plot 4: Detection Score Distribution (bottom right)
    # Create histogram bins
    bins = np.linspace(0, 1, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Stack true positives and false positives for each species
    bottom_tp = np.zeros(len(bin_centers))
    bottom_fp = np.zeros(len(bin_centers))

    for idx, (species_name, data) in enumerate(species_data.items()):
        df_tp = data['df_true_positives']
        df_fp = data['df_false_positives']

        # Calculate histogram for true positives
        tp_hist, _ = np.histogram(df_tp['scores'], bins=bins, density=True)
        ax4.bar(bin_centers, tp_hist, width=bins[1] - bins[0],
                bottom=bottom_tp,
                color=colors[idx],
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5,
                label=f'{species_name} - TP')
        bottom_tp += tp_hist

        # Calculate histogram for false positives
        fp_hist, _ = np.histogram(df_fp['scores'], bins=bins, density=True)
        ax4.bar(bin_centers, fp_hist, width=bins[1] - bins[0],
                bottom=bottom_fp,
                color=colors[idx],
                alpha=0.4,
                edgecolor='black',
                linewidth=0.5,
                label=f'{species_name} - FP',
                hatch='//')
        bottom_fp += fp_hist

    ax4.set_xlabel('Confidence Score', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Detection Score Distribution', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, (ax1, ax2, ax3, ax4)


def plot_precision_recall_f1_curve(df_recall_curve,
                                   x_label="confidence_threshold",
                                   title_flag=False,
                                   title="Precision, Recall, and F1 Score vs Confidence Threshold",
                                   save_path=None):
    """
    Plot Precision, Recall, and F1 Score vs confidence threshold.

    Args:
        df_recall_curve: DataFrame with columns including confidence_threshold and metrics
        x_label: Column name for x-axis (default: "confidence_threshold")
        title_flag: Whether to show the title
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Find optimal threshold based on F1 score
    optimal_idx = df_recall_curve['f1'].idxmax()
    optimal_threshold = df_recall_curve.loc[optimal_idx, x_label]
    optimal_f1_score = df_recall_curve.loc[optimal_idx, 'f1']

    plt.figure(figsize=(10, 6))

    # Plot metrics
    plt.plot(df_recall_curve[x_label], df_recall_curve['precision'], 'b-', label='Precision', linewidth=2)
    plt.plot(df_recall_curve[x_label], df_recall_curve['recall'], 'r-', label='Recall', linewidth=2)
    plt.plot(df_recall_curve[x_label], df_recall_curve['f1'], 'g-', label='F1 Score', linewidth=2)

    # Add vertical line at optimal threshold
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {optimal_threshold:.3f}\n(F1 Score: {optimal_f1_score:.3f})')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    if title_flag:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_error_metrics_curve(df_recall_curve,
                             x_label="confidence_threshold",
                             title_flag=False,
                             title="Error Metrics vs Confidence Threshold",
                             save_path=None):
    """
    Plot error metrics vs confidence threshold.

    Args:
        df_recall_curve: DataFrame with columns including confidence_threshold and error metrics
        x_label: Column name for x-axis (default: "confidence_threshold")
        title_flag: Whether to show the title
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Find confidence threshold where mean_error is closest to zero
    optimal_idx = df_recall_curve['mean_error'].abs().idxmin()
    optimal_threshold = df_recall_curve.loc[optimal_idx, x_label]
    optimal_error = df_recall_curve.loc[optimal_idx, 'mean_error']

    plt.figure(figsize=(10, 6))

    # Plot error metrics
    plt.plot(df_recall_curve[x_label], df_recall_curve['mean_error'], 'purple',
             label='Mean Error', linewidth=2, marker='o', markersize=4)
    plt.plot(df_recall_curve[x_label], df_recall_curve['mean_absolute_error'], 'orange',
             label='Mean Absolute Error', linewidth=2, marker='s', markersize=4)

    # Add vertical line at optimal threshold
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {optimal_threshold:.3f}\n(Error: {optimal_error:.3f})')

    # Add zero line for mean error
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Zero Error')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Error')
    if title_flag:
        plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



def plot_both_curves(df_recall_curve,
                     x_label="confidence_threshold",
                     title_flag=False,
                        visualisations_path = None,
                     save_path_prefix=None):
    """
    Plot both precision/recall/F1 and error metrics curves.

    Args:
        df_recall_curve: DataFrame with all metrics
        x_label: Column name for x-axis
        title_flag: Whether to show titles
        save_path_prefix: Optional path prefix for saving (will append '_pr.png' and '_error.png')
    """
    pr_save_path = visualisations_path / f"{save_path_prefix}_precision_recall.png" if save_path_prefix else None
    error_save_path = visualisations_path / f"{save_path_prefix}_error.png" if save_path_prefix else None

    plot_precision_recall_f1_curve(df_recall_curve, x_label, title_flag, save_path=pr_save_path)
    plot_error_metrics_curve(df_recall_curve, x_label, title_flag, save_path=error_save_path)


def plot_error_curve_2(df_recall_curve,
                       x_label="confidence_threshold",
                       y_label="mean_error",
                       plot_error=False,
                       title="Error Curve",
                       save_path=None,
                       title_flag=False,
                       x_range=None, y_range=None):
    """
    Plot a single metric vs confidence threshold.

    Args:
        df_recall_curve: DataFrame with the data
        x_label: Column name for x-axis
        y_label: Column name for y-axis
        title: Plot title
        save_path: Optional path to save the plot
    """

    min_error_row = df_recall_curve.loc[df_recall_curve['mean_error'].abs().idxmin()]
    optimal_threshold = min_error_row['confidence_threshold']
    optimal_error = min_error_row['mean_error']
    logger.info(f"Confidence threshold with minimum mean_error: {optimal_threshold}")
    logger.info(f"Mean error: {optimal_error}")

    plt.figure(figsize=(10, 6))

    plt.plot(df_recall_curve[x_label], df_recall_curve[y_label],
             'b-', linewidth=2, marker='o', markersize=6, label='Mean Error')

    plt.xlabel(x_label.replace('_', ' ').title())
    plt.ylabel(y_label.replace('_', ' ').title())
    if title_flag:
        plt.title(title)
    plt.grid(True, alpha=0.3)

    # Set axis ranges
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    # Add zero line if plotting error metrics
    if plot_error:
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')

    # Add vertical line at optimal threshold
    plt.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7,
                label=f'Optimal Threshold: {optimal_threshold:.3f}\n(Error: {optimal_error:.2f})')

    # Add marker at the optimal point
    plt.plot(optimal_threshold, optimal_error, 'go', markersize=10,
             markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)

    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_single_metric_curve(df_recall_curve,
                             x_label="confidence_threshold",
                             y_label="mean_error",
                             plot_error = False,
                             title="Error Curve",
                             save_path=None,
                             title_flag = False,
                             x_range = None, y_range= None):
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
    if title_flag:
        plt.title(title)
    plt.grid(True, alpha=0.3)

    # Set axis ranges
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    # Add zero line if plotting error metrics
    if plot_error:
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



def plot_comprehensive_curves(df_recall_curve, save_path=None, title_flag = True):
    """
    Create a comprehensive 2x2 plot showing all metrics.
    """

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    if title_flag:
        fig.suptitle('Comprehensive Performance Analysis vs Confidence Threshold', fontsize=16, fontweight='bold')

    x = df_recall_curve['confidence_threshold']

    # Plot 1: Precision, Recall, F1
    ax1.plot(x, df_recall_curve['precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(x, df_recall_curve['recall'], 'r-', label='Recall', linewidth=2)
    ax1.plot(x, df_recall_curve['f1'], 'g-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')

    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # # Plot 2: Mean Error
    # ax2.plot(x, df_recall_curve['mean_error'], 'purple', linewidth=2, marker='o')
    # ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    # ax2.set_xlabel('Confidence Threshold')
    # ax2.set_ylabel('Mean Error')
    # ax2.set_title('Mean Error (Bias)')
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()