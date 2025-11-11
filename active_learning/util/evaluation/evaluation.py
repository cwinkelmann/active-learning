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



def plot_error_curve(df_recall_curve,
                     x_label="confidence_threshold",
                     y_label="mean_error",
                    title_flag=False,
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

    if title_flag:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    # Find confidence threshold where mean_error is closest to zero
    optimal_idx = df_recall_curve['f1'].abs().idxmax()
    optimal_threshold = df_recall_curve.loc[optimal_idx, x_label]
    optimal_f1_score = df_recall_curve.loc[optimal_idx, 'f1']
    # Add vertical line at optimal threshold
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {optimal_threshold:.3f}\n(F1 Score: {optimal_f1_score:.3f})')

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

    # Find confidence threshold where mean_error is closest to zero
    optimal_idx = df_recall_curve['mean_error'].abs().idxmin()
    optimal_threshold = df_recall_curve.loc[optimal_idx, x_label]
    optimal_error = df_recall_curve.loc[optimal_idx, 'mean_error']
    # Add vertical line at optimal threshold
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {optimal_threshold:.3f}\n(Error: {optimal_error:.3f})')
    # Add zero line for mean error
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Zero Error')



    # Formatting
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Mean Error & MAE', color='black')
    ax2.set_title('Error Metrics vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)

    # # Combine legends
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax2.legend(lines, labels, loc='upper left')
    ax2.legend(loc='upper left')


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_single_metric_curve(df_recall_curve,
                             x_label="confidence_threshold",
                             y_label="mean_error",
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
    if 'error' in y_label.lower():
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