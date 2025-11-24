"""
Take Detections from a model, compare them with ground truth and display them in a FiftyOne Dataset

see inference_test to get the detections

"""

# import pytest

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from loguru import logger
from pathlib import Path
from typing import Optional

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import Evaluator, plot_confidence_density, plot_single_metric_curve, \
    plot_comprehensive_curves, plot_error_curve_2, plot_both_curves, plot_fp_tp_confidence_histogram
from active_learning.util.visualisation.draw import draw_thumbnail


# import pytest


# class PlotPrecisionRecall:
#
#     def __init__(
#             self,
#             figsize: tuple = (7, 7),
#             legend: bool = True,  # Changed to True for species plotting
#             seed: int = 1,
#             best_f1 = True
#     ) -> None:
#
#         self.figsize = figsize
#         self.legend = legend
#         self.seed = seed
#
#         self._data = []
#         self._labels = []
#
#     def feed(self, recalls: list, precisions: list, label: Optional[str] = None) -> None:
#         """Feed precision-recall data for a single curve"""
#         self._data.append((recalls, precisions))
#         self._labels.append(label)
#
#     def plot(self) -> None:
#         """Generate the precision-recall plot"""
#         random.seed(self.seed)
#         colors = self._gen_colors(len(self._data))
#
#         fig = plt.figure(figsize=self.figsize)
#         ax = fig.add_subplot(1, 1, 1)
#         ax.set_xlim(0, 1.02)
#         ax.set_ylim(0, 1.02)
#         ax.set_xlabel('Recall', fontsize=12)
#         ax.set_ylabel('Precision', fontsize=12)
#         ax.set_title('Precision-Recall Curve by Species', fontsize=14)
#         ax.grid(True, alpha=0.3)
#
#         markers = self._markers
#         for i, (recall, precision) in enumerate(self._data):
#             ax.plot(recall, precision,
#                     color=colors[i],
#                     marker=next(markers),
#                     markevery=0.1,
#                     alpha=0.7,
#                     linewidth=2,
#                     label=self._labels[i])
#
#         if self.legend:
#             lg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
#                           ncol=min(3, len(self._data)), frameon=True, fancybox=True)
#             # Adjust layout to prevent legend cutoff
#             plt.tight_layout()
#
#         self.fig = fig
#
#     def save(self, path: Path) -> None:
#         """Save the plot to a file"""
#         if 'fig' not in self.__dict__:
#             self.plot()
#
#         self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')
#
#     def _gen_colors(self, n: int) -> list:
#         """Generate random colors for each curve"""
#         colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#                   for i in range(n)]
#         return colors
#
#     @property
#     def _markers(self) -> itertools.cycle:
#         """Return a cycle of marker styles"""
#         return itertools.cycle(('^', 'o', 's', 'x', 'D', 'v', '>'))


class PlotPrecisionRecall:

    def __init__(
            self,
            figsize: tuple = (7, 7),
            legend: bool = True,
            seed: int = 1,
            best_f1: bool = True
    ) -> None:

        self.figsize = figsize
        self.legend = legend
        self.seed = seed
        self.best_f1 = best_f1

        self._data = []
        self._labels = []
        self._best_f1_data = []
        self._ap_scores = []

    def feed(self, recalls: list, precisions: list, label: Optional[str] = None) -> None:
        """Feed precision-recall data and calculate F1 scores and AP"""
        recalls = np.array(recalls)
        precisions = np.array(precisions)

        # Calculate F1 scores
        f1_scores = np.zeros_like(recalls)
        for i in range(len(recalls)):
            if precisions[i] + recalls[i] > 0:
                f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            else:
                f1_scores[i] = 0

        # Find best F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_f1_score = f1_scores[best_f1_idx]
        best_f1_recall = recalls[best_f1_idx]
        best_f1_precision = precisions[best_f1_idx]

        # Calculate Average Precision (AP) using trapezoidal rule
        # Sort by recall to ensure proper integration
        sorted_indices = np.argsort(recalls)
        sorted_recalls = recalls[sorted_indices]
        sorted_precisions = precisions[sorted_indices]
        ap = np.trapz(sorted_precisions, sorted_recalls)

        self._data.append((recalls.tolist(), precisions.tolist()))
        self._labels.append(label)
        self._best_f1_data.append({
            'recall': best_f1_recall,
            'precision': best_f1_precision,
            'f1': best_f1_score
        })
        self._ap_scores.append(ap)

    def plot(self) -> None:

        random.seed(self.seed)
        colors = self._gen_colors(len(self._data))

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)

        # Calculate mAP
        mAP = np.mean(self._ap_scores) if self._ap_scores else 0
        ax.set_title(f'Precision-Recall Curve (mAP: {mAP:.3f})', fontsize=14)

        markers = self._markers
        for i, (recall, precision) in enumerate(self._data):
            # Plot PR curve with AP in label
            label_with_ap = f"{self._labels[i]} (AP: {self._ap_scores[i]:.3f})" if self._labels[
                i] else f"AP: {self._ap_scores[i]:.3f}"
            ax.plot(recall, precision,
                    color=colors[i],
                    marker=next(markers),
                    markevery=0.1,
                    alpha=0.7,
                    linewidth=2,
                    label=label_with_ap)

            # Add vertical line at best F1 score
            if self.best_f1:
                best_f1_info = self._best_f1_data[i]
                ax.axvline(x=best_f1_info['recall'],
                           color=colors[i],
                           linestyle='--',
                           alpha=0.5,
                           linewidth=1.5)
                # Add marker at best F1 point
                ax.plot(best_f1_info['recall'],
                        best_f1_info['precision'],
                        marker='*',
                        markersize=15,
                        color=colors[i],
                        markeredgecolor='black',
                        markeredgewidth=1,
                        zorder=5)

        if self.legend:
            # Place legend below the plot
            lg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                            ncol=min(3, len(self._data)), frameon=True, fancybox=True,
                            fontsize=9)
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()

        self.fig = fig

    def save(self, path: Path) -> None:
        if 'fig' not in self.__dict__:
            self.plot()

        self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')

    def _gen_colors(self, n: int) -> list:

        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(n)]

        return colors

    @property
    def _markers(self) -> itertools.cycle:
        return itertools.cycle(('^', 'o', 's', 'x', 'D', 'v', '>'))


def plot_precision_recall_by_species(df_detections, df_ground_truth, radius, visualisations_path,
                                     confidence_thresholds=None):
    """
    Plot precision-recall curves for each species class

    Args:
        df_detections: DataFrame with detections including 'species' column
        df_ground_truth: DataFrame with ground truth including 'species' column
        radius: Matching radius for detection evaluation
        visualisations_path: Path to save visualizations
        confidence_thresholds: List of confidence thresholds to evaluate at
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.0, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98,
                                 0.99, 0.999, 0.9999, 1.0]

    # Get unique species from both detections and ground truth
    species_list = sorted(set(df_detections['species'].unique()) |
                          set(df_ground_truth['species'].unique()))

    logger.info(f"Found {len(species_list)} species: {species_list}")

    # Initialize the plotter
    pr_plotter = PlotPrecisionRecall(figsize=(10, 8), legend=True, best_f1 = True)

    # For each species, calculate precision-recall curve
    for species in species_list:
        logger.info(f"Calculating PR curve for species: {species}")

        # Filter detections and ground truth for this species
        df_det_species = df_detections[df_detections['species'] == species].copy()
        df_gt_species = df_ground_truth[df_ground_truth['species'] == species].copy()

        if len(df_gt_species) == 0:
            logger.warning(f"No ground truth for species {species}, skipping")
            continue

        if len(df_det_species) == 0:
            logger.warning(f"No detections for species {species}, skipping")
            continue

        # Create evaluator for this species
        ev_species = Evaluator(df_detections=df_det_species,
                               df_ground_truth=df_gt_species,
                               radius=radius)

        # Get precision-recall curve
        df_pr_curve = ev_species.get_precition_recall_curve(values=confidence_thresholds)

        # Extract precision and recall values
        recalls = df_pr_curve['recall'].tolist()
        precisions = df_pr_curve['precision'].tolist()

        # Feed to plotter
        pr_plotter.feed(recalls, precisions, label=f"{species} (n={len(df_gt_species)})")

        logger.info(f"  Species {species}: GT={len(df_gt_species)}, Det={len(df_det_species)}, "
                    f"Final P={precisions[-1]:.3f}, R={recalls[-1]:.3f}")

    # Save the plot
    save_path = visualisations_path / "precision_recall_by_species.png"
    pr_plotter.save(save_path)
    logger.info(f"Saved species-wise precision-recall plot to {save_path}")

    # Also show the plot
    pr_plotter.plot()
    plt.show()

    return pr_plotter

def evaluate_point_detections(base_path: Path, df_detections: pd.DataFrame, herdnet_annotation_name,
                              images_path: Path, suffix, radius, box_size, CONFIDENCE_THRESHOLD):
    visualisations_path = base_path / "visualisations"
    visualisations_path.mkdir(exist_ok=True, parents=True)
    IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    image_list_all = [i.image_name for i in IL_detections]

    df_ground_truth = pd.read_csv(base_path / herdnet_annotation_name)

    images = list(images_path.glob(f"*.{suffix}"))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {images_path}" )

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
                                      title="Confidence Score Histogram Distribution of True Positives",
                                      save_path=visualisations_path / "true_positives_confidence_density.png")
    plt.show()

    plot_fp_tp_confidence_histogram(df_false_positives, df_true_positives,
                                    title_flag=False,
                                    show_stats=False,
                                    bins=30, stacked=True,
                                    save_path=visualisations_path / "fp_tp_histogram_stats.png")
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
    
    confidence_values = [0.0, 0.01, 0.05, 0.08,
                                                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                            0.7, 0.72, 0.75, 0.78,
                                                            0.8, 0.82, 0.85, 0.88,
                                                            0.9,
                                                            0.95,
                                                            0.96, 0.97, 0.98, 0.99,
                                                            0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998,
                                                            0.999,
                                                            0.9999, 1.0]
    
    df_recall_curve = ev.get_precition_recall_curve(values=confidence_values)

    # plot_species_detection_analysis(df_recall_curve, title="Performance Analysis", save_path=visualisations_path / "species_performance_analysis.png",)
    # Plot comprehensive view
    plot_both_curves(df_recall_curve, title_flag=False, save_path_prefix="fwk_error_",
                     visualisations_path=visualisations_path)

    # TODO plot that QQ Plot here to see if there is a bias depending on animals in the images

    min_error_row = df_recall_curve.loc[df_recall_curve['mean_error'].abs().idxmin()]
    optimal_threshold = min_error_row['confidence_threshold']
    logger.info(f"Confidence threshold with minimum mean_error: {optimal_threshold}")
    logger.info(f"Mean error: {min_error_row['mean_error']}")

    # Plot single metric
    plot_error_curve_2(df_recall_curve, y_label='mean_error',
                             title="Mean Error Analysis", title_flag = False, plot_error=True,
                             save_path=visualisations_path / "mean_error_curve.png", )

    # TODO plot Precison Score Curve
    plot_single_metric_curve(df_recall_curve,
                             x_label="recall", y_label='precision',
                             x_range = (0.0, 1.0), y_range=(0.0, 1.0),
                             title_flag = False,
                             title="Precision Recall Curve",
                             save_path=visualisations_path / "precision_recall_curve.png", )

    # Plot all metrics separately
    plot_comprehensive_curves(df_recall_curve,
                              save_path=visualisations_path / "comprehensive_performance_analysis.png", title_flag = False)

    # NEW: Plot precision-recall curves by species
    # TODO add best f1 score

    if 'species' in df_detections.columns and 'species' in df_ground_truth.columns:
        logger.info("Generating species-wise precision-recall curves...")
        plot_precision_recall_by_species(
            df_detections=df_detections,  # Use unfiltered detections for PR curve
            df_ground_truth=df_ground_truth,
            radius=radius,
            confidence_thresholds=confidence_values,
            visualisations_path=visualisations_path
        )
    else:
        logger.warning("Species column not found in detections or ground truth, skipping species-wise PR curves")

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
        draw_thumbnail(df_fp[df_fp.scores > 0.99], i,
                       suffix="fp_hc",
                       images_path=visualisations_path, box_size=box_size,
                       df_gt=gdf_gt,
                       title_flag=False)
        #
        # df_fp_hc = df_fp[df_fp.scores > 0.9]
        #
        logger.info(f"Drawing false positives to {visualisations_path} done.")
        #
        # high medium confidence false positives
        draw_thumbnail(df_fp[(df_fp.scores > 0.8) & (df_fp.scores < 0.9)], i, suffix="fp_hmc",
                       images_path=visualisations_path, box_size=box_size,
                       df_gt=gdf_gt)
        logger.info(f"Drawing false positives to {visualisations_path} done.")
        #
        # # low confidence false positives
        # draw_thumbnail(df_fp[(df_fp.scores > 0.4) & (df_fp.scores < 0.7)], i, suffix="fp_lc",
        #                images_path=visualisations_path, box_size=box_size,
        #                df_gt=gdf_gt)
        # #
        # # low confidence false positives
        # # draw_thumbnail(df_fp[df_fp.scores <= 0.4], i, suffix="fp_lc", images_path=visualisations_path, box_size=box_size)
        #
        # False negatives
        if len(df_fn) > 0:
            draw_thumbnail(df_fn, i, suffix="fn",
                           images_path=visualisations_path, box_size=box_size,
                           DETECTECTED_COLOR="red",
                           GT_COLOR="blue",
                           df_gt=gdf_gt)

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
    # ds= "delplanque2023"
    # ds= "Eikelboom2019_October_30"
    # ds= "Eikelboom2019_October_30_recall_90"
    ds= "iguana_fwk"

    suffix = "JPG"
    
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

    if ds == "Eikelboom2019_October_30":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/eikelboom2019/eikelboom_512_overlap_0_ebFalse/eikelboom_test/test/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-31/09-05-22/detections.csv') # dla102

        # rename_species = {
        #     "Buffalo": "Elephant",
        #     "Alcelaphinae": "Giraffe",
        #     "Kob": "Zebra"
        # }
        # df_detections['species'] = df_detections['species'].replace(rename_species)
        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"


    if ds == "Eikelboom2019_October_30_recall_90":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/eikelboom2019/eikelboom_512_overlap_0_ebFalse/eikelboom_test/test/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-31/09-06-54/detections.csv') # dla102

        # rename_species = {
        #     "Buffalo": "Elephant",
        #     "Alcelaphinae": "Giraffe",
        #     "Kob": "Zebra"
        # }
        # df_detections['species'] = df_detections['species'].replace(rename_species)
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

    elif ds == "iguana_fwk":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_fwk/val')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-11-22/14-42-58/detections.csv') # dla34


        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
        suffix = "jpg"

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
        base_path = Path(f'/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/Delplanque2022_512_overlap_0_ebFalse/delplanque_test/test')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-30/10-11-32/detections.csv')
        dataset_name = "delplanque_test"
        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
    # /test/Default
    else:
        raise ValueError("Unknown dataset: " + ds)
        


    # hA_ground_truth_path = base_path / hasty_annotation_name
    # hA_ground_truth = HastyAnnotationV2.from_file(hA_ground_truth_path)

    # ## On cropped images
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47/detections.csv')
    # hasty_annotation_name = 'hasty_format_crops_512_0.json'
    # herdnet_annotation_name = 'herdnet_format_512_0_crops.csv'
    # images_path = base_path / "crops_512_numNone_overlap0"
    # suffix = "jpg"

    box_size = 224
    radius = 150
    CONFIDENCE_THRESHOLD = 0.0

    evaluate_point_detections(base_path, 
                              df_detections, 
                              herdnet_annotation_name, 
                              images_path,
                              suffix, radius, box_size, CONFIDENCE_THRESHOLD)


