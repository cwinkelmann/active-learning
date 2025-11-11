from codecs import ignore_errors
from typing import List

import pandas as pd
import geopandas as gpd
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_geospatial_hungarian, \
    create_matching_visualization_geojson
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.types.Exceptions import NoLabelsError
from active_learning.util.convenience_functions import get_tiles
from active_learning.util.evaluation.evaluation import plot_confidence_density, Evaluator, plot_error_curve
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def simple_scaling_calibration(gdf_enriched_grid,
                               pred_column='predictions', ref_column='reference'):
    """
    Simple proportional scaling to match totals.
    """
    pred_total = gdf_enriched_grid[pred_column].sum()
    ref_total = gdf_enriched_grid[ref_column].sum()

    scaling_factor = ref_total / pred_total

    calibrated_gdf = gdf_enriched_grid.copy()
    calibrated_gdf[f'{pred_column}_calibrated'] = calibrated_gdf[pred_column] * scaling_factor

    logger.info(f"Original total: {pred_total:.2f}, Reference total: {ref_total:.2f}, Difference: {(ref_total - pred_total):.2f}")
    logger.info(f"Scaling factor: {scaling_factor:.4f}")
    logger.info(f"Calibrated total: {calibrated_gdf[f'{pred_column}_calibrated'].sum():.2f}")

    logger.debug(f"Assuimg the reference is correct, the predictor missed by {((1 - scaling_factor)*100):.2f}% ")

    return calibrated_gdf, scaling_factor


def global_estimate_geospatial_correction_factor(gdf_enriched_grid,
                                                 pred_column='predictions', ref_column='reference'):
    """
    using a enriched grid estimate the needed correction factor by using the density
    :param gdf_enriched_grid:
    :return:
    """

    # TODO group by prediction
    group_stats = {}
    for reference_count, group in gdf_enriched_grid.groupby(ref_column):
        group_stats[reference_count] = {
            "group_size": len(group),
            "avg_avg_score": group["avg_score"].mean(),
            "reference_count": reference_count,
            "avg_predictions": group[pred_column].mean(),
            "median_predictions": group[pred_column].median(),
            "diff_predictions": group[pred_column].mean() - reference_count,
        }

    df_group_stats = pd.DataFrame(group_stats)

    return df_group_stats


def compare_geospatial_slices(gdf_grid, gdf_sliced_reference: gpd.GeoDataFrame, gdf_sliced_detections: gpd.GeoDataFrame):
    """

    :param gdf_sliced_reference:
    :param gdf_sliced_detections:
    :return:
    """

    gdf_grid = gdf_grid.reset_index(drop=False, inplace=False).rename(columns={"index": "grid_id"}, inplace=False)
    pred_stats = gdf_sliced_detections.groupby('grid_id').agg(
        predictions=('geometry', 'count'),
        avg_score=('scores', 'mean'),
        min_score=('scores', 'min'),
        median_score=('scores', 'median'),
    ).reset_index()

    ref_stats = gdf_sliced_reference.groupby('grid_id').size().reset_index(name='reference')

    gdf_grid = gdf_grid.merge(pred_stats, on='grid_id', how='left')
    gdf_grid = gdf_grid.merge(ref_stats, on='grid_id', how='left')
    gdf_grid = gdf_grid.fillna(0)

    gdf_grid["abs_diff"] = abs(gdf_grid["predictions"] - gdf_grid["reference"])

    gdf_grid["pct_diff"] = gdf_grid.apply(
        lambda row: ((row["predictions"] - row["reference"]) / row["reference"] * 100)
        if row["reference"] > 0 else 0,
        axis=1
    )

    return gdf_grid

def estimate_geospatial_correction_factor(prediction_path,
         reference_path,
         orthomosaic_path,
         outputdir,
         vis_output_dir,
         tile_size=5000,
         radius = 0.6,
         confidence_threshold = 0.1):
    """
    Main function to compare geospatial predictions with reference data and evaluate performance.

    A

    :param prediction_path:
    :param reference_path:
    :param orthomosaic_path:
    :param outputdir:
    :param vis_output_dir:
    :param tile_size:
    :param radius:
    :param confidence_threshold:
    :return:
    """


    gdf_detections = gpd.read_file(prediction_path)
    gdf_detections.rename(columns={"images": "tile_name"}, inplace=True)
    gdf_reference = gpd.read_file(reference_path)
    # df_detections.species = "iguana_point" # if that does not show up in CVAT you will have to create it manually

    # everything should be in the right projection already, but just to be sure
    gdf_detections = project_gdfcrs(gdf_detections, orthomosaic_path)
    gdf_reference = project_gdfcrs(gdf_reference, orthomosaic_path)

    gdf_detections = gdf_detections[gdf_detections.scores > confidence_threshold]
    gdf_detections.drop(columns=["x", "y", "count_1", "count_2", "count_3", "count_4", "count_5", "count_6", "count_7"], inplace=True)
    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    gdf_grid = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0)
    slicer_occupied = GeoSlicer(base_path=orthomosaic_path.parent,
                                image_name=orthomosaic_path.name,
                                grid=gdf_grid,
                                output_dir=outputdir)


    grid_manager.gdf_raster_mask.to_file(vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)
    gdf_grid.to_file(vis_output_dir / f"grid_all_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

    try:
        gdf_sliced_reference = slicer_occupied.slice_annotations_regular_grid(gdf_reference, gdf_grid)
    except NoLabelsError:
        gdf_sliced_reference = gpd.GeoDataFrame(pd.DataFrame(columns=["tile_name", "grid_id",  "geometry", "grid_geometry", "pixel_x", "pixel_y", "local_pixel_x", "local_pixel_y", "image_name"]), crs=gdf_grid.crs)
    try:
        gdf_sliced_detections = slicer_occupied.slice_annotations_regular_grid(gdf_detections, gdf_grid)
    except NoLabelsError:
        gdf_sliced_detections = gpd.GeoDataFrame(pd.DataFrame(columns=["tile_name_left", "grid_id", "species", "scores", "dscores",  "geometry", "grid_geometry", "pixel_x", "pixel_y", "local_pixel_x", "local_pixel_y", "image_name"]), crs=gdf_grid.crs)

    gdf_sliced_detections = gdf_sliced_detections[["tile_name_left", "grid_id", "species", "scores", "dscores",  "geometry", "grid_geometry", "pixel_x", "pixel_y", "local_pixel_x", "local_pixel_y", "image_name"]]
    gdf_sliced_detections.rename(columns={"tile_name_left": "tile_name"}, inplace=True)
    gdf_sliced_reference = gdf_sliced_reference[["tile_name", "grid_id",  "geometry", "grid_geometry", "pixel_x", "pixel_y", "local_pixel_x", "local_pixel_y", "image_name"]]


    gdf_grid_enriched = compare_geospatial_slices(gdf_grid, gdf_sliced_reference, gdf_sliced_detections)


    # Analyse the grid statististic

    gdf_detections["orthomosaic_name"] = orthomosaic_path.stem
    gdf_reference["orthomosaic_name"] = orthomosaic_path.stem

    # TODO this should be done per tile
    gdf_false_positives, gdf_true_positives, gdf_false_negatives = analyse_point_detections_geospatial_hungarian(
        gdf_detections=gdf_detections,
        gdf_ground_truth=gdf_reference,
        radius_m=radius,
        tile_name="orthomosaic_name",
        tile_name_prediction="orthomosaic_name",
    )
    logger.info(f"Both predictors agreed on {len(gdf_true_positives)}, predictor a found {len(gdf_false_positives)} predictor b didn't and missed {len(gdf_false_negatives)}")

    gdf_concat = pd.concat([gdf_false_positives, gdf_true_positives, gdf_false_negatives])
    gdf_concat = gpd.GeoDataFrame(gdf_concat, geometry=gdf_concat.geometry)
    gdf_concat.drop(columns=["buffer"], inplace=True)
    # .to_file(vis_output_dir / f"grid_all_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)
    gdf_concat


    return {
        "gdf_detections": gdf_detections,
        "gdf_reference": gdf_reference,
        "gdf_grid_enriched": gdf_grid_enriched,
        # gdf_false_positives, gdf_true_positives, gdf_false_negatives
        "gdf_false_positives": gdf_false_positives,
        "gdf_true_positives": gdf_true_positives,
        "gdf_false_negatives": gdf_false_negatives,
        "gdf_concat": gdf_concat,
    }


def plot_density_scaling_factors(density_scaling_factors, overall_scaling_factor=None,
                                 save_path=None, show=True, title=None):
    """
    Visualize the relationship between reference counts and predictions,
    showing bias at different densities.

    Parameters:
    -----------
    density_scaling_factors : dict or DataFrame
        Contains 'reference_count', 'avg_predictions', 'median_predictions',
        'diff_predictions', 'group_size'
    overall_scaling_factor : float, optional
        Overall scaling factor to display
    save_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """


    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Extract data
    reference_count = np.array(density_scaling_factors['reference_count'])
    avg_predictions = np.array(density_scaling_factors['avg_predictions'])
    median_predictions = np.array(density_scaling_factors['median_predictions'])
    diff_predictions = np.array(density_scaling_factors['diff_predictions'])
    group_size = np.array(density_scaling_factors['group_size'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    # ===== Plot 1: Main scatter plot with perfect agreement line =====
    ax1 = axes[0, 0]

    # Plot perfect agreement line
    max_val = max(reference_count.max(), avg_predictions.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2,
             label='Perfect Agreement (y=x)', zorder=1)

    # Plot mean predictions with size based on group_size
    sizes = (group_size / group_size.max()) * 500 + 50
    scatter = ax1.scatter(reference_count, avg_predictions, s=sizes,
                          c=diff_predictions, cmap='RdYlGn_r', alpha=0.7,
                          edgecolors='black', linewidth=1.5, zorder=3)

    # # Connect points with line
    # ax1.plot(reference_count, avg_predictions, 'b-', alpha=0.4, linewidth=1.5, zorder=2)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Prediction Bias\n(Avg Pred - Reference)', rotation=270, labelpad=20)

    ax1.set_xlabel('Reference Count (Ground Truth)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Predicted Count', fontsize=12, fontweight='bold')
    ax1.set_title('Predictions vs Reference Count\n(Bubble size = sample size)',
                  fontsize=11, pad=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Bias vs Density =====
    ax2 = axes[0, 1]

    # Create bar plot of bias
    colors = ['#d62728' if x > 0.5 else '#2ca02c' if x < -0.5 else '#ffcc00'
              for x in diff_predictions]
    bars = ax2.bar(range(len(reference_count)), diff_predictions, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add zero line
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, diff_predictions)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=8, fontweight='bold')

    ax2.set_xlabel('Reference Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Bias (Avg Pred - Ref)', fontsize=12, fontweight='bold')
    ax2.set_title('Bias at Different Density Levels\n(Green: under-predict, Red: over-predict)',
                  fontsize=11, pad=10)
    ax2.set_xticks(range(len(reference_count)))
    ax2.set_xticklabels([f'{int(x)}' for x in reference_count], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Mean vs Median Predictions =====
    ax3 = axes[1, 0]

    x_pos = np.arange(len(reference_count))
    width = 0.35

    ax3.bar(x_pos - width / 2, avg_predictions, width, label='Mean Predictions',
            alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax3.bar(x_pos + width / 2, median_predictions, width, label='Median Predictions',
            alpha=0.7, color='coral', edgecolor='black', linewidth=1.5)
    # ax3.plot(x_pos, reference_count, 'ko-', linewidth=2, markersize=8,
    #          label='Reference Count', zorder=5)

    ax3.set_xlabel('Reference Count Level', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Mean vs Median Predictions Compared to Reference',
                  fontsize=11, pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{int(x)}' for x in reference_count], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: Sample Size Distribution =====
    ax4 = axes[1, 1]

    # Log scale for better visualization if needed
    use_log = group_size.max() / group_size[group_size > 0].min() > 100

    bars = ax4.bar(range(len(reference_count)), group_size,
                   color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, group_size)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(val)}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    if use_log:
        ax4.set_yscale('log')

    ax4.set_xlabel('Reference Count', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sample Size (Number of Images)', fontsize=12, fontweight='bold')
    ax4.set_title('Sample Size Distribution per Density Level', fontsize=11, pad=10)
    ax4.set_xticks(range(len(reference_count)))
    ax4.set_xticklabels([f'{int(x)}' for x in reference_count], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add statistics text box
    total_samples = group_size.sum()
    mae = np.abs(diff_predictions).mean()
    me = diff_predictions.mean()
    rmse = np.sqrt(np.mean(diff_predictions ** 2))

    textstr = f'Overall Statistics:\n'
    textstr += f'Total Samples: {int(total_samples)}\n'
    textstr += f'MAE: {mae:.3f}\n'
    textstr += f'RMSE: {rmse:.3f}\n'
    textstr += f'Counting Error: {me:.3f}\n'
    if overall_scaling_factor:
        textstr += f'Scaling Factor: {overall_scaling_factor:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.98, 0.30, textstr, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes