"""
Analyse the box sizes of a Hasty Annotation dataset.


"""
import json
import shapely

import shutil

import uuid

import pandas as pd
from pathlib import Path

from active_learning.util.image import get_image_id, get_image_dimensions
from active_learning.util.visualisation.annotation_vis import create_simple_histograms, \
    visualise_hasty_annotation_statistics, plot_bbox_sizes
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, ImageLabel, HastyAnnotationV2, LabelClass

from loguru import logger
from pathlib import Path

def parse_bbox_and_calculate_area(bbox: shapely.Polygon):
    try:
        area = bbox.area
        width = bbox.bounds[2] - bbox.bounds[0]  # x2 - x1
        height = bbox.bounds[3] - bbox.bounds[1]
        return area, width, height
    except Exception as e:
        logger.error(f"Error parsing bbox {bbox}: {e}")
        return 0, 0, 0


def create_label_distribution_table(df_flat, group_by, target_labels = ['iguana', "iguana_point"]):
    """
    Create label distribution table grouped by dataset

    Args:
        df_flat: Flat dataframe from Hasty annotations

    Returns:
        pandas.DataFrame: Label distribution table
    """

    # Define the label categories we're interested in


    # Create pivot table for label distribution
    # Group by dataset and label, count occurrences
    label_counts = df_flat.groupby([group_by, 'class_name']).size().reset_index(name='count')

    # Pivot to get labels as columns
    pivot_table = label_counts.pivot(index=group_by, columns='class_name', values='count').fillna(0)

    # Ensure all target labels are present as columns
    for label in target_labels:
        if label not in pivot_table.columns:
            pivot_table[label] = 0

    # Reorder columns to match the original table
    pivot_table = pivot_table[target_labels]



    # Convert to int type
    pivot_table = pivot_table.astype(int)

    # Sort by total count (descending)
    pivot_table['total'] = pivot_table.sum(axis=1)
    pivot_table = pivot_table.sort_values('total', ascending=False)
    pivot_table = pivot_table.drop('total', axis=1)

    return pivot_table


def get_aggregations(df_flat, group_by=['class_name']):
    """

    :param df_flat:
    :return:
    """
    # Apply bbox parsing to create new columns
    bbox_data = df_flat['bbox_polygon'].apply(parse_bbox_and_calculate_area)
    df_filtered = df_flat.copy()  # Avoid SettingWithCopyWarning
    df_filtered['bbox_area'] = [x[0] for x in bbox_data]
    df_filtered['bbox_width'] = [x[1] for x in bbox_data]
    df_filtered['bbox_height'] = [x[2] for x in bbox_data]

    # Create aggregated statistics table
    aggregation_stats = df_filtered.groupby(group_by).agg({
        'image_id': ['count', 'nunique'],  # count of annotations, unique images
        'bbox_area': ['mean', 'std'],
        'bbox_width': ['mean', 'std'],
        'bbox_height': ['mean', 'std']
    }).round(2)


    # Flatten column names
    aggregation_stats.columns = ['_'.join(col).strip() for col in aggregation_stats.columns]

    # Rename columns for clarity
    column_mapping = {
        'image_id_count': 'Annotations',
        'image_id_nunique': 'Images',
        'bbox_area_mean': 'Avg. Box Area',
        'bbox_area_std': 'Box Area Std.',
        'bbox_area_min': 'Min. Box Area',
        'bbox_area_max': 'Max. Box Area',
        'bbox_width_mean': 'Avg. Box Width',
        'bbox_width_std': 'Box Width Std.',
        'bbox_height_mean': 'Avg. Box Height',
        'bbox_height_std': 'Box Height Std.',

    }
    aggregation_stats = aggregation_stats.rename(columns=column_mapping)

    # Calculate average annotations per image
    if 'Annotations' in aggregation_stats.columns and 'Images' in aggregation_stats.columns:
        aggregation_stats['Avg. Ann./Image'] = (
                aggregation_stats['Annotations'] / aggregation_stats['Images']
        ).round(2)
    else:
        raise ValueError('aggregation_stats must have "Annotations" and "Images"')

    # Reset index to make dataset_name and class_name regular columns
    aggregation_stats = aggregation_stats.reset_index()

    logger.info("Dataset Analysis Summary:")
    logger.info("=" * 50)
    logger.info(f"Total records in filtered dataset: {len(df_filtered)}")
    logger.info(f"Unique images: {df_filtered['image_id'].nunique()}")
    logger.info(f"Unique classes: {df_filtered['class_name'].nunique()}")
    logger.info("\nClass distribution:")
    class_counts = df_filtered['class_name'].value_counts()
    logger.info(class_counts)

    return aggregation_stats


if __name__ == "__main__":
    base_path = Path("/Users/christian/data/training_data/2025_07_10_final_analysis/unzipped_hasty_annotation")
    base_path = Path("/raid/cwinkelmann/training_data/eikelboom2019/")


    hA = HastyAnnotationV2.from_file(base_path / "eikelboom_hasty.json")
    annotated_images = hA.images

    annotated_images = [ai for ai in annotated_images if ai.dataset_name not in [
        "Zooniverse_expert_phase_3", "Zooniverse_expert_phase_2"]]

    # keep only "iguana"
    # annotated_images = [
    #     AnnotatedImage(
    #         **{**ai.__dict__, 'labels': [label for label in ai.labels if label.class_name == "iguana"]}
    #     )
    #     for ai in annotated_images
    # ]
    # create plots for the dataset
    pd.DataFrame(hA.dataset_statistics())



    visualise_hasty_annotation_statistics(annotated_images)

    dataset_names = set(ai.dataset_name for ai in annotated_images)

    df_flat = hA.get_flat_df()
    aggregation_stats = get_aggregations(df_flat, group_by=['class_name'])

    for split in dataset_names:
        create_simple_histograms(annotated_images, dataset_name=split)
        annotated_images_split = [ai for ai in annotated_images if ai.dataset_name == split]
        # create_simple_histograms(annotated_images_split)
        plot_bbox_sizes(annotated_images_split, suffix=f"{split}", plot_name = f"box_sizes_{split}.png")

        aggregation_stats = get_aggregations(df_flat[df_flat['dataset_name'] == split], group_by=['class_name'])

        # visualise_hasty_annotation_statistics(annotated_images_split)

        print(f"{split}: {aggregation_stats}")