"""
Data which was produced by the training data preparation data
001_hasty_to_tile_point_detection.py
046_prepare_pix4d_orthomosaics_detection_hard_neg.py

Load each folder and analyse what is in there
"""
import numpy as np
import typing

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

"""
A marked iguana is not consistent in size an sharpness etc


"""
import PIL
import pandas as pd
from matplotlib import pyplot as plt

from active_learning.util.image_manipulation import crop_out_individual_object
from active_learning.util.visualisation.annotation_vis import plot_frequency_distribution, plot_visibility_scatter, \
    plot_image_grid_by_visibility

"""
Create patches from images and labels from hasty to be used in CVAT
"""
import shutil
from loguru import logger
from pathlib import Path
import geopandas as gpd

from active_learning.pipelines.data_prep import AnnotationsIntermediary


## TODO Download annotations from hasty
base_path = Path("/Users/christian/data/training_data/2025-07-02")

if __name__ == "__main__":


    template = "_detection"

    mt_training_data = [
        # "All",
        "Fernandina_m",
        "Fernandina_s",
        "Floreana",
        "Genovesa"
    ]
    mt_training_data = [f"{s}{template}" for s in mt_training_data] + [f"Floreana{template}_il_{i}" for i in
                                                                       range(1, 36)]

    splits = ["train", "val", "test"]
    gdf_all = gpd.read_parquet(
        '/Users/christian/data/Iguanas_From_Above/2020_2021_2022_2023_2024_database_analysis_ready.parquet')

    report: typing.List = []

    for dataset in mt_training_data:  # , "val", "test"]:
        for split in splits:
            output_path_dset = base_path / dataset

            if not output_path_dset.joinpath(split).exists():
                logger.info(f"Dataset {dataset}, {split} not found, skipping")
                continue

            hA_crops = HastyAnnotationV2.from_file( output_path_dset / split / "hasty_format_crops_512_0.json" )
            output_path_analysis = output_path_dset / "analysis"
            output_path_analysis.mkdir(parents=True, exist_ok=True)
            # keep only the images with labels

            individual_object_cropped_labels = []

            hA_crops.get_flat_df()

            num_annotations = len(hA_crops.get_flat_df())
            report.append( {"dataset": dataset, "split": split, "num_annotations": num_annotations} )


    # Visualise the report
    df_report = pd.DataFrame(report)

    # Plot the frequency distribution of annotations per dataset and split
    # Separate main datasets from iterative learning datasets
    main_df = df_report[~df_report['dataset'].str.contains('_il_')].copy()
    il_df = df_report[df_report['dataset'].str.contains('_il_')].copy()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Main datasets - grouped bar chart
    ax1 = plt.subplot(2, 2, (1, 2))

    # Prepare data for grouped bar chart
    main_pivot = main_df.pivot(index='dataset', columns='split', values='num_annotations')
    main_pivot.index = main_pivot.index.str.replace('_detection', '')

    # Create grouped bar chart
    x = np.arange(len(main_pivot.index))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, main_pivot['train'], width, label='Train',
                    color='#3B82F6', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, main_pivot['val'], width, label='Validation',
                    color='#EF4444', alpha=0.8)

    # Customize main chart
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Annotations', fontsize=12, fontweight='bold')
    ax1.set_title('Main Datasets - Train vs Validation Split', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(main_pivot.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)


    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)


    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)

    # Iterative Learning chart
    ax2 = plt.subplot(2, 2, (3, 4))

    # Prepare IL data
    il_df['iteration'] = il_df['dataset'].str.extract(r'il_(\d+)').astype(int)
    il_df_sorted = il_df.sort_values('iteration')

    bars3 = ax2.bar(range(len(il_df_sorted)), il_df_sorted['num_annotations'],
                    color='#10B981', alpha=0.8)

    # Customize IL chart
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Annotations', fontsize=12, fontweight='bold')
    # ax2.set_title('Floreana Iterative Learning Progression', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(il_df_sorted)))
    ax2.set_xticklabels([f'IL-{i}' for i in il_df_sorted['iteration']], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels for IL chart
    for i, (bar, value) in enumerate(zip(bars3, il_df_sorted['num_annotations'])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{int(value):,}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Create summary statistics
    total_annotations = df_report['num_annotations'].sum()
    main_annotations = main_df['num_annotations'].sum()
    il_annotations = il_df['num_annotations'].sum()

    print("=" * 60)
    print("DATASET ANNOTATIONS SUMMARY")
    print("=" * 60)
    print(f"Total Annotations:           {total_annotations:,}")
    print(f"Main Datasets:               {main_annotations:,}")
    print(f"Iterative Learning:          {il_annotations:,}")
    print("=" * 60)

    # Print detailed breakdown
    print("\nMAIN DATASETS BREAKDOWN:")
    print("-" * 40)
    for dataset in main_pivot.index:
        train_val = main_pivot.loc[dataset]
        train_count = train_val['train'] if pd.notna(train_val['train']) else 0
        val_count = train_val['val'] if pd.notna(train_val['val']) else 0
        total = train_count + val_count
        print(f"{dataset:20} | Train: {train_count:4,} | Val: {val_count:4,} | Total: {total:4,}")

    print("\nITERATIVE LEARNING PROGRESSION:")
    print("-" * 40)
    for _, row in il_df_sorted.iterrows():
        iteration = row['iteration']
        count = row['num_annotations']
        print(f"Iteration {iteration:2d}           | Annotations: {count:3,}")

    # Save the plot
    plt.savefig(base_path / 'dataset_annotations_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # Create a detailed data table
    print("\n" + "=" * 80)
    print("COMPLETE DATASET TABLE")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Split':<10} {'Annotations':>12}")
    print("-" * 52)

    for _, row in df_report.iterrows():
        dataset = row['dataset']
        split = row['split']
        annotations = row['num_annotations']
        print(f"{dataset:<30} {split:<10} {annotations:>12,}")

    print("=" * 80)


    # Optional: Create individual charts for detailed analysis
    def create_individual_charts():
        """Create separate detailed charts"""

        # 1. Main datasets only - horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        main_df_plot = main_df.copy()
        main_df_plot['dataset_clean'] = main_df_plot['dataset'].str.replace('_detection', '')
        main_df_plot['split_annotations'] = main_df_plot['split'] + ': ' + main_df_plot['num_annotations'].astype(str)

        # Create horizontal bar chart
        colors = {'train': '#3B82F6', 'val': '#EF4444'}

        for i, split in enumerate(['train', 'val']):
            split_data = main_df_plot[main_df_plot['split'] == split]
            y_pos = np.arange(len(split_data)) + i * 0.4
            ax.barh(y_pos, split_data['num_annotations'], 0.35,
                    label=split.capitalize(), color=colors[split], alpha=0.8)

            # Add value labels
            for j, (pos, value) in enumerate(zip(y_pos, split_data['num_annotations'])):
                ax.text(value + 50, pos, f'{value:,}', va='center', fontsize=9)

        ax.set_yticks(np.arange(len(main_df_plot['dataset_clean'].unique())) + 0.2)
        ax.set_yticklabels(main_df_plot['dataset_clean'].unique())
        ax.set_xlabel('Number of Annotations')
        # ax.set_title('Main Datasets - Detailed View')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(base_path / 'main_datasets_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Iterative learning trend line
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(il_df_sorted['iteration'], il_df_sorted['num_annotations'],
                marker='o', linewidth=2, markersize=8, color='#10B981')
        ax.fill_between(il_df_sorted['iteration'], il_df_sorted['num_annotations'],
                        alpha=0.3, color='#10B981')

        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Number of Annotations')
        ax.set_title('Floreana Iterative Learning - Growth Trend')
        ax.grid(True, alpha=0.3)

        # Add value labels on points
        for x, y in zip(il_df_sorted['iteration'], il_df_sorted['num_annotations']):
            ax.annotate(f'{y}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(base_path / 'iterative_learning_trend.png', dpi=300, bbox_inches='tight')
        plt.show()


    # Uncomment to create additional detailed charts
    create_individual_charts()


