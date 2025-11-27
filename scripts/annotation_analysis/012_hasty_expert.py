"""
This script performs a analysis of expert annotations using Hasty data on the phase 4 goldstandard dataset which was labeled by four experts
"""
from typing import Dict

import numpy as np
from loguru import logger
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.util.visualisation.annotation_vis import visualise_points_only
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, ImageLabelCollection
from com.biospheredata.visualization.visualize_result import visualise_image, visualise_polygons
from playground.correction.basic_agreement import analyze_counts, count_user_annotations, compare_on_image_level, \
    visualize_multi_user_disagreement, plot_all_users_vs_consensus, analyse_normal_distribution_expert, \
    filter_by_agreement, analyze_count_dependent_bias
from playground.correction.correction_factor_analysis import calculate_correction_factors, \
    get_best_correction_for_expert, apply_ensemble_method
from playground.correction.error_type_analysis import analyze_error_compensation, identify_problematic_images, \
    get_consensus_values


def visualise_detections(df, title, image_column, images_path, output_path=None):
    grouped = df.groupby('images')

    # Color mapping for different classes
    unique_classes = df['class_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))

    print(f"Found {len(grouped)} unique images to process...")

    for image_name, image_data in grouped:
        print(f"\nProcessing: {image_name}")
        print(f"  - {len(image_data)} annotations")

        # Load image
        image_path = images_path / image_name
        if not image_path.exists():
            print(f"  - Warning: Image file not found at {image_path}")
            continue

        try:
            img = Image.open(image_path)
            img_array = np.array(img)
        except Exception as e:
            print(f"  - Error loading image: {e}")
            continue

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.imshow(img_array)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        ax.set_title(f"{image_name}\n{len(image_data)} annotations", fontsize=12, pad=20)

        # Plot annotations
        for idx, row in image_data.iterrows():
            class_name = row['class_name']
            color = class_color_map.get(class_name, 'red')

def visualise_hasty_annotation(image: ImageLabelCollection, images_path: Path,
                               output_path: Path | None = None,
                               show: bool = False, title: str | None = None,
                               suffix = "all_annotations", plot_boxes=False):
    linewidth = 1
    filename = output_path / f"{image.image_name}_{suffix}.jpg"

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    labels = image.labels
    if labels is None or len(labels) == 0:
        logger.warning(f"Image {image.image_name} has no labels")
    else:
        image_name = image.image_name
        i_width, i_height = image.width, image.height

        ax_ig = visualise_image(image_path=images_path / image_name, show=False, title=title,
                                figsize=(5, 5), dpi=150)

        # Create color mapping for class names
        unique_classes = list(set(il.class_name for il in labels if il.class_name is not None))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))  # Use tab10 colormap
        class_color_map: Dict[str, tuple] = dict(zip(unique_classes, colors))

        max_x = i_width
        max_y = i_height
        if max_x:
            plt.xlim(0, max_x)
        if max_y:
            plt.ylim(0, max_y)
        if title:
            plt.title(title)


        if plot_boxes:
            for i, il in enumerate(image.labels):
                if il.bbox_polygon is None:
                    continue

                polygon = il.bbox_polygon
                x, y = polygon.exterior.xy

                color = class_color_map.get(il.class_name, 'white')
                ax_ig.plot(x, y, color=color, linewidth=linewidth)
        else:
            for i, il in enumerate(image.labels):
                if il.centroid is None:
                    continue

                centroid = il.centroid
                x, y = centroid.x, centroid.y

                color = class_color_map.get(il.class_name, 'white')
                ax_ig.plot(x, y, 'o', color=color, markersize=5)  # 'o' for circular dots

        ax_ig.set_xticks([])
        ax_ig.set_yticks([])

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()
            plt.close()

        plt.tight_layout()

        return ax_ig



if __name__ == '__main__':
    hasty_data_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Goldstandard Quality Analysis/2025_04_09/")

    VISUALISE_ANNOTATIONS = True

    coco_path = "coco_annotations.json"
    hasty_path = hasty_data_path / "annotations.json"

    EXPERT_1 = "Iguana_Andrea"
    EXPERT_2 = "Iguana_Andres"
    EXPERT_3 = "Iguana_Amy"
    EXPERT_4 = "Iguana_Robin"
    All_Experts = [EXPERT_1, EXPERT_2, EXPERT_3, EXPERT_4]
    CONSENSUS = "consensus"

    # read the coco annotations
    coco_annotations = hasty_data_path / coco_path
    if not coco_annotations.exists():
        raise FileNotFoundError(f"COCO annotations file not found at {coco_annotations}")

    hA = HastyAnnotationV2.from_file(hasty_path)


    if VISUALISE_ANNOTATIONS:
        for i in hA.images:
            ic = i.copy()
            ie = i.copy()
            iref = i.copy()
            iref.labels = []
            if i.labels is None or len(i.labels) == 0:
                logger.warning(f"Image {i.image_name} has no labels, skipping visualization")
                continue
            # remove all labels which are not consensus
            ic.labels = [il for il in ic.labels if il.class_name == "concenso"]
            ie.labels = [il for il in ie.labels if il.class_name != "concenso"]

            visualise_hasty_annotation(iref, images_path=hasty_data_path / i.dataset_name,
                                       output_path= hasty_data_path / "visualizations", show=False,
                                       suffix="empty")

            visualise_hasty_annotation(ic, images_path=hasty_data_path / i.dataset_name,
                                       output_path= hasty_data_path / "visualizations", show=False,
                                       suffix="consensus_only")

            visualise_hasty_annotation(ie, images_path=hasty_data_path / i.dataset_name,
                                       output_path= hasty_data_path / "visualizations", show=False,
                                       suffix="expert_only")

    for expert in All_Experts:
        # create a the simple herdnet style dataframe
        hA_flat = hA.get_flat_df()
        hA_flat['x'] = hA_flat['centroid'].apply(lambda point: point.x)
        hA_flat['y'] = hA_flat['centroid'].apply(lambda point: point.y)
        hA_flat["labels"] = "iguana"
        hA_flat["species"] = "iguana"
        hA_flat["scores"] = 1 # no confidence score available because of human labellers

        hA_flat['class_name'] = hA_flat['class_name'].replace({
            'concenso': 'consensus',
            'conceso_parcial': 'consensus',
        })
        # rename a column
        hA_flat = hA_flat.rename(columns={'image_name': 'images'})
        image_list = hA_flat['images'].unique().tolist()
        image_list_all = [i.image_name for i in hA.images]

        df_ground_truth = hA_flat[hA_flat.class_name == "consensus"]
        basic_statistics = df_ground_truth['images'].value_counts()


        # images with at least one consensus annotation
        image_list_with_consensus = basic_statistics[basic_statistics > 0].index.tolist()

        radius = 25


        df_detections_amy = hA_flat[hA_flat.class_name == expert]

        basic_statistics_expert = df_detections_amy['images'].value_counts()
        image_labelled = df_detections_amy['images'].unique().tolist()
        image_blank = list(set(image_list_all) - set(image_labelled))

        print(f"images blank: {len(image_blank)}, images labelled: {len(image_labelled)}")

        df_false_positives_amy, df_true_positives_amy, df_false_negatives_amy, gdf_ground_truth_all = analyse_point_detections_greedy(
            df_detections=df_detections_amy,
            df_ground_truth=df_ground_truth,
            radius=radius,
            image_list=image_list_all
        )

        print(f"Expert {expert}: False Positives: {len(df_false_positives_amy)}, True Positives: {len(df_true_positives_amy)}, False_Negatives: {len(df_false_negatives_amy)}")
        # Calculate precision and recall
        precision = len(df_true_positives_amy) / (len(df_true_positives_amy) + len(df_false_positives_amy)) if (len(df_true_positives_amy) + len(df_false_positives_amy)) > 0 else 0
        recall = len(df_true_positives_amy) / (len(df_true_positives_amy) + len(df_false_negatives_amy)) if (len(df_true_positives_amy) + len(df_false_negatives_amy)) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Expert {expert}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")
        # Error

        from active_learning.util.converter import herdnet_prediction_to_hasty

        IL_detections_fp = herdnet_prediction_to_hasty(df_false_positives_amy, images_path=hasty_data_path / "GS 4th phase")
        IL_detections_fn = herdnet_prediction_to_hasty(df_false_negatives_amy,
                                                       images_path=hasty_data_path / "GS 4th phase")

        if VISUALISE_ANNOTATIONS:
            for i in IL_detections_fp:

                visualise_hasty_annotation(i, images_path=hasty_data_path / "GS 4th phase",
                                           output_path= hasty_data_path / "visualizations", show=False,
                                           suffix=f"{expert}_fp")
            for i in IL_detections_fn:

                visualise_hasty_annotation(i, images_path=hasty_data_path / "GS 4th phase",
                                           output_path= hasty_data_path / "visualizations", show=False,
                                           suffix=f"{expert}_fn")






    # Filter out images where 3+ experts agree exactly:
    hA_flat = hA.get_flat_df()
    hA_flat["labels"] = "iguana"
    hA_flat["species"] = "iguana"
    hA_flat["scores"] = 1  # no confidence score available because of human labellers

    hA_flat['class_name'] = hA_flat['class_name'].replace({
        'concenso': 'consensus',
        'conceso_parcial': 'consensus',
    })

    hA_flat['class_name'] = hA_flat['class_name'].replace({
        'Iguana_Andrea': 'Annotator 1',
        'Iguana_Andres': 'Annotator 2',
        'Iguana_Amy': 'Annotator 3',
        'Iguana_Robin': 'Annotator 4',
    })

    results = filter_by_agreement(
        hA_flat,
        expert_names=["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4"],
        agreement_threshold=3,
        agreement_type="exact",
        image_column="image_name"
    )

    pd.DataFrame(results["removed_images"])


    # hA_flat = results['filtered_data']

    # Run your correction factor analysis on the filtered data:

    filtered_correction_results = calculate_correction_factors(
        hA_flat,
        consensus_name="consensus",
        expert_names=["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4"]
    )


    # Run count analysis
    summary = analyze_counts(hA_flat)
    print(summary)

    count_user_annotations(hA_flat, username_1="Annotator 1", username_2="Annotator 2")



    count_user_annotations(hA_flat, username_1="Annotator 1", username_2="consensus")
    count_user_annotations(hA_flat, username_1="Annotator 2", username_2="consensus")
    count_user_annotations(hA_flat, username_1="Annotator 3", username_2="consensus")
    count_user_annotations(hA_flat, username_1="Annotator 4", username_2="consensus")

    results = compare_on_image_level(hA_flat, username_1="Annotator 1", username_2="consensus")

    # Access key metrics
    print(f"MAE: {results['mae']:.3f}")
    print(f"RMSE: {results['rmse']:.3f}")

    results = visualize_multi_user_disagreement(hA_flat)

    # Access specific metrics
    print(f"Average disagreement: {np.mean(results['std_per_image']):.2f}")
    print(f"Most disagreeing users: {results['mae_matrix'].values.max():.2f} MAE")

    # Run analysis
    results = plot_all_users_vs_consensus(hA_flat, consensus_name="consensus")

    # Test the normal distribution analysis
    normal_dist_results, fig = analyse_normal_distribution_expert(hA_flat,
                                                             username_1="Annotator 1",
                                                             username_2="consensus")
    fig.savefig(hasty_data_path / "visualizations" / "normal_distribution_annotator_1_vs_consensus.png")
    # Test the normal distribution analysis
    normal_dist_results, fig = analyse_normal_distribution_expert(hA_flat,
                                                             username_1="Annotator 2",
                                                             username_2="consensus")
    fig.savefig(hasty_data_path / "visualizations" / "normal_distribution_annotator_2_vs_consensus.png")

    # Test the normal distribution analysis
    normal_dist_results, fig = analyse_normal_distribution_expert(hA_flat,
                                                             username_1="Annotator 3",
                                                             username_2="consensus")
    fig.savefig(hasty_data_path / "visualizations" / "normal_distribution_annotator_3_vs_consensus.png")

    normal_dist_results, fig = analyse_normal_distribution_expert(hA_flat,
                                                             username_1="Annotator 4",
                                                             username_2="consensus")
    fig.savefig(hasty_data_path / "visualizations" / "normal_distribution_annotator_4_vs_consensus.png")

    analyze_count_dependent_bias(hA_flat)

    expert_analysis = analyze_error_compensation(
        hA_flat,
        consensus_name="consensus",
        expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]
    )

    # Find problematic images:

    all_images = sorted(hA_flat['image_name'].unique())
    consensus_values = get_consensus_values(hA_flat, all_images)  # You'll need to implement this

    problematic = identify_problematic_images(expert_analysis, consensus_values, all_images)

    print(f"Found {len(problematic)} problematic images:")
    for img_info in problematic[:5]:  # Show first 5
        print(f"  {img_info['image']}: consensus={img_info['consensus_count']}")
        for expert, data in img_info['expert_errors'].items():
            print(f"    {expert}: {data['count']} (error: {data['error']:+d})")



    # Run analysis with your data
    results = calculate_correction_factors(
        hA_flat,
        consensus_name="consensus",
        expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]
    )

    # Get the best method
    best_method = results['best_method']['best_method']
    print(f"Best method: {best_method}")

    # 3. Get the best overall method:

    best_method_name = results['best_method']['best_method']
    print(f"Best method: {best_method_name}")

    # 4. Get best correction for a specific expert:

    andrea_best = get_best_correction_for_expert(results, "Iguana_Andrea")
    print(f"Best method for Andrea: {andrea_best['method']}")
    print(f"Andrea's best MAE: {andrea_best['performance']['mae']:.3f}")

    # 5. Apply correction to new data:

    new_andrea_counts = [3, 7, 1, 12, 0, 5]
    corrected_counts = andrea_best['apply_function'](new_andrea_counts)
    print(f"Original: {new_andrea_counts}")
    print(f"Corrected: {corrected_counts}")

    # 6. Use ensemble method on new data:

    new_expert_data = {
        "Iguana_Andrea": [3, 7, 1, 12, 0, 5],
        "Iguana_Andres": [2, 8, 1, 11, 1, 4],
        "Iguana_Amy": [4, 6, 2, 13, 0, 6],
        "Iguana_Robin": [3, 7, 1, 10, 0, 5]
    }

    ensemble_counts = apply_ensemble_method(new_expert_data, 'weighted', results)
    print(f"Ensemble result: {ensemble_counts}")
