"""
This script performs a analysis of expert annotations using Hasty data on the phase 4 goldstandard dataset which was labeled by four experts
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from playground.correction.basic_agreement import analyze_counts, count_user_annotations, compare_on_image_level, \
    visualize_multi_user_disagreement, plot_all_users_vs_consensus, analyse_normal_distribution_expert, \
    filter_by_agreement
from playground.correction.correction_factor_analysis import calculate_correction_factors, \
    get_best_correction_for_expert, apply_ensemble_method
from playground.correction.error_type_analysis import analyze_error_compensation, identify_problematic_images, \
    get_consensus_values





if __name__ == '__main__':
    hasty_data_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Goldstandard Quality Analysis/2025_04_09/")

    coco_path = "coco_annotations.json"
    hasty_path = hasty_data_path / "annotations.json"

    EXPERT_1 = "Iguana_Andrea"
    EXPERT_2 = "Iguana_Andres"
    EXPERT_3 = "Iguana_Amy"
    EXPERT_4 = "Iguana_Robin"

    CONSENSUS = "consensus"

    # read the coco annotations
    coco_annotations = hasty_data_path / coco_path
    if not coco_annotations.exists():
        raise FileNotFoundError(f"COCO annotations file not found at {coco_annotations}")

    hA = HastyAnnotationV2.from_file(hasty_path)

    hA_flat = hA.get_flat_df()

    hA_flat['class_name'] = hA_flat['class_name'].replace({
        'concenso': 'consensus',
        'conceso_parcial': 'consensus'
    })

    # Filter out images where 3+ experts agree exactly:

    results = filter_by_agreement(
        hA_flat,
        expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"],
        agreement_threshold=3,
        agreement_type="exact"
    )

    # Use the filtered data for further analysis:

    hA_flat = results['filtered_data']

    # Run your correction factor analysis on the filtered data:

    filtered_correction_results = calculate_correction_factors(
        hA_flat,
        consensus_name="consensus",
        expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]
    )


    # Run count analysis
    summary = analyze_counts(hA_flat)
    print(summary)

    count_user_annotations(hA_flat, username_1="Iguana_Andrea", username_2="Iguana_Andres")
    count_user_annotations(hA_flat, username_1="Iguana_Robin", username_2="consensus")

    results = compare_on_image_level(hA_flat, username_1="Iguana_Andrea", username_2="consensus")

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
    normal_dist_results = analyse_normal_distribution_expert(hA_flat,
                                                             username_1=EXPERT_1,
                                                             username_2="consensus")

    # Test the normal distribution analysis
    normal_dist_results = analyse_normal_distribution_expert(hA_flat,
                                                             username_1=EXPERT_2,
                                                             username_2="consensus")


    # Test the normal distribution analysis
    normal_dist_results = analyse_normal_distribution_expert(hA_flat,
                                                             username_1=EXPERT_3,
                                                             username_2="consensus")


    normal_dist_results = analyse_normal_distribution_expert(hA_flat,
                                                             username_1=EXPERT_4,
                                                             username_2="consensus")

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
