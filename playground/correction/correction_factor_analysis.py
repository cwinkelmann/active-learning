import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def calculate_correction_factors(hA_flat, consensus_name="consensus",
                                 expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]):
    """
    Calculate various types of correction factors for iguana counting methods.

    Parameters:
    - hA_flat: DataFrame with annotation data
    - consensus_name: Name of the consensus annotations (ground truth)
    - expert_names: List of expert annotator names

    Returns:
    - Dictionary with correction factors and analysis results
    """

    print("=== IGUANA COUNT CORRECTION FACTOR ANALYSIS ===\n")

    # Get all unique images
    all_images = sorted(hA_flat['image_name'].unique())

    # Get consensus counts (ground truth)
    consensus_data = hA_flat[hA_flat['class_name'] == consensus_name]
    consensus_counts = consensus_data.groupby('image_name').size()
    consensus_values = np.array([consensus_counts.get(img, 0) for img in all_images])

    print(f"Total images: {len(all_images)}")
    print(f"Consensus total count: {np.sum(consensus_values)}")
    print(f"Consensus mean per image: {np.mean(consensus_values):.2f}")
    print(f"Consensus std per image: {np.std(consensus_values):.2f}")

    # Calculate expert data
    expert_data = {}

    print(f"\n=== INDIVIDUAL EXPERT ANALYSIS ===")

    for expert in expert_names:
        if expert not in hA_flat['class_name'].unique():
            print(f"Warning: Expert '{expert}' not found in dataset!")
            continue

        # Get expert counts
        expert_df = hA_flat[hA_flat['class_name'] == expert]
        expert_counts = expert_df.groupby('image_name').size()
        expert_values = np.array([expert_counts.get(img, 0) for img in all_images])

        # Calculate basic metrics
        mae = mean_absolute_error(consensus_values, expert_values)
        rmse = np.sqrt(mean_squared_error(consensus_values, expert_values))
        r2 = r2_score(consensus_values, expert_values) if np.var(consensus_values) > 0 else 0

        # Calculate different types of correction factors
        cf_results = calculate_expert_correction_factors(consensus_values, expert_values, expert)

        expert_data[expert] = {
            'values': expert_values,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correction_factors': cf_results
        }

        print(f"\n{expert}:")
        print(f"  Total count: {np.sum(expert_values)} (vs consensus: {np.sum(consensus_values)})")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Simple ratio correction: {cf_results['simple_ratio']:.3f}")
        print(
            f"  Linear model correction: slope={cf_results['linear_slope']:.3f}, intercept={cf_results['linear_intercept']:.3f}")

    # Calculate ensemble methods
    print(f"\n=== ENSEMBLE METHODS ===")
    ensemble_results = calculate_ensemble_methods(expert_data, consensus_values, all_images)

    # Determine best method
    print(f"\n=== BEST METHOD RECOMMENDATION ===")
    best_method = determine_best_method(expert_data, ensemble_results, consensus_values)

    # Create comprehensive visualization
    create_correction_visualization(expert_data, ensemble_results, consensus_values, expert_names)

    return {
        'consensus_values': consensus_values,
        'expert_data': expert_data,
        'ensemble_results': ensemble_results,
        'best_method': best_method,
        'images': all_images
    }


def calculate_expert_correction_factors(consensus_values, expert_values, expert_name):
    """Calculate various correction factors for a single expert."""

    results = {}

    # 1. Simple ratio correction (multiplicative)
    total_consensus = np.sum(consensus_values)
    total_expert = np.sum(expert_values)
    simple_ratio = total_consensus / total_expert if total_expert > 0 else 1.0
    results['simple_ratio'] = simple_ratio

    # 2. Linear regression correction
    # Only fit on non-zero expert values to avoid division issues
    valid_mask = expert_values > 0
    if np.sum(valid_mask) > 1:
        X_valid = expert_values[valid_mask].reshape(-1, 1)
        y_valid = consensus_values[valid_mask]

        linear_model = LinearRegression()
        linear_model.fit(X_valid, y_valid)

        results['linear_slope'] = linear_model.coef_[0]
        results['linear_intercept'] = linear_model.intercept_
        results['linear_model'] = linear_model
    else:
        results['linear_slope'] = simple_ratio
        results['linear_intercept'] = 0
        results['linear_model'] = None

    # 3. Robust linear regression (using median)
    if np.sum(valid_mask) > 1:
        # Calculate slope using median of ratios
        ratios = consensus_values[valid_mask] / expert_values[valid_mask]
        robust_slope = np.median(ratios)
        results['robust_slope'] = robust_slope
    else:
        results['robust_slope'] = simple_ratio

    # 4. Count-dependent correction (piecewise linear)
    count_ranges = [(0, 2), (3, 5), (6, 10), (11, 1000)]  # Use large number instead of inf
    count_corrections = {}

    for low, high in count_ranges:
        mask = (expert_values >= low) & (expert_values <= high) & (expert_values > 0)
        if np.sum(mask) > 0:
            range_consensus = consensus_values[mask]
            range_expert = expert_values[mask]
            if np.sum(range_expert) > 0:
                range_ratio = np.sum(range_consensus) / np.sum(range_expert)
            else:
                range_ratio = simple_ratio
            count_corrections[f'{low}-{high}'] = range_ratio
        else:
            count_corrections[f'{low}-{high}'] = simple_ratio

    results['count_dependent'] = count_corrections

    # 5. Apply corrections and calculate corrected values
    corrected_values = {}

    # Simple ratio correction
    corrected_values['simple'] = expert_values * simple_ratio

    # Linear correction
    if results['linear_model'] is not None:
        corrected_values['linear'] = results['linear_model'].predict(expert_values.reshape(-1, 1))
        corrected_values['linear'] = np.maximum(corrected_values['linear'], 0)  # Ensure non-negative
    else:
        corrected_values['linear'] = corrected_values['simple']

    # Robust correction
    corrected_values['robust'] = expert_values * results['robust_slope']

    # Count-dependent correction
    corrected_count_dependent = expert_values.copy().astype(float)
    for i, count in enumerate(expert_values):
        for range_str, factor in count_corrections.items():
            low, high = range_str.split('-')
            low = int(low)
            high = int(high)
            if low <= count <= high:
                corrected_count_dependent[i] = count * factor
                break
    corrected_values['count_dependent'] = corrected_count_dependent

    results['corrected_values'] = corrected_values

    # Calculate performance metrics for each correction method
    performance = {}
    for method, corrected in corrected_values.items():
        mae = mean_absolute_error(consensus_values, corrected)
        rmse = np.sqrt(mean_squared_error(consensus_values, corrected))
        r2 = r2_score(consensus_values, corrected) if np.var(consensus_values) > 0 else 0

        performance[method] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    results['performance'] = performance

    return results


def calculate_ensemble_methods(expert_data, consensus_values, all_images):
    """Calculate ensemble-based correction methods."""

    ensemble_results = {}
    available_experts = list(expert_data.keys())

    if len(available_experts) < 2:
        print("Warning: Need at least 2 experts for ensemble methods")
        return ensemble_results

    # Prepare expert value matrix
    expert_matrix = np.array([expert_data[expert]['values'] for expert in available_experts]).T

    # 1. Simple average
    ensemble_average = np.mean(expert_matrix, axis=1)
    ensemble_results['average'] = {
        'values': ensemble_average,
        'mae': mean_absolute_error(consensus_values, ensemble_average),
        'rmse': np.sqrt(mean_squared_error(consensus_values, ensemble_average)),
        'r2': r2_score(consensus_values, ensemble_average) if np.var(consensus_values) > 0 else 0
    }

    # 2. Weighted average (weights based on individual R² scores)
    r2_scores = [max(expert_data[expert]['r2'], 0) for expert in available_experts]  # Ensure non-negative
    total_r2 = sum(r2_scores)
    if total_r2 > 0:
        weights = np.array(r2_scores) / total_r2
    else:
        weights = np.ones(len(r2_scores)) / len(r2_scores)

    ensemble_weighted = np.average(expert_matrix, axis=1, weights=weights)
    ensemble_results['weighted'] = {
        'values': ensemble_weighted,
        'weights': dict(zip(available_experts, weights)),
        'mae': mean_absolute_error(consensus_values, ensemble_weighted),
        'rmse': np.sqrt(mean_squared_error(consensus_values, ensemble_weighted)),
        'r2': r2_score(consensus_values, ensemble_weighted) if np.var(consensus_values) > 0 else 0
    }

    # 3. Median ensemble (robust to outliers)
    ensemble_median = np.median(expert_matrix, axis=1)
    ensemble_results['median'] = {
        'values': ensemble_median,
        'mae': mean_absolute_error(consensus_values, ensemble_median),
        'rmse': np.sqrt(mean_squared_error(consensus_values, ensemble_median)),
        'r2': r2_score(consensus_values, ensemble_median) if np.var(consensus_values) > 0 else 0
    }

    # 4. Machine learning ensemble (Random Forest)
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(expert_matrix, consensus_values)

        ensemble_rf = rf_model.predict(expert_matrix)
        ensemble_rf = np.maximum(ensemble_rf, 0)  # Ensure non-negative

        ensemble_results['random_forest'] = {
            'values': ensemble_rf,
            'model': rf_model,
            'feature_importance': dict(zip(available_experts, rf_model.feature_importances_)),
            'mae': mean_absolute_error(consensus_values, ensemble_rf),
            'rmse': np.sqrt(mean_squared_error(consensus_values, ensemble_rf)),
            'r2': r2_score(consensus_values, ensemble_rf) if np.var(consensus_values) > 0 else 0
        }
    except Exception as e:
        print(f"Warning: Could not create Random Forest ensemble: {e}")

    # 5. Best expert selection
    if available_experts:
        best_expert = min(available_experts, key=lambda x: expert_data[x]['mae'])
        ensemble_results['best_expert'] = {
            'expert': best_expert,
            'values': expert_data[best_expert]['values'],
            'mae': expert_data[best_expert]['mae'],
            'rmse': expert_data[best_expert]['rmse'],
            'r2': expert_data[best_expert]['r2']
        }

    print(f"Ensemble method performance:")
    for method, data in ensemble_results.items():
        if method != 'best_expert':
            print(f"  {method}: MAE={data['mae']:.3f}, RMSE={data['rmse']:.3f}, R²={data['r2']:.3f}")
        else:
            print(f"  {method} ({data['expert']}): MAE={data['mae']:.3f}, RMSE={data['rmse']:.3f}, R²={data['r2']:.3f}")

    return ensemble_results


def determine_best_method(expert_data, ensemble_results, consensus_values):
    """Determine the best counting method based on multiple criteria."""

    all_methods = {}

    # Add individual expert methods
    for expert, data in expert_data.items():
        # Original expert
        all_methods[f"{expert}_original"] = {
            'mae': data['mae'],
            'rmse': data['rmse'],
            'r2': data['r2'],
            'type': 'individual',
            'description': f"Original {expert} counts"
        }

        # Corrected expert methods
        for correction_method, performance in data['correction_factors']['performance'].items():
            method_name = f"{expert}_{correction_method}"
            all_methods[method_name] = {
                'mae': performance['mae'],
                'rmse': performance['rmse'],
                'r2': performance['r2'],
                'type': 'corrected_individual',
                'description': f"{expert} with {correction_method} correction"
            }

    # Add ensemble methods
    for method, data in ensemble_results.items():
        all_methods[f"ensemble_{method}"] = {
            'mae': data['mae'],
            'rmse': data['rmse'],
            'r2': data['r2'],
            'type': 'ensemble',
            'description': f"Ensemble {method}"
        }

    # Rank methods by different criteria
    rankings = {
        'mae': sorted(all_methods.items(), key=lambda x: x[1]['mae']),
        'rmse': sorted(all_methods.items(), key=lambda x: x[1]['rmse']),
        'r2': sorted(all_methods.items(), key=lambda x: x[1]['r2'], reverse=True)
    }

    print(f"Top 5 methods by MAE:")
    for i, (method, metrics) in enumerate(rankings['mae'][:5]):
        print(f"  {i + 1}. {method}: MAE={metrics['mae']:.3f}")

    print(f"\nTop 5 methods by RMSE:")
    for i, (method, metrics) in enumerate(rankings['rmse'][:5]):
        print(f"  {i + 1}. {method}: RMSE={metrics['rmse']:.3f}")

    print(f"\nTop 5 methods by R²:")
    for i, (method, metrics) in enumerate(rankings['r2'][:5]):
        print(f"  {i + 1}. {method}: R²={metrics['r2']:.3f}")

    # Overall best method (weighted score)
    best_method_name = rankings['mae'][0][0]
    best_method_metrics = rankings['mae'][0][1]

    print(f"\n🏆 RECOMMENDED BEST METHOD: {best_method_name}")
    print(f"   Description: {best_method_metrics['description']}")
    print(f"   MAE: {best_method_metrics['mae']:.3f}")
    print(f"   RMSE: {best_method_metrics['rmse']:.3f}")
    print(f"   R²: {best_method_metrics['r2']:.3f}")

    return {
        'best_method': best_method_name,
        'best_metrics': best_method_metrics,
        'all_methods': all_methods,
        'rankings': rankings
    }


def create_correction_visualization(expert_data, ensemble_results, consensus_values, expert_names):
    """Create comprehensive visualization of correction methods."""

    available_experts = [e for e in expert_names if e in expert_data]
    if not available_experts:
        print("No expert data available for visualization")
        return

    fig = plt.figure(figsize=(20, 16))

    # 1-4. Original vs Consensus scatter plots for each expert
    for i, expert in enumerate(available_experts[:4]):
        plt.subplot(4, 4, i + 1)
        expert_values = expert_data[expert]['values']
        plt.scatter(consensus_values, expert_values, alpha=0.6, s=30)

        max_val = max(np.max(consensus_values), np.max(expert_values))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        plt.xlabel('Consensus counts')
        plt.ylabel(f'{expert} counts')
        plt.title(f'{expert} vs Consensus\nMAE: {expert_data[expert]["mae"]:.2f}')
        plt.grid(True, alpha=0.3)

    # 5. Correction factor comparison
    plt.subplot(4, 4, 5)
    simple_ratios = [expert_data[e]['correction_factors']['simple_ratio'] for e in available_experts]
    linear_slopes = [expert_data[e]['correction_factors']['linear_slope'] for e in available_experts]

    x = np.arange(len(available_experts))
    width = 0.35

    plt.bar(x - width / 2, simple_ratios, width, label='Simple Ratio', alpha=0.7)
    plt.bar(x + width / 2, linear_slopes, width, label='Linear Slope', alpha=0.7)

    plt.xlabel('Experts')
    plt.ylabel('Correction Factor')
    plt.title('Correction Factors Comparison')
    plt.xticks(x, available_experts, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Performance comparison of correction methods
    plt.subplot(4, 4, 6)
    methods = ['simple', 'linear', 'robust', 'count_dependent']

    for expert in available_experts:
        maes = [expert_data[expert]['correction_factors']['performance'][method]['mae'] for method in methods]
        plt.plot(methods, maes, marker='o', label=expert, linewidth=2)

    plt.xlabel('Correction Method')
    plt.ylabel('MAE')
    plt.title('Correction Method Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 7. Ensemble method comparison
    plt.subplot(4, 4, 7)
    if ensemble_results:
        ensemble_methods = [m for m in ensemble_results.keys() if m != 'best_expert']
        if ensemble_methods:
            ensemble_maes = [ensemble_results[m]['mae'] for m in ensemble_methods]

            bars = plt.bar(ensemble_methods, ensemble_maes, alpha=0.7, color='lightblue')
            plt.xlabel('Ensemble Method')
            plt.ylabel('MAE')
            plt.title('Ensemble Method Performance')
            plt.xticks(rotation=45)

            # Add values on bars
            for bar, mae in zip(bars, ensemble_maes):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mae:.2f}', ha='center', va='bottom')

    # 8. Best corrected vs original scatter
    plt.subplot(4, 4, 8)
    if available_experts:
        # Find best individual correction method
        best_expert = min(available_experts,
                          key=lambda e: min(expert_data[e]['correction_factors']['performance'][m]['mae']
                                            for m in expert_data[e]['correction_factors']['performance']))
        best_method = min(expert_data[best_expert]['correction_factors']['performance'],
                          key=lambda m: expert_data[best_expert]['correction_factors']['performance'][m]['mae'])

        original_values = expert_data[best_expert]['values']
        corrected_values = expert_data[best_expert]['correction_factors']['corrected_values'][best_method]

        plt.scatter(consensus_values, original_values, alpha=0.5, label='Original', s=30)
        plt.scatter(consensus_values, corrected_values, alpha=0.7, label='Corrected', s=30)

        max_val = max(np.max(consensus_values), np.max(corrected_values))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        plt.xlabel('Consensus counts')
        plt.ylabel('Predicted counts')
        plt.title(f'Best Correction: {best_expert} ({best_method})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 9-12. Individual expert correction comparisons
    for i, expert in enumerate(available_experts[:4]):
        plt.subplot(4, 4, 9 + i)

        original = expert_data[expert]['values']
        corrections = expert_data[expert]['correction_factors']['corrected_values']

        # Plot original
        plt.scatter(consensus_values, original, alpha=0.4, label='Original', s=20, color='gray')

        # Plot best correction method for this expert
        best_correction = min(corrections.keys(),
                              key=lambda m: expert_data[expert]['correction_factors']['performance'][m]['mae'])
        corrected = corrections[best_correction]

        plt.scatter(consensus_values, corrected, alpha=0.7, label=f'{best_correction}', s=30)

        max_val = max(np.max(consensus_values), np.max(corrected))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        plt.xlabel('Consensus counts')
        plt.ylabel('Predicted counts')
        plt.title(f'{expert} Best Correction')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def apply_correction_to_new_data(expert_counts, correction_method, correction_params):
    """
    Apply a trained correction method to new expert count data.

    Parameters:
    - expert_counts: Array of expert counts for new images
    - correction_method: Type of correction ('simple', 'linear', 'robust', etc.)
    - correction_params: Parameters from the correction factor analysis

    Returns:
    - Corrected counts
    """

    expert_counts = np.array(expert_counts)

    if correction_method == 'simple':
        return expert_counts * correction_params['simple_ratio']

    elif correction_method == 'linear':
        if correction_params['linear_model'] is not None:
            predictions = correction_params['linear_model'].predict(expert_counts.reshape(-1, 1))
            return np.maximum(predictions, 0)  # Ensure non-negative
        else:
            return expert_counts * correction_params['simple_ratio']

    elif correction_method == 'robust':
        return expert_counts * correction_params['robust_slope']

    elif correction_method == 'count_dependent':
        corrected = expert_counts.copy().astype(float)
        for i, count in enumerate(expert_counts):
            for range_str, factor in correction_params['count_dependent'].items():
                low, high = range_str.split('-')
                low = int(low)
                high = int(high)
                if low <= count <= high:
                    corrected[i] = count * factor
                    break
        return corrected

    else:
        raise ValueError(f"Unknown correction method: {correction_method}")


def get_best_correction_for_expert(results, expert_name):
    """Get the best correction method and parameters for a specific expert."""

    if expert_name not in results['expert_data']:
        raise ValueError(f"Expert '{expert_name}' not found in results")

    expert_data = results['expert_data'][expert_name]
    performance = expert_data['correction_factors']['performance']

    # Find best method by MAE
    best_method = min(performance.keys(), key=lambda m: performance[m]['mae'])
    best_performance = performance[best_method]

    return {
        'method': best_method,
        'performance': best_performance,
        'params': expert_data['correction_factors'],
        'apply_function': lambda counts: apply_correction_to_new_data(
            counts, best_method, expert_data['correction_factors']
        )
    }


def apply_ensemble_method(expert_counts_dict, ensemble_type, results):
    """
    Apply ensemble method to new expert count data.

    Parameters:
    - expert_counts_dict: Dictionary with expert names as keys and count arrays as values
    - ensemble_type: Type of ensemble ('average', 'weighted', 'median', 'random_forest')
    - results: Results from calculate_correction_factors

    Returns:
    - Ensemble corrected counts
    """

    if ensemble_type not in results['ensemble_results']:
        raise ValueError(f"Ensemble type '{ensemble_type}' not found in results")

    # Prepare expert matrix
    expert_names = list(expert_counts_dict.keys())
    expert_matrix = np.array([expert_counts_dict[expert] for expert in expert_names]).T

    if ensemble_type == 'average':
        return np.mean(expert_matrix, axis=1)

    elif ensemble_type == 'median':
        return np.median(expert_matrix, axis=1)

    elif ensemble_type == 'weighted':
        weights_dict = results['ensemble_results']['weighted']['weights']
        weights = [weights_dict.get(expert, 0) for expert in expert_names]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        return np.average(expert_matrix, axis=1, weights=weights)

    elif ensemble_type == 'random_forest':
        if 'model' in results['ensemble_results']['random_forest']:
            model = results['ensemble_results']['random_forest']['model']
            predictions = model.predict(expert_matrix)
            return np.maximum(predictions, 0)  # Ensure non-negative
        else:
            raise ValueError("Random Forest model not available")

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


# Example usage function integrated with your existing code
def example_integration_with_existing_code():
    """
    Example of how to integrate this with your existing analysis code.
    """

    print("=== INTEGRATION EXAMPLE ===")
    print("""
    # 1. After loading your data (from your existing code):

    hA = HastyAnnotationV2.from_file(hasty_path)
    hA_flat = hA.get_flat_df()

    # Clean the consensus names (from your existing code)
    hA_flat['class_name'] = hA_flat['class_name'].replace({
        'concenso': 'consensus',
        'conceso_parcial': 'consensus'
    })

    # 2. Run the correction factor analysis:

    EXPERT_NAMES = ["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]
    CONSENSUS = "consensus"

    results = calculate_correction_factors(
        hA_flat, 
        consensus_name=CONSENSUS, 
        expert_names=EXPERT_NAMES
    )

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
    """)


if __name__ == "__main__":
    example_integration_with_existing_code()