import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def analyze_error_compensation(hA_flat, consensus_name="consensus",
                               expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"]):
    """
    Analyze the error compensation patterns in expert annotations.

    This function examines how experts make errors that compensate across images,
    resulting in good total counts despite per-image inaccuracies.
    """

    print("=== ERROR COMPENSATION ANALYSIS ===\n")

    # Get all unique images
    all_images = sorted(hA_flat['image_name'].unique())

    # Get consensus counts (ground truth)
    consensus_data = hA_flat[hA_flat['class_name'] == consensus_name]
    consensus_counts = consensus_data.groupby('image_name').size()
    consensus_values = np.array([consensus_counts.get(img, 0) for img in all_images])
    consensus_total = np.sum(consensus_values)

    print(f"Dataset Overview:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Consensus total count: {consensus_total}")
    print(f"  Consensus mean per image: {np.mean(consensus_values):.2f}")
    print(f"  Consensus std per image: {np.std(consensus_values):.2f}")

    # Analyze each expert
    expert_analysis = {}

    print(f"\n=== EXPERT ERROR PATTERNS ===")

    for expert in expert_names:
        if expert not in hA_flat['class_name'].unique():
            print(f"Warning: Expert '{expert}' not found in dataset!")
            continue

        # Get expert counts
        expert_df = hA_flat[hA_flat['class_name'] == expert]
        expert_counts = expert_df.groupby('image_name').size()
        expert_values = np.array([expert_counts.get(img, 0) for img in all_images])
        expert_total = np.sum(expert_values)

        # Calculate errors and compensation metrics
        analysis = analyze_expert_compensation(consensus_values, expert_values, expert, consensus_total)
        expert_analysis[expert] = analysis

        print(f"\n{expert}:")
        print(f"  Total count: {expert_total} (vs consensus: {consensus_total})")
        print(f"  Total error: {analysis['total_error']:+d} ({analysis['total_error_pct']:+.1f}%)")
        print(f"  Mean absolute error per image: {analysis['mae']:.2f}")
        print(f"  Images overcounted: {analysis['overcount_images']} ({analysis['overcount_pct']:.1f}%)")
        print(f"  Images undercounted: {analysis['undercount_images']} ({analysis['undercount_pct']:.1f}%)")
        print(f"  Images exact: {analysis['exact_images']} ({analysis['exact_pct']:.1f}%)")
        print(f"  Compensation ratio: {analysis['compensation_ratio']:.2f}")
        print(f"  Error correlation with count level: r={analysis['error_correlation']:.3f}")

    # Create comprehensive visualization
    create_compensation_visualization(expert_analysis, consensus_values, all_images)

    # Analyze systematic patterns
    print(f"\n=== SYSTEMATIC ERROR PATTERNS ===")
    analyze_systematic_patterns(expert_analysis, consensus_values)

    return expert_analysis


def analyze_expert_compensation(consensus_values, expert_values, expert_name, consensus_total):
    """Analyze compensation patterns for a single expert."""

    analysis = {}

    # Basic metrics
    expert_total = np.sum(expert_values)
    total_error = expert_total - consensus_total
    total_error_pct = (total_error / consensus_total) * 100 if consensus_total > 0 else 0

    # Per-image errors
    image_errors = expert_values - consensus_values
    abs_errors = np.abs(image_errors)
    mae = np.mean(abs_errors)

    # Error direction analysis
    overcount_mask = image_errors > 0
    undercount_mask = image_errors < 0
    exact_mask = image_errors == 0

    overcount_images = np.sum(overcount_mask)
    undercount_images = np.sum(undercount_mask)
    exact_images = np.sum(exact_mask)

    total_images = len(image_errors)
    overcount_pct = (overcount_images / total_images) * 100
    undercount_pct = (undercount_images / total_images) * 100
    exact_pct = (exact_images / total_images) * 100

    # Compensation analysis
    total_overcount = np.sum(image_errors[overcount_mask]) if overcount_images > 0 else 0
    total_undercount = abs(np.sum(image_errors[undercount_mask])) if undercount_images > 0 else 0

    # Compensation ratio: how much undercounting compensates for overcounting
    compensation_ratio = min(total_overcount, total_undercount) / max(total_overcount, total_undercount) if max(
        total_overcount, total_undercount) > 0 else 0

    # Error correlation with count level
    if np.std(consensus_values) > 0:
        error_correlation = np.corrcoef(consensus_values, image_errors)[0, 1]
    else:
        error_correlation = 0

    # Error magnitude analysis by count level
    count_levels = [(0, 2), (3, 5), (6, 10), (11, float('inf'))]
    error_by_level = {}

    for low, high in count_levels:
        if high == float('inf'):
            mask = consensus_values >= low
            level_name = f"{low}+"
        else:
            mask = (consensus_values >= low) & (consensus_values <= high)
            level_name = f"{low}-{high}"

        if np.sum(mask) > 0:
            level_mae = np.mean(abs_errors[mask])
            level_bias = np.mean(image_errors[mask])
            level_images = np.sum(mask)

            error_by_level[level_name] = {
                'mae': level_mae,
                'bias': level_bias,
                'images': level_images,
                'overcount': np.sum(overcount_mask & mask),
                'undercount': np.sum(undercount_mask & mask),
                'exact': np.sum(exact_mask & mask)
            }

    # Store all analysis results
    analysis.update({
        'expert_name': expert_name,
        'expert_values': expert_values,
        'image_errors': image_errors,
        'abs_errors': abs_errors,
        'expert_total': expert_total,
        'total_error': total_error,
        'total_error_pct': total_error_pct,
        'mae': mae,
        'overcount_images': overcount_images,
        'undercount_images': undercount_images,
        'exact_images': exact_images,
        'overcount_pct': overcount_pct,
        'undercount_pct': undercount_pct,
        'exact_pct': exact_pct,
        'total_overcount': total_overcount,
        'total_undercount': total_undercount,
        'compensation_ratio': compensation_ratio,
        'error_correlation': error_correlation,
        'error_by_level': error_by_level
    })

    return analysis


def create_compensation_visualization(expert_analysis, consensus_values, all_images):
    """Create comprehensive visualization of error compensation patterns."""

    available_experts = list(expert_analysis.keys())
    n_experts = len(available_experts)

    if n_experts == 0:
        print("No expert data available for visualization")
        return

    fig = plt.figure(figsize=(20, 16))

    # 1. Total count comparison (bar chart)
    plt.subplot(4, 4, 1)
    consensus_total = np.sum(consensus_values)
    expert_totals = [expert_analysis[expert]['expert_total'] for expert in available_experts]

    x = np.arange(len(available_experts) + 1)
    totals = [consensus_total] + expert_totals
    labels = ['Consensus'] + available_experts
    colors = ['red'] + ['lightblue'] * n_experts

    bars = plt.bar(x, totals, color=colors, alpha=0.7)
    plt.ylabel('Total Count')
    plt.title('Total Counts Comparison')
    plt.xticks(x, labels, rotation=45)

    # Add values on bars
    for bar, total in zip(bars, totals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(totals) * 0.01,
                 str(total), ha='center', va='bottom')

    # 2. Error compensation visualization
    plt.subplot(4, 4, 2)
    for expert in available_experts:
        overcount = expert_analysis[expert]['total_overcount']
        undercount = expert_analysis[expert]['total_undercount']

        plt.barh(expert, overcount, color='red', alpha=0.7, label='Overcount' if expert == available_experts[0] else "")
        plt.barh(expert, -undercount, color='blue', alpha=0.7,
                 label='Undercount' if expert == available_experts[0] else "")

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Total Error (Overcount - Undercount)')
    plt.title('Error Compensation Patterns')
    plt.legend()

    # 3. Error distribution histograms
    plt.subplot(4, 4, 3)
    for expert in available_experts:
        errors = expert_analysis[expert]['image_errors']
        plt.hist(errors, bins=20, alpha=0.6, label=expert, density=True)

    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No error')
    plt.xlabel('Error per Image (Expert - Consensus)')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()

    # 4. MAE vs Total Error scatter
    plt.subplot(4, 4, 4)
    maes = [expert_analysis[expert]['mae'] for expert in available_experts]
    total_errors = [abs(expert_analysis[expert]['total_error']) for expert in available_experts]

    plt.scatter(maes, total_errors, s=100, alpha=0.7)
    for i, expert in enumerate(available_experts):
        plt.annotate(expert, (maes[i], total_errors[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

    plt.xlabel('Mean Absolute Error per Image')
    plt.ylabel('Total Error (Absolute)')
    plt.title('Individual vs Total Error')
    plt.grid(True, alpha=0.3)

    # 5. Compensation ratio comparison
    plt.subplot(4, 4, 5)
    comp_ratios = [expert_analysis[expert]['compensation_ratio'] for expert in available_experts]
    bars = plt.bar(available_experts, comp_ratios, alpha=0.7, color='green')
    plt.ylabel('Compensation Ratio')
    plt.title('Error Compensation Efficiency')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Add values on bars
    for bar, ratio in zip(bars, comp_ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{ratio:.2f}', ha='center', va='bottom')

    # 6. Error by count level heatmap
    plt.subplot(4, 4, 6)
    count_levels = []
    expert_biases = []

    # Get all count levels
    first_expert = available_experts[0]
    levels = list(expert_analysis[first_expert]['error_by_level'].keys())

    bias_matrix = []
    for expert in available_experts:
        expert_bias_row = []
        for level in levels:
            if level in expert_analysis[expert]['error_by_level']:
                bias = expert_analysis[expert]['error_by_level'][level]['bias']
                expert_bias_row.append(bias)
            else:
                expert_bias_row.append(0)
        bias_matrix.append(expert_bias_row)

    bias_df = pd.DataFrame(bias_matrix, index=available_experts, columns=levels)
    sns.heatmap(bias_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Error'})
    plt.title('Error Bias by Count Level')
    plt.ylabel('Experts')
    plt.xlabel('Count Level')

    # 7. Error correlation with count level
    plt.subplot(4, 4, 7)
    correlations = [expert_analysis[expert]['error_correlation'] for expert in available_experts]
    colors = ['red' if c > 0 else 'blue' for c in correlations]

    bars = plt.bar(available_experts, correlations, color=colors, alpha=0.7)
    plt.ylabel('Correlation (Error vs Count Level)')
    plt.title('Error-Count Correlation')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Add values on bars
    for bar, corr in zip(bars, correlations):
        y_pos = bar.get_height() + 0.02 if corr >= 0 else bar.get_height() - 0.05
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f'{corr:.2f}', ha='center', va='bottom' if corr >= 0 else 'top')

    # 8. Cumulative error plots
    plt.subplot(4, 4, 8)
    for expert in available_experts:
        errors = expert_analysis[expert]['image_errors']
        cumulative_error = np.cumsum(errors)
        plt.plot(range(len(cumulative_error)), cumulative_error, label=expert, linewidth=2)

    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Image Index')
    plt.ylabel('Cumulative Error')
    plt.title('Cumulative Error Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9-12. Individual expert error patterns
    for i, expert in enumerate(available_experts[:4]):
        plt.subplot(4, 4, 9 + i)

        consensus_vals = consensus_values
        expert_vals = expert_analysis[expert]['expert_values']
        errors = expert_analysis[expert]['image_errors']

        # Color points by error direction
        colors = ['red' if e > 0 else 'blue' if e < 0 else 'green' for e in errors]

        plt.scatter(consensus_vals, expert_vals, c=colors, alpha=0.6, s=30)

        max_val = max(np.max(consensus_vals), np.max(expert_vals))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

        plt.xlabel('Consensus Count')
        plt.ylabel(f'{expert} Count')
        plt.title(f'{expert} Error Pattern\nRed=Over, Blue=Under, Green=Exact')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_systematic_patterns(expert_analysis, consensus_values):
    """Analyze systematic patterns across all experts."""

    available_experts = list(expert_analysis.keys())

    if not available_experts:
        return

    print(f"Analysis across {len(available_experts)} experts:")

    # Overall compensation effectiveness
    avg_compensation = np.mean([expert_analysis[expert]['compensation_ratio'] for expert in available_experts])
    print(f"  Average compensation ratio: {avg_compensation:.3f}")

    # Total error vs individual error relationship
    total_errors = [abs(expert_analysis[expert]['total_error']) for expert in available_experts]
    maes = [expert_analysis[expert]['mae'] for expert in available_experts]

    if len(total_errors) > 1:
        error_correlation = np.corrcoef(total_errors, maes)[0, 1]
        print(f"  Correlation between total error and MAE: {error_correlation:.3f}")

    # Count level bias patterns
    print(f"\n  Error patterns by count level:")
    first_expert = available_experts[0]
    levels = list(expert_analysis[first_expert]['error_by_level'].keys())

    for level in levels:
        level_biases = []
        level_maes = []
        for expert in available_experts:
            if level in expert_analysis[expert]['error_by_level']:
                level_biases.append(expert_analysis[expert]['error_by_level'][level]['bias'])
                level_maes.append(expert_analysis[expert]['error_by_level'][level]['mae'])

        if level_biases:
            avg_bias = np.mean(level_biases)
            avg_mae = np.mean(level_maes)
            print(f"    Count level {level}: bias={avg_bias:+.2f}, MAE={avg_mae:.2f}")

    # Identify compensation strategies
    print(f"\n  Expert compensation strategies:")
    for expert in available_experts:
        data = expert_analysis[expert]

        if abs(data['total_error']) < 0.05 * np.sum(consensus_values):  # Within 5%
            strategy = "Excellent total accuracy"
        elif data['compensation_ratio'] > 0.8:
            strategy = "High compensation (balanced errors)"
        elif data['total_error'] > 0:
            strategy = "Systematic overcounting"
        else:
            strategy = "Systematic undercounting"

        print(f"    {expert}: {strategy} (comp ratio: {data['compensation_ratio']:.2f})")


# Function to identify problematic images
def identify_problematic_images(expert_analysis, consensus_values, all_images, threshold=2):
    """Identify images where most experts have high errors."""

    available_experts = list(expert_analysis.keys())
    n_experts = len(available_experts)

    if n_experts == 0:
        return []

    # Count high errors per image
    high_error_counts = np.zeros(len(all_images))

    for expert in available_experts:
        abs_errors = expert_analysis[expert]['abs_errors']
        high_error_mask = abs_errors >= threshold
        high_error_counts += high_error_mask

    # Find images where most experts have high errors
    problematic_threshold = n_experts * 0.7  # 70% of experts
    problematic_mask = high_error_counts >= problematic_threshold
    problematic_indices = np.where(problematic_mask)[0]

    problematic_images = []
    for idx in problematic_indices:
        image_info = {
            'image': all_images[idx],
            'consensus_count': consensus_values[idx],
            'experts_with_high_error': int(high_error_counts[idx]),
            'expert_errors': {}
        }

        for expert in available_experts:
            expert_count = expert_analysis[expert]['expert_values'][idx]
            error = expert_analysis[expert]['image_errors'][idx]
            image_info['expert_errors'][expert] = {
                'count': expert_count,
                'error': error
            }

        problematic_images.append(image_info)

    return problematic_images


def get_consensus_values(hA_flat, all_images, consensus_name="consensus"):
    """
    Extract consensus values from the hA_flat DataFrame.

    Parameters:
    - hA_flat: DataFrame with annotation data
    - all_images: List of all image names
    - consensus_name: Name of the consensus annotations

    Returns:
    - numpy array of consensus counts per image
    """
    consensus_data = hA_flat[hA_flat['class_name'] == consensus_name]
    consensus_counts = consensus_data.groupby('image_name').size()
    consensus_values = np.array([consensus_counts.get(img, 0) for img in all_images])
    return consensus_values


# Example usage function
def example_usage_with_existing_code():
    """Example of how to use this analysis with existing code."""

    print("=== USAGE EXAMPLE ===")
    print("""
    # After loading your data:

    hA = HastyAnnotationV2.from_file(hasty_path)
    hA_flat = hA.get_flat_df()

    hA_flat['class_name'] = hA_flat['class_name'].replace({
        'concenso': 'consensus',
        'conceso_parcial': 'consensus'
    })

    # Run the error compensation analysis:

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
    """)


if __name__ == "__main__":
    example_usage_with_existing_code()