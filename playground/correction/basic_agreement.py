
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def analyze_counts(df):
    """
    Simple count analysis for bounding box dataset
    """

    print("=== DATASET COUNTS ANALYSIS ===\n")

    # 1. Basic counts
    print("1. BASIC COUNTS")
    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {df['image_name'].nunique()}")
    print(f"Unique classes: {df['class_name'].nunique()}")
    print(f"Unique datasets: {df['dataset_name'].nunique()}")

    # 2. Class distribution
    print("\n2. CLASS DISTRIBUTION")
    class_counts = df['class_name'].value_counts()
    print(class_counts)
    print(f"\nClass percentages:")
    class_percentages = (class_counts / len(df) * 100).round(1)
    for class_name, percentage in class_percentages.items():
        print(f"{class_name}: {percentage}%")

    # 3. Dataset distribution
    print("\n3. DATASET DISTRIBUTION")
    dataset_counts = df['dataset_name'].value_counts()
    print(dataset_counts)

    # 4. Images with annotation counts
    print("\n4. ANNOTATIONS PER IMAGE")
    annotations_per_image = df.groupby('image_name').size()
    print(f"Mean annotations per image: {annotations_per_image.mean():.2f}")
    print(f"Median annotations per image: {annotations_per_image.median():.2f}")
    print(f"Max annotations in single image: {annotations_per_image.max()}")
    print(f"Min annotations in single image: {annotations_per_image.min()}")

    # Images with most annotations
    top_images = annotations_per_image.nlargest(5)
    print(f"\nTop 5 images with most annotations:")
    for image, count in top_images.items():
        print(f"{image}: {count} annotations")

    # 5. Class distribution per image
    print("\n5. CLASSES PER IMAGE")
    classes_per_image = df.groupby('image_name')['class_name'].nunique()
    print(f"Mean classes per image: {classes_per_image.mean():.2f}")
    print(f"Max classes in single image: {classes_per_image.max()}")

    # 6. Cross-tabulation
    print("\n6. CLASS vs DATASET CROSSTAB")
    crosstab = pd.crosstab(df['class_name'], df['dataset_name'], margins=True)
    print(crosstab)

    # 7. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Class distribution bar plot
    class_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Annotations per image histogram
    axes[0, 1].hist(annotations_per_image, bins=20, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Annotations per Image Distribution')
    axes[0, 1].set_xlabel('Number of Annotations')
    axes[0, 1].set_ylabel('Number of Images')

    # Classes per image histogram
    axes[1, 0].hist(classes_per_image, bins=10, color='salmon', alpha=0.7)
    axes[1, 0].set_title('Classes per Image Distribution')
    axes[1, 0].set_xlabel('Number of Classes')
    axes[1, 0].set_ylabel('Number of Images')

    # Class distribution pie chart
    axes[1, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Class Distribution (Pie Chart)')

    plt.tight_layout()
    plt.show()

    # Return summary
    summary = {
        'total_annotations': len(df),
        'unique_images': df['image_name'].nunique(),
        'unique_classes': df['class_name'].nunique(),
        'class_counts': class_counts.to_dict(),
        'mean_annotations_per_image': annotations_per_image.mean(),
        'mean_classes_per_image': classes_per_image.mean()
    }

    return summary


def analyze_count_dependent_bias(df):
    """
    Analyzes whether annotators show count-dependent bias
    (i.e., do they overcount more when there are more objects?)

    Parameters:
    df: DataFrame with columns 'image_name', 'class_name' (annotator name)
    """

    print("=== COUNT-DEPENDENT BIAS ANALYSIS ===\n")

    # Get counts per image per annotator
    counts_per_image = df.groupby(['image_name', 'class_name']).size().reset_index(name='count')

    # Pivot to have annotators as columns
    counts_pivot = counts_per_image.pivot(index='image_name',
                                          columns='class_name',
                                          values='count').fillna(0)

    # Extract consensus and annotators
    consensus_counts = counts_pivot['consensus']
    annotators = [col for col in counts_pivot.columns if col.startswith('Annotator')]

    # Filter to only images with consensus > 0 (labeled images)
    labeled_images = consensus_counts > 0
    consensus_counts_labeled = consensus_counts[labeled_images]

    # Calculate errors for each annotator
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, annotator in enumerate(annotators):
        annotator_counts = counts_pivot[annotator][labeled_images]
        errors = annotator_counts - consensus_counts_labeled

        # Calculate statistics
        correlation, p_value = stats.pearsonr(consensus_counts_labeled, errors)

        # Linear regression
        slope, intercept, r_value, p_val_reg, std_err = stats.linregress(
            consensus_counts_labeled, errors
        )

        # Absolute error correlation
        abs_errors = np.abs(errors)
        abs_correlation, abs_p_value = stats.pearsonr(consensus_counts_labeled, abs_errors)

        results[annotator] = {
            'correlation': correlation,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'abs_correlation': abs_correlation,
            'mean_error': errors.mean(),
            'std_error': errors.std()
        }

        # Print statistics
        print(f"\n{annotator}:")
        print(f"  Error-Count Correlation: {correlation:.3f} (p={p_value:.4f})")
        print(f"  |Error|-Count Correlation: {abs_correlation:.3f} (p={abs_p_value:.4f})")
        print(f"  Regression slope: {slope:.3f} (error per additional object)")
        print(f"  R²: {r_value ** 2:.3f}")
        print(f"  Mean error: {errors.mean():.2f} ± {errors.std():.2f}")

        if p_value < 0.05:
            print(f"  ⚠️  SIGNIFICANT count-dependent bias detected!")
            if slope > 0:
                print(f"     → Tends to add {slope:.2f} extra annotations per additional object")
            else:
                print(f"     → Tends to miss {abs(slope):.2f} annotations per additional object")
        else:
            print(f"  ✓ No significant count-dependent bias")

        # Scatter plot with regression line
        ax = axes[idx]
        ax.scatter(consensus_counts_labeled, errors, alpha=0.5, s=30)

        # Add regression line
        x_line = np.array([consensus_counts_labeled.min(), consensus_counts_labeled.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2,
                label=f'y = {slope:.3f}x + {intercept:.3f}')

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Consensus Count', fontsize=11)
        ax.set_ylabel('Error (Annotator - Consensus)', fontsize=11)
        ax.set_title(f'{annotator}\nr={correlation:.3f}, p={p_value:.4f}, R^2={r_value ** 2:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('count_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional analysis: binned error rates
    print("\n" + "=" * 60)
    print("BINNED ANALYSIS (Error Rate by Object Count)")
    print("=" * 60)

    bins = [0, 5, 10, 15, 20, 100]
    bin_labels = ['1-5', '6-10', '11-15', '16-20', '20+']

    for annotator in annotators:
        print(f"\n{annotator}:")
        annotator_counts = counts_pivot[annotator][labeled_images]
        errors = annotator_counts - consensus_counts_labeled

        # Bin the data
        consensus_binned = pd.cut(consensus_counts_labeled, bins=bins, labels=bin_labels)
        binned_data = pd.DataFrame({
            'bin': consensus_binned,
            'error': errors,
            'abs_error': np.abs(errors),
            'consensus': consensus_counts_labeled
        })

        # Calculate statistics per bin
        bin_stats = binned_data.groupby('bin').agg({
            'error': ['mean', 'std', 'count'],
            'abs_error': 'mean',
            'consensus': 'mean'
        }).round(2)

        print(bin_stats)

    # Create comparison plot of all annotators
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: All annotators on same plot
    for annotator in annotators:
        annotator_counts = counts_pivot[annotator][labeled_images]
        errors = annotator_counts - consensus_counts_labeled

        ax1.scatter(consensus_counts_labeled, errors, alpha=0.4, s=20, label=annotator)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Consensus Count', fontsize=12)
    ax1.set_ylabel('Error (Annotator - Consensus)', fontsize=12)
    ax1.set_title('All Annotators: Error vs Consensus Count', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plots by count bins
    all_binned_data = []
    for annotator in annotators:
        annotator_counts = counts_pivot[annotator][labeled_images]
        errors = annotator_counts - consensus_counts_labeled
        consensus_binned = pd.cut(consensus_counts_labeled, bins=bins, labels=bin_labels)

        temp_df = pd.DataFrame({
            'Annotator': annotator,
            'Bin': consensus_binned,
            'Error': errors
        })
        all_binned_data.append(temp_df)

    combined_binned = pd.concat(all_binned_data, ignore_index=True)
    sns.boxplot(data=combined_binned, x='Bin', y='Error', hue='Annotator', ax=ax2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Consensus Count Bin', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Error Distribution by Count Range', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('count_bias_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, counts_pivot


def count_user_annotations(hA_flat, username_1="Iguana_Andrea", username_2="concenso"):
    """
    Compare annotation counts between two specific users

    Parameters:
    - hA_flat: DataFrame with columns including 'class_name' (user names)
    - username_1: First user to compare (default: "Iguana_Andrea")
    - username_2: Second user to compare, typically consensus (default: "concenso")

    Returns:
    - Dictionary with comparison results
    """

    print(f"=== USER COMPARISON: {username_1} vs {username_2} ===\n")

    # Count annotations for both users
    user1_count = len(hA_flat[hA_flat['class_name'] == username_1])
    user2_count = len(hA_flat[hA_flat['class_name'] == username_2])
    total_annotations = len(hA_flat)

    # Check if users exist
    if user1_count == 0:
        print(f"Warning: No annotations found for '{username_1}'")
    if user2_count == 0:
        print(f"Warning: No annotations found for '{username_2}'")

    # Calculate percentages and differences
    user1_percentage = (user1_count / total_annotations * 100) if total_annotations > 0 else 0
    user2_percentage = (user2_count / total_annotations * 100) if total_annotations > 0 else 0

    difference = user1_count - user2_count
    ratio = user1_count / user2_count if user2_count > 0 else float('inf')

    # Print comparison results
    print(f"Total annotations in dataset: {total_annotations}")
    print(f"\n{username_1}:")
    print(f"  Count: {user1_count}")
    print(f"  Percentage: {user1_percentage:.1f}%")

    print(f"\n{username_2}:")
    print(f"  Count: {user2_count}")
    print(f"  Percentage: {user2_percentage:.1f}%")

    print(f"\nComparison:")
    print(f"  Difference ({username_1} - {username_2}): {difference:+d}")
    print(f"  Ratio ({username_1} / {username_2}): {ratio:.2f}x")

    if difference > 0:
        print(f"  → {username_1} has {difference} more annotations than {username_2}")
    elif difference < 0:
        print(f"  → {username_2} has {abs(difference)} more annotations than {username_1}")
    else:
        print(f"  → Both users have the same number of annotations")

    # Per-image comparison
    print(f"\n=== PER-IMAGE COMPARISON ===")

    # Count annotations per image for each user
    user1_per_image = hA_flat[hA_flat['class_name'] == username_1].groupby('image_name').size()
    user2_per_image = hA_flat[hA_flat['class_name'] == username_2].groupby('image_name').size()

    # Get all images that have annotations from either user
    all_images = set(user1_per_image.index) | set(user2_per_image.index)

    print(f"Images with annotations from {username_1}: {len(user1_per_image)}")
    print(f"Images with annotations from {username_2}: {len(user2_per_image)}")
    print(f"Images with annotations from both users: {len(set(user1_per_image.index) & set(user2_per_image.index))}")
    print(f"Total unique images: {len(all_images)}")

    # Create comparison visualization
    plt.figure(figsize=(12, 8))

    # Bar comparison
    plt.subplot(2, 2, 1)
    users = [username_1, username_2]
    counts = [user1_count, user2_count]
    colors = ['lightcoral', 'lightblue']

    bars = plt.bar(users, counts, color=colors)
    plt.title('Annotation Count Comparison')
    plt.ylabel('Number of Annotations')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                 str(count), ha='center', va='bottom')

    # Pie chart comparison
    plt.subplot(2, 2, 2)
    if user1_count > 0 or user2_count > 0:
        plt.pie([user1_count, user2_count], labels=users, autopct='%1.1f%%', colors=colors)
        plt.title('Annotation Distribution')

    # Per-image comparison (if data available)
    plt.subplot(2, 2, 3)
    if len(user1_per_image) > 0 and len(user2_per_image) > 0:
        common_images = set(user1_per_image.index) & set(user2_per_image.index)
        if common_images:
            common_user1 = [user1_per_image.get(img, 0) for img in common_images]
            common_user2 = [user2_per_image.get(img, 0) for img in common_images]

            plt.scatter(common_user1, common_user2, alpha=0.6)
            plt.plot([0, max(max(common_user1), max(common_user2))],
                     [0, max(max(common_user1), max(common_user2))], 'r--', alpha=0.5)
            plt.xlabel(f'{username_1} annotations per image')
            plt.ylabel(f'{username_2} annotations per image')
            plt.title('Per-Image Annotation Comparison')
        else:
            plt.text(0.5, 0.5, 'No common images', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Per-Image Annotation Comparison')

    # Show available users in dataset
    plt.subplot(2, 2, 4)
    all_user_counts = hA_flat['class_name'].value_counts()
    all_user_counts.plot(kind='bar', color='lightgreen')
    plt.title('All Users in Dataset')
    plt.ylabel('Annotation Count')
    plt.xticks(rotation=45)

    # Highlight the two users being compared
    user_positions = []
    for i, user in enumerate(all_user_counts.index):
        if user in [username_1, username_2]:
            user_positions.append(i)

    if user_positions:
        bars = plt.gca().patches
        for pos in user_positions:
            if pos < len(bars):
                bars[pos].set_color('orange')

    plt.tight_layout()
    plt.show()

    # Return summary
    return {
        'user1': username_1,
        'user2': username_2,
        'user1_count': user1_count,
        'user2_count': user2_count,
        'user1_percentage': user1_percentage,
        'user2_percentage': user2_percentage,
        'difference': difference,
        'ratio': ratio,
        'total_annotations': total_annotations,
        'common_images': len(set(user1_per_image.index) & set(user2_per_image.index)) if len(
            user1_per_image) > 0 and len(user2_per_image) > 0 else 0
    }


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compare_on_image_level(hA_flat, username_1="Iguana_Andrea", username_2="concenso"):
    """
    Compare the error rates on the image level between two users.

    Parameters:
    - hA_flat: DataFrame with columns including 'class_name' (user names) and 'image_name'
    - username_1: First user to compare
    - username_2: Second user to compare (typically consensus/reference)

    Returns:
    - Dictionary with MAE, RMSE and detailed comparison results
    """

    print(f"=== IMAGE-LEVEL COMPARISON: {username_1} vs {username_2} ===\n")

    # Get annotation counts per image for each user
    user1_counts = hA_flat[hA_flat['class_name'] == username_1].groupby('image_name').size()
    user2_counts = hA_flat[hA_flat['class_name'] == username_2].groupby('image_name').size()

    # Get all unique images
    all_images = set(hA_flat['image_name'].unique())

    # Create aligned arrays for comparison
    user1_values = []
    user2_values = []
    image_names = []

    for image in all_images:
        count1 = user1_counts.get(image, 0)
        count2 = user2_counts.get(image, 0)

        user1_values.append(count1)
        user2_values.append(count2)
        image_names.append(image)

    user1_values = np.array(user1_values)
    user2_values = np.array(user2_values)

    # Calculate error metrics
    mae = mean_absolute_error(user2_values, user1_values)  # user2 as reference
    mse = mean_squared_error(user2_values, user1_values)
    rmse = np.sqrt(mse)

    # Additional metrics
    differences = user1_values - user2_values
    abs_differences = np.abs(differences)

    # Statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    median_abs_diff = np.median(abs_differences)
    max_abs_diff = np.max(abs_differences)

    # Count different error types
    overcount_images = np.sum(differences > 0)  # user1 has more annotations
    undercount_images = np.sum(differences < 0)  # user1 has fewer annotations
    exact_match_images = np.sum(differences == 0)  # exact match

    # Print results
    print(f"Comparing annotation counts per image:")
    print(f"Total images analyzed: {len(all_images)}")
    print(f"\nError Metrics ({username_2} as reference):")
    print(f"  Mean Absolute Error (MAE): {mae:.3f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.3f}")
    print(f"  Mean Squared Error (MSE): {mse:.3f}")

    print(f"\nDifference Statistics ({username_1} - {username_2}):")
    print(f"  Mean difference: {mean_diff:.3f}")
    print(f"  Standard deviation: {std_diff:.3f}")
    print(f"  Median absolute difference: {median_abs_diff:.3f}")
    print(f"  Maximum absolute difference: {max_abs_diff}")

    print(f"\nImage-level Analysis:")
    print(
        f"  Images where {username_1} has MORE annotations: {overcount_images} ({overcount_images / len(all_images) * 100:.1f}%)")
    print(
        f"  Images where {username_1} has FEWER annotations: {undercount_images} ({undercount_images / len(all_images) * 100:.1f}%)")
    print(f"  Images with EXACT match: {exact_match_images} ({exact_match_images / len(all_images) * 100:.1f}%)")

    # Detailed breakdown by difference magnitude
    print(f"\nError Magnitude Breakdown:")
    for threshold in [0, 1, 2, 5]:
        count = np.sum(abs_differences <= threshold)
        print(f"  Images with |difference| ≤ {threshold}: {count} ({count / len(all_images) * 100:.1f}%)")

    # Find most problematic images
    print(f"\nMost Problematic Images (highest absolute differences):")
    problem_indices = np.argsort(abs_differences)[-5:][::-1]  # Top 5 worst
    for i, idx in enumerate(problem_indices, 1):
        if abs_differences[idx] > 0:
            print(
                f"  {i}. {image_names[idx]}: {username_1}={user1_values[idx]}, {username_2}={user2_values[idx]}, diff={differences[idx]:+d}")

    # Create detailed comparison DataFrame
    comparison_df = pd.DataFrame({
        'image_name': image_names,
        f'{username_1}_count': user1_values,
        f'{username_2}_count': user2_values,
        'difference': differences,
        'abs_difference': abs_differences
    })

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Scatter plot: User1 vs User2 counts
    axes[0, 0].scatter(user2_values, user1_values, alpha=0.6)
    max_val = max(np.max(user1_values), np.max(user2_values))
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect agreement')
    axes[0, 0].set_xlabel(f'{username_2} annotations per image')
    axes[0, 0].set_ylabel(f'{username_1} annotations per image')
    axes[0, 0].set_title('Per-Image Annotation Counts')
    axes[0, 0].legend()

    # 2. Difference histogram
    axes[0, 1].hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect agreement')
    axes[0, 1].set_xlabel(f'Difference ({username_1} - {username_2})')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].set_title('Distribution of Differences')
    axes[0, 1].legend()

    # 3. Absolute difference histogram
    axes[0, 2].hist(abs_differences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_xlabel('Absolute Difference')
    axes[0, 2].set_ylabel('Number of Images')
    axes[0, 2].set_title('Distribution of Absolute Differences')

    # 4. Box plot comparison
    box_data = [user1_values, user2_values]
    axes[1, 0].boxplot(box_data, labels=[username_1, username_2])
    axes[1, 0].set_ylabel('Annotations per Image')
    axes[1, 0].set_title('Annotation Count Distribution by User')

    # 5. Error magnitude breakdown
    error_ranges = ['0', '≤1', '≤2', '≤5', '>5']
    error_counts = [
        np.sum(abs_differences == 0),
        np.sum(abs_differences <= 1) - np.sum(abs_differences == 0),
        np.sum((abs_differences <= 2) & (abs_differences > 1)),
        np.sum((abs_differences <= 5) & (abs_differences > 2)),
        np.sum(abs_differences > 5)
    ]

    axes[1, 1].bar(error_ranges, error_counts, color='lightgreen', alpha=0.7)
    axes[1, 1].set_xlabel('Absolute Difference Range')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].set_title('Error Magnitude Distribution')

    # Add count labels on bars
    for i, count in enumerate(error_counts):
        axes[1, 1].text(i, count + max(error_counts) * 0.01, str(count),
                        ha='center', va='bottom')

    # 6. Cumulative error plot
    sorted_abs_diff = np.sort(abs_differences)
    cumulative_pct = np.arange(1, len(sorted_abs_diff) + 1) / len(sorted_abs_diff) * 100

    axes[1, 2].plot(sorted_abs_diff, cumulative_pct, linewidth=2)
    axes[1, 2].set_xlabel('Absolute Difference')
    axes[1, 2].set_ylabel('Cumulative Percentage of Images')
    axes[1, 2].set_title('Cumulative Error Distribution')
    axes[1, 2].grid(True, alpha=0.3)

    # Add some reference lines
    for pct in [50, 80, 95]:
        idx = int(len(sorted_abs_diff) * pct / 100) - 1
        if idx < len(sorted_abs_diff):
            axes[1, 2].axhline(y=pct, color='red', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(x=sorted_abs_diff[idx], color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Return comprehensive results
    results = {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'median_abs_difference': median_abs_diff,
        'max_abs_difference': max_abs_diff,
        'total_images': len(all_images),
        'overcount_images': overcount_images,
        'undercount_images': undercount_images,
        'exact_match_images': exact_match_images,
        'comparison_dataframe': comparison_df,
        'user1_values': user1_values,
        'user2_values': user2_values,
        'differences': differences,
        'abs_differences': abs_differences
    }

    return results


def visualize_multi_user_disagreement(hA_flat):
    """
    Visualize disagreement patterns among all users in the dataset

    Parameters:
    - hA_flat: DataFrame with annotation data

    Returns:
    - Dictionary with disagreement metrics and analysis
    """

    print("=== MULTI-USER DISAGREEMENT ANALYSIS ===\n")

    # Get all users
    all_users = sorted(hA_flat['class_name'].unique())
    n_users = len(all_users)

    print(f"Found {n_users} users: {all_users}")

    # Get annotation counts per image for each user
    user_counts_per_image = {}
    all_images = sorted(hA_flat['image_name'].unique())

    for user in all_users:
        user_data = hA_flat[hA_flat['class_name'] == user]
        counts = user_data.groupby('image_name').size()
        user_counts_per_image[user] = [counts.get(img, 0) for img in all_images]

    # Create matrix for easier analysis
    count_matrix = np.array([user_counts_per_image[user] for user in all_users])

    # Calculate disagreement metrics
    disagreement_metrics = calculate_disagreement_metrics(count_matrix, all_users, all_images)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))


    # 2. Pairwise MAE heatmap
    plt.subplot(3, 4, 2)
    mae_matrix = calculate_pairwise_mae(count_matrix, all_users)
    sns.heatmap(mae_matrix, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=all_users, yticklabels=all_users)
    plt.title('Pairwise Mean Absolute Error')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # 3. Agreement frequency matrix
    plt.subplot(3, 4, 3)
    agreement_matrix = calculate_agreement_matrix(count_matrix, all_users)
    sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=all_users, yticklabels=all_users)
    plt.title('Agreement Frequency (%)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # 4. Standard deviation per image (disagreement intensity)
    std_per_image = np.std(count_matrix, axis=0)




    # 9. Consensus vs Individual comparison
    plt.subplot(3, 4, 9)
    consensus_counts = np.round(np.mean(count_matrix, axis=0))
    for i, user in enumerate(all_users):
        deviations = np.abs(user_counts_per_image[user] - consensus_counts)
        plt.scatter([i] * len(deviations), deviations, alpha=0.6, s=30)
    plt.xticks(range(n_users), all_users, rotation=45)
    plt.ylabel('Deviation from Consensus')
    plt.title('Individual vs Consensus Deviation')

    # 10. Agreement patterns (how often users agree exactly)
    plt.subplot(3, 4, 10)
    perfect_agreement = np.sum(std_per_image == 0)
    partial_agreement = np.sum((std_per_image > 0) & (std_per_image <= 1))
    high_disagreement = np.sum(std_per_image > 1)

    overall_std = np.mean(std_per_image)
    max_disagreement = np.max(std_per_image)

    categories = ['Perfect\nAgreement', 'Minor\nDisagreement', 'Major\nDisagreement']
    counts = [perfect_agreement, partial_agreement, high_disagreement]
    colors = ['green', 'orange', 'red']

    bars = plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.ylabel('Number of Images')
    plt.title('Agreement Pattern Distribution')

    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom')

    # 11. Pairwise scatter plots matrix (simplified)
    plt.subplot(3, 4, 11)
    # Show most disagreeing pair
    max_mae_pair = find_most_disagreeing_pair(mae_matrix, all_users)
    user1_idx = all_users.index(max_mae_pair[0])
    user2_idx = all_users.index(max_mae_pair[1])

    plt.scatter(count_matrix[user1_idx], count_matrix[user2_idx], alpha=0.6)
    max_val = max(np.max(count_matrix[user1_idx]), np.max(count_matrix[user2_idx]))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    plt.xlabel(f'{max_mae_pair[0]} annotations')
    plt.ylabel(f'{max_mae_pair[1]} annotations')
    plt.title(f'Most Disagreeing Pair\n(MAE: {mae_matrix.loc[max_mae_pair[0], max_mae_pair[1]]:.2f})')

    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')


    summary_text = f"""DISAGREEMENT SUMMARY

Total Images: {len(all_images)}
Total Users: {n_users}

Agreement Patterns:
• Perfect agreement: {perfect_agreement} images ({perfect_agreement / len(all_images) * 100:.1f}%)
• Minor disagreement: {partial_agreement} images ({partial_agreement / len(all_images) * 100:.1f}%)
• Major disagreement: {high_disagreement} images ({high_disagreement / len(all_images) * 100:.1f}%)

Average disagreement: {overall_std:.2f}
Max disagreement: {max_disagreement}
Most disagreeing pair: {max_mae_pair[0]} vs {max_mae_pair[1]}

Interpretation:
Red = High disagreement
Orange = Moderate disagreement  
Green = Good agreement"""

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=9, fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    # Print detailed analysis
    print_detailed_analysis(disagreement_metrics, all_users, all_images, std_per_image)

    std_per_image = np.std(count_matrix, axis=0)
    mean_per_image = np.mean(count_matrix, axis=0)

    cv_per_image = std_per_image / mean_per_image  # Coefficient of variation
    range_per_image = np.max(count_matrix, axis=0) - np.min(count_matrix, axis=0)

    return {
        'users': all_users,
        'images': all_images,
        'count_matrix': count_matrix,
        'mae_matrix': mae_matrix,
        'agreement_matrix': agreement_matrix,
        'disagreement_metrics': disagreement_metrics,
        'std_per_image': std_per_image,
        'cv_per_image': cv_per_image,
        'range_per_image': range_per_image
    }


def calculate_disagreement_metrics(count_matrix, users, images):
    """Calculate various disagreement metrics"""
    metrics = {}

    # Overall disagreement (average standard deviation)
    metrics['overall_disagreement'] = np.mean(np.std(count_matrix, axis=0))

    # Image-wise metrics
    metrics['std_per_image'] = np.std(count_matrix, axis=0)
    metrics['range_per_image'] = np.max(count_matrix, axis=0) - np.min(count_matrix, axis=0)

    # User-wise metrics
    metrics['user_totals'] = np.sum(count_matrix, axis=1)
    metrics['user_means'] = np.mean(count_matrix, axis=1)

    return metrics


def calculate_pairwise_mae(count_matrix, users):
    """Calculate MAE between all pairs of users"""
    n_users = len(users)
    mae_matrix = pd.DataFrame(index=users, columns=users, dtype=float)

    for i in range(n_users):
        for j in range(n_users):
            if i == j:
                mae_matrix.iloc[i, j] = 0
            else:
                mae = mean_absolute_error(count_matrix[i], count_matrix[j])
                mae_matrix.iloc[i, j] = mae

    return mae_matrix


def calculate_agreement_matrix(count_matrix, users):
    """Calculate percentage of images where users agree exactly"""
    n_users = len(users)
    n_images = count_matrix.shape[1]
    agreement_matrix = pd.DataFrame(index=users, columns=users, dtype=float)

    for i in range(n_users):
        for j in range(n_users):
            if i == j:
                agreement_matrix.iloc[i, j] = 100.0
            else:
                agreements = np.sum(count_matrix[i] == count_matrix[j])
                agreement_matrix.iloc[i, j] = (agreements / n_images) * 100

    return agreement_matrix


def find_most_disagreeing_pair(mae_matrix, users):
    """Find the pair of users with highest MAE"""
    max_mae = 0
    max_pair = None

    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users):
            if i < j:  # Only check upper triangle
                mae = mae_matrix.loc[user1, user2]
                if mae > max_mae:
                    max_mae = mae
                    max_pair = (user1, user2)

    return max_pair


def print_detailed_analysis(metrics, users, images, std_per_image):
    """Print detailed disagreement analysis"""
    print(f"\n=== DETAILED DISAGREEMENT ANALYSIS ===")
    print(f"Overall disagreement score: {metrics['overall_disagreement']:.3f}")

    # Most problematic images
    worst_images = np.argsort(std_per_image)[-3:][::-1]
    print(f"\nMost disagreed-upon images:")
    for i, img_idx in enumerate(worst_images, 1):
        if std_per_image[img_idx] > 0:
            print(f"  {i}. {images[img_idx]}: std={std_per_image[img_idx]:.2f}")

    # User annotation patterns
    print(f"\nUser annotation patterns:")
    for i, user in enumerate(users):
        total = metrics['user_totals'][i]
        mean_per_img = metrics['user_means'][i]
        print(f"  {user}: {total} total annotations, {mean_per_img:.1f} avg per image")


def plot_all_users_vs_consensus(hA_flat, consensus_name="consensu"):
    """
    Plot all users against the consensus user

    Parameters:
    - hA_flat: DataFrame with annotation data
    - consensus_name: Name of the consensus user (default: "consensu")

    Returns:
    - Dictionary with comparison metrics for each user
    """

    print(f"=== ALL USERS vs {consensus_name.upper()} ===\n")

    # Get all users except consensus
    all_users = sorted(hA_flat['class_name'].unique())
    other_users = [user for user in all_users if user != consensus_name]

    if consensus_name not in all_users:
        print(f"Warning: Consensus user '{consensus_name}' not found in dataset!")
        print(f"Available users: {all_users}")
        return None

    print(f"Consensus user: {consensus_name}")
    print(f"Other users: {other_users}")

    # Get all unique images
    all_images = sorted(hA_flat['image_name'].unique())

    # Get consensus counts per image
    consensus_data = hA_flat[hA_flat['class_name'] == consensus_name]
    consensus_counts = consensus_data.groupby('image_name').size()
    consensus_values = [consensus_counts.get(img, 0) for img in all_images]

    # Get counts for all other users
    user_data = {}
    user_metrics = {}

    for user in other_users:
        user_df = hA_flat[hA_flat['class_name'] == user]
        user_counts = user_df.groupby('image_name').size()
        user_values = [user_counts.get(img, 0) for img in all_images]

        # Calculate metrics
        mae = mean_absolute_error(consensus_values, user_values)
        rmse = np.sqrt(mean_squared_error(consensus_values, user_values))

        differences = np.array(user_values) - np.array(consensus_values)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)

        # Store data
        user_data[user] = {
            'values': user_values,
            'differences': differences,
            'abs_differences': np.abs(differences)
        }

        user_metrics[user] = {
            'mae': mae,
            'rmse': rmse,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'total_annotations': sum(user_values),
            'perfect_matches': np.sum(differences == 0),
            'overcount': np.sum(differences > 0),
            'undercount': np.sum(differences < 0)
        }

    # Print summary metrics
    print(f"\nMETRICS SUMMARY:")
    print(f"{'User':<15} {'MAE':<8} {'RMSE':<8} {'Mean Diff':<10} {'Perfect Match':<12} {'Total Ann.':<10}")
    print("-" * 70)

    for user in other_users:
        metrics = user_metrics[user]
        perfect_pct = metrics['perfect_matches'] / len(all_images) * 100
        print(f"{user:<15} {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} {metrics['mean_diff']:<10.2f} "
              f"{metrics['perfect_matches']}/{len(all_images)} ({perfect_pct:.1f}%) {metrics['total_annotations']:<10}")

    # Create comprehensive visualization
    n_users = len(other_users)
    fig = plt.figure(figsize=(20, 15))

    # 1. Scatter plots: Each user vs consensus (2x2 grid)
    for i, user in enumerate(other_users):
        plt.subplot(3, 4, i + 1)

        user_vals = user_data[user]['values']
        plt.scatter(consensus_values, user_vals, alpha=0.6, s=50)

        # Perfect agreement line
        max_val = max(max(consensus_values), max(user_vals))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect agreement')

        plt.xlabel(f'{consensus_name} annotations')
        plt.ylabel(f'{user} annotations')
        plt.title(
            f'{user} vs {consensus_name}\nMAE: {user_metrics[user]["mae"]:.2f}, RMSE: {user_metrics[user]["rmse"]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 5. Combined scatter plot (all users)
    plt.subplot(3, 4, 5)
    colors = ['blue', 'green', 'orange', 'purple']

    for i, user in enumerate(other_users):
        user_vals = user_data[user]['values']
        plt.scatter(consensus_values, user_vals, alpha=0.6,
                    label=f'{user} (MAE: {user_metrics[user]["mae"]:.2f})',
                    color=colors[i % len(colors)], s=40)

    max_val = max([max(consensus_values)] + [max(user_data[user]['values']) for user in other_users])
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect agreement')

    plt.xlabel(f'{consensus_name} annotations')
    plt.ylabel('User annotations')
    plt.title('All Users vs Consensus')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 6. MAE comparison bar chart
    plt.subplot(3, 4, 6)
    mae_values = [user_metrics[user]['mae'] for user in other_users]
    bars = plt.bar(other_users, mae_values, color='lightcoral', alpha=0.7)
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE vs Consensus')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, mae in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_values) * 0.01,
                 f'{mae:.2f}', ha='center', va='bottom')

    # 7. RMSE comparison bar chart
    plt.subplot(3, 4, 7)
    rmse_values = [user_metrics[user]['rmse'] for user in other_users]
    bars = plt.bar(other_users, rmse_values, color='lightblue', alpha=0.7)
    plt.ylabel('Root Mean Square Error')
    plt.title('RMSE vs Consensus')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, rmse in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_values) * 0.01,
                 f'{rmse:.2f}', ha='center', va='bottom')

    # 8. Perfect match percentages
    plt.subplot(3, 4, 8)
    perfect_pcts = [user_metrics[user]['perfect_matches'] / len(all_images) * 100 for user in other_users]
    bars = plt.bar(other_users, perfect_pcts, color='lightgreen', alpha=0.7)
    plt.ylabel('Perfect Match Percentage')
    plt.title('Perfect Agreement with Consensus')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, pct in zip(bars, perfect_pcts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(perfect_pcts) * 0.01,
                 f'{pct:.1f}%', ha='center', va='bottom')

    # 9. Difference distributions (box plot)
    plt.subplot(3, 4, 9)
    diff_data = [user_data[user]['differences'] for user in other_users]
    box_plot = plt.boxplot(diff_data, labels=other_users, patch_artist=True)

    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Perfect agreement')
    plt.ylabel(f'Difference (User - {consensus_name})')
    plt.title('Difference Distributions')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 10. Total annotations comparison
    plt.subplot(3, 4, 10)
    consensus_total = sum(consensus_values)
    user_totals = [user_metrics[user]['total_annotations'] for user in other_users]

    # Add consensus as reference
    all_totals = [consensus_total] + user_totals
    all_labels = [consensus_name] + other_users
    bar_colors = ['red'] + colors[:len(other_users)]

    bars = plt.bar(all_labels, all_totals, color=bar_colors, alpha=0.7)
    plt.ylabel('Total Annotations')
    plt.title('Total Annotation Counts')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, total in zip(bars, all_totals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(all_totals) * 0.01,
                 str(total), ha='center', va='bottom')

    # 11. Error patterns heatmap
    plt.subplot(3, 4, 11)

    # Create error pattern matrix
    error_patterns = []
    for user in other_users:
        overcount = user_metrics[user]['overcount']
        undercount = user_metrics[user]['undercount']
        perfect = user_metrics[user]['perfect_matches']
        error_patterns.append([overcount, perfect, undercount])

    error_df = pd.DataFrame(error_patterns,
                            columns=['Overcount', 'Perfect', 'Undercount'],
                            index=other_users)

    sns.heatmap(error_df, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Number of Images'})
    plt.title('Error Pattern Heatmap')
    plt.xticks(rotation=45)
    plt.ylabel('Users')

    # 12. Summary statistics text
    plt.subplot(3, 4, 12)
    plt.axis('off')

    # Find best and worst users
    best_user = min(other_users, key=lambda u: user_metrics[u]['mae'])
    worst_user = max(other_users, key=lambda u: user_metrics[u]['mae'])

    summary_text = f"""SUMMARY vs {consensus_name}

Total Images: {len(all_images)}
Consensus Annotations: {consensus_total}

BEST AGREEMENT:
{best_user}
• MAE: {user_metrics[best_user]['mae']:.2f}
• Perfect matches: {user_metrics[best_user]['perfect_matches']}/{len(all_images)}

WORST AGREEMENT:
{worst_user}
• MAE: {user_metrics[worst_user]['mae']:.2f}
• Perfect matches: {user_metrics[worst_user]['perfect_matches']}/{len(all_images)}

RANKING (by MAE):"""

    # Add ranking
    ranked_users = sorted(other_users, key=lambda u: user_metrics[u]['mae'])
    for i, user in enumerate(ranked_users, 1):
        summary_text += f"\n{i}. {user}: {user_metrics[user]['mae']:.2f}"

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10, fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    return {
        'consensus_user': consensus_name,
        'other_users': other_users,
        'consensus_values': consensus_values,
        'user_data': user_data,
        'user_metrics': user_metrics,
        'images': all_images,
        'best_user': best_user,
        'worst_user': worst_user
    }


def analyse_normal_distribution_expert(hA_flat: pd.DataFrame,
                                       username_1="Iguana_Andrea",
                                       username_2="consensus"):
    """
    Analyze the normal distribution of annotations by an expert user compared to a consensus.
    :param hA_flat: DataFrame with annotation data
    :param username_1: Expert user to analyze
    :param username_2: Consensus user to compare against
    :return: Dictionary with analysis results
    """

    print(f"=== NORMAL DISTRIBUTION ANALYSIS: {username_1} vs {username_2} ===\n")

    # Check if users exist in the dataset
    if username_1 not in hA_flat['class_name'].unique():
        print(f"Warning: User '{username_1}' not found in dataset!")
        return None

    if username_2 not in hA_flat['class_name'].unique():
        print(f"Warning: User '{username_2}' not found in dataset!")
        return None

    # Get all unique images
    all_images = sorted(hA_flat['image_name'].unique())

    # Calculate the deviations from the consensus
    user1_counts = hA_flat[hA_flat['class_name'] == username_1].groupby('image_name').size()
    user2_counts = hA_flat[hA_flat['class_name'] == username_2].groupby('image_name').size()

    # Create aligned arrays for comparison
    user1_values = []
    user2_values = []
    image_names = []

    for image in all_images:
        count1 = user1_counts.get(image, 0)
        count2 = user2_counts.get(image, 0)

        user1_values.append(count1)
        user2_values.append(count2)
        image_names.append(image)

    user1_values = np.array(user1_values)
    user2_values = np.array(user2_values)

    # Calculate deviations
    deviations = user1_values - user2_values
    abs_deviations = np.abs(deviations)

    # Basic statistics
    mean_deviation = np.mean(deviations)
    std_deviation = np.std(deviations)
    median_deviation = np.median(deviations)

    print(f"Deviation Statistics ({username_1} - {username_2}):")
    print(f"  Mean deviation: {mean_deviation:.3f}")
    print(f"  Standard deviation: {std_deviation:.3f}")
    print(f"  Median deviation: {median_deviation:.3f}")
    print(f"  Min deviation: {np.min(deviations)}")
    print(f"  Max deviation: {np.max(deviations)}")

    # Plot the deviations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of deviations
    axes[0, 0].hist(deviations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No deviation')
    axes[0, 0].axvline(x=mean_deviation, color='green', linestyle='-', alpha=0.7, label=f'Mean: {mean_deviation:.2f}')
    axes[0, 0].set_xlabel(f'Deviation ({username_1} - {username_2})')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Deviations')
    axes[0, 0].legend()

    # 2. Q-Q plot for visual normality check
    from scipy import stats

    # Calculate z-scores for the deviations
    z_scores = (deviations - mean_deviation) / std_deviation

    # Create Q-Q plot
    stats.probplot(deviations, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normal Distribution Check)')

    # 3. Scatter plot: Expert vs Consensus
    axes[1, 0].scatter(user2_values, user1_values, alpha=0.6)
    max_val = max(np.max(user1_values), np.max(user2_values))
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect agreement')
    axes[1, 0].set_xlabel(f'{username_2} annotations per image')
    axes[1, 0].set_ylabel(f'{username_1} annotations per image')
    axes[1, 0].set_title('Expert vs Consensus Annotations')
    axes[1, 0].legend()

    # 4. Deviation vs Consensus count
    axes[1, 1].scatter(user2_values, deviations, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No deviation')
    axes[1, 1].set_xlabel(f'{username_2} annotations per image')
    axes[1, 1].set_ylabel(f'Deviation ({username_1} - {username_2})')
    axes[1, 1].set_title('Deviation vs Consensus Count')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Test for normal distribution of the deviations
    print("\nNormal Distribution Tests:")

    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(deviations)
    print(f"Shapiro-Wilk Test:")
    print(f"  W-statistic: {shapiro_test.statistic:.4f}")
    print(f"  p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("  Result: Deviations are NOT normally distributed (p < 0.05)")
    else:
        print("  Result: Deviations appear to be normally distributed (p >= 0.05)")

    # DAgostino K^2 test # https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html#hypothesis-normaltest
    k2_test = stats.normaltest(deviations)
    print(f"\nD'Agostino's K^2 Test:")
    print(f"  K^2-statistic: {k2_test.statistic:.4f}")
    print(f"  p-value: {k2_test.pvalue:.4f}")
    if k2_test.pvalue < 0.05:
        print("  Result: Deviations are NOT normally distributed (p < 0.05)")
    else:
        print("  Result: Deviations appear to be normally distributed (p >= 0.05)")

    # Return results
    return {
        'user1': username_1,
        'user2': username_2,
        'deviations': deviations,
        'mean_deviation': mean_deviation,
        'std_deviation': std_deviation,
        'median_deviation': median_deviation,
        'shapiro_test': shapiro_test,
        'k2_test': k2_test,
        'is_normal_shapiro': shapiro_test.pvalue >= 0.05,
        'is_normal_k2': k2_test.pvalue >= 0.05
    }, fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def filter_by_agreement(hA_flat, expert_names=["Iguana_Andrea", "Iguana_Andres", "Iguana_Amy", "Iguana_Robin"],
                        agreement_threshold=3, consensus_name="consensus",
                        agreement_type="exact", tolerance=0, image_column="images"):
    """
    Filter out images where a specified number of experts agree.

    Parameters:
    - hA_flat: DataFrame with annotation data
    - expert_names: List of expert names to consider
    - agreement_threshold: Minimum number of experts that must agree (default: 3 out of 4)
    - consensus_name: Name of consensus annotations (optional, for comparison)
    - agreement_type: Type of agreement to check
        - "exact": Experts must have exactly the same count
        - "tolerance": Experts must be within tolerance range
        - "majority": Most common count among experts
    - tolerance: Allowed difference for "tolerance" agreement type

    Returns:
    - Dictionary with filtered data and analysis
    """

    print(f"=== AGREEMENT-BASED FILTERING ===")
    print(f"Filtering images where {agreement_threshold} out of {len(expert_names)} experts agree")
    print(f"Agreement type: {agreement_type}")
    if agreement_type == "tolerance":
        print(f"Tolerance: ±{tolerance}")

    # Get all unique images
    all_images = sorted(hA_flat[image_column].unique())

    # Get expert counts for each image
    expert_counts = {}
    for expert in expert_names:
        if expert in hA_flat['class_name'].unique():
            expert_data = hA_flat[hA_flat['class_name'] == expert]
            expert_count_series = expert_data.groupby(image_column).size()
            expert_counts[expert] = [expert_count_series.get(img, 0) for img in all_images]
        else:
            print(f"Warning: Expert '{expert}' not found in dataset!")
            expert_counts[expert] = [0] * len(all_images)

    # Get consensus counts if available
    consensus_counts = None
    if consensus_name in hA_flat['class_name'].unique():
        consensus_data = hA_flat[hA_flat['class_name'] == consensus_name]
        consensus_count_series = consensus_data.groupby(image_column).size()
        consensus_counts = [consensus_count_series.get(img, 0) for img in all_images]

    # Analyze agreement for each image
    agreement_analysis = analyze_image_agreements(
        all_images, expert_counts, agreement_type, tolerance, agreement_threshold
    )

    # Filter images based on agreement
    filtered_results = apply_agreement_filter(
        hA_flat, all_images, agreement_analysis, agreement_threshold, consensus_counts
    )

    # Create visualization
    visualize_agreement_filtering(
        agreement_analysis, expert_counts, consensus_counts, all_images, expert_names
    )

    return filtered_results


def analyze_image_agreements(all_images, expert_counts, agreement_type, tolerance, agreement_threshold):
    """Analyze agreement patterns for each image."""

    expert_names = list(expert_counts.keys())
    n_experts = len(expert_names)

    agreement_analysis = {
        'images': all_images,
        'expert_counts_matrix': [],
        'agreement_counts': [],
        'agreement_levels': [],
        'most_common_counts': [],
        'agreement_details': [],
        'meets_threshold': []
    }

    for i, image in enumerate(all_images):
        # Get counts for this image from all experts
        image_counts = [expert_counts[expert][i] for expert in expert_names]
        agreement_analysis['expert_counts_matrix'].append(image_counts)

        # Analyze agreement
        if agreement_type == "exact":
            agreement_info = analyze_exact_agreement(image_counts, agreement_threshold)
        elif agreement_type == "tolerance":
            agreement_info = analyze_tolerance_agreement(image_counts, tolerance, agreement_threshold)
        elif agreement_type == "majority":
            agreement_info = analyze_majority_agreement(image_counts, agreement_threshold)
        else:
            raise ValueError(f"Unknown agreement type: {agreement_type}")

        agreement_analysis['agreement_counts'].append(agreement_info['agreement_count'])
        agreement_analysis['agreement_levels'].append(agreement_info['agreement_level'])
        agreement_analysis['most_common_counts'].append(agreement_info['most_common_count'])
        agreement_analysis['agreement_details'].append(agreement_info['details'])
        agreement_analysis['meets_threshold'].append(agreement_info['meets_threshold'])

    return agreement_analysis


def analyze_exact_agreement(image_counts, agreement_threshold):
    """Analyze exact agreement (all counts must be identical)."""

    count_frequencies = Counter(image_counts)
    most_common_count, max_frequency = count_frequencies.most_common(1)[0]

    agreement_info = {
        'agreement_count': max_frequency,
        'agreement_level': max_frequency / len(image_counts),
        'most_common_count': most_common_count,
        'meets_threshold': max_frequency >= agreement_threshold,
        'details': {
            'count_distribution': dict(count_frequencies),
            'agreeing_experts': max_frequency,
            'agreement_value': most_common_count
        }
    }

    return agreement_info


def analyze_tolerance_agreement(image_counts, tolerance, agreement_threshold):
    """Analyze tolerance-based agreement (counts within tolerance range)."""

    max_agreement = 0
    best_center = None
    best_agreeing_counts = []

    # Try each unique count as a potential center
    unique_counts = set(image_counts)

    for center in unique_counts:
        agreeing_counts = [count for count in image_counts
                           if abs(count - center) <= tolerance]

        if len(agreeing_counts) > max_agreement:
            max_agreement = len(agreeing_counts)
            best_center = center
            best_agreeing_counts = agreeing_counts

    agreement_info = {
        'agreement_count': max_agreement,
        'agreement_level': max_agreement / len(image_counts),
        'most_common_count': best_center,
        'meets_threshold': max_agreement >= agreement_threshold,
        'details': {
            'center_value': best_center,
            'tolerance': tolerance,
            'agreeing_counts': best_agreeing_counts,
            'all_counts': image_counts
        }
    }

    return agreement_info


def analyze_majority_agreement(image_counts, agreement_threshold):
    """Analyze majority agreement (most frequent count)."""

    count_frequencies = Counter(image_counts)
    most_common_count, max_frequency = count_frequencies.most_common(1)[0]

    # For majority, we also consider if it's truly a majority (>50%)
    is_majority = max_frequency > len(image_counts) / 2

    agreement_info = {
        'agreement_count': max_frequency,
        'agreement_level': max_frequency / len(image_counts),
        'most_common_count': most_common_count,
        'meets_threshold': max_frequency >= agreement_threshold,
        'details': {
            'count_distribution': dict(count_frequencies),
            'is_true_majority': is_majority,
            'majority_count': most_common_count
        }
    }

    return agreement_info


def apply_agreement_filter(hA_flat, all_images, agreement_analysis, agreement_threshold, consensus_counts):
    """Apply the agreement filter and return results."""

    # Identify images to keep (those that DON'T meet agreement threshold)
    images_to_keep = []
    images_to_remove = []

    for i, image in enumerate(all_images):
        if agreement_analysis['meets_threshold'][i]:
            images_to_remove.append({
                'image': image,
                'agreement_count': agreement_analysis['agreement_counts'][i],
                'agreement_level': agreement_analysis['agreement_levels'][i],
                'most_common_count': agreement_analysis['most_common_counts'][i],
                'expert_counts': agreement_analysis['expert_counts_matrix'][i],
                'consensus_count': consensus_counts[i] if consensus_counts else None
            })
        else:
            images_to_keep.append(image)

    # Filter the original DataFrame
    filtered_hA_flat = hA_flat[hA_flat['image_name'].isin(images_to_keep)].copy()

    # Print summary
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Original images: {len(all_images)}")
    print(f"Images with high agreement (removed): {len(images_to_remove)}")
    print(f"Images with low agreement (kept): {len(images_to_keep)}")
    print(f"Filtering rate: {len(images_to_remove) / len(all_images) * 100:.1f}%")

    # Show some examples of removed images
    print(f"\nExamples of removed images (high agreement):")
    for i, img_info in enumerate(images_to_remove[:10]):
        consensus_str = f", consensus: {img_info['consensus_count']}" if img_info['consensus_count'] is not None else ""
        print(
            f"  {i + 1}. {img_info['image']}: {img_info['agreement_count']}/{len(agreement_analysis['expert_counts_matrix'][0])} experts agree on {img_info['most_common_count']}{consensus_str}")
        print(f"      Expert counts: {img_info['expert_counts']}")

    return {
        'filtered_data': filtered_hA_flat,
        'kept_images': images_to_keep,
        'removed_images': images_to_remove,
        'agreement_analysis': agreement_analysis,
        'summary': {
            'original_count': len(all_images),
            'kept_count': len(images_to_keep),
            'removed_count': len(images_to_remove),
            'filtering_rate': len(images_to_remove) / len(all_images) * 100
        }
    }


def visualize_agreement_filtering(agreement_analysis, expert_counts, consensus_counts, all_images, expert_names):
    """Create visualizations for the agreement filtering process."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Agreement level distribution
    axes[0, 0].hist(agreement_analysis['agreement_levels'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Agreement Level (Fraction of Experts)')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].set_title('Distribution of Agreement Levels')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Agreement count distribution
    axes[0, 1].hist(agreement_analysis['agreement_counts'], bins=range(1, len(expert_names) + 2),
                    alpha=0.7, color='lightcoral', edgecolor='black', align='left')
    axes[0, 1].set_xlabel('Number of Agreeing Experts')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].set_title('Distribution of Expert Agreement Counts')
    axes[0, 1].set_xticks(range(1, len(expert_names) + 1))
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Most common count distribution
    axes[0, 2].hist(agreement_analysis['most_common_counts'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Most Common Count Value')
    axes[0, 2].set_ylabel('Number of Images')
    axes[0, 2].set_title('Distribution of Agreed-Upon Counts')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Agreement vs consensus (if available)
    if consensus_counts:
        axes[1, 0].scatter(consensus_counts, agreement_analysis['most_common_counts'],
                           c=agreement_analysis['agreement_levels'], cmap='viridis', alpha=0.6)

        max_val = max(max(consensus_counts), max(agreement_analysis['most_common_counts']))
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        axes[1, 0].set_xlabel('Consensus Count')
        axes[1, 0].set_ylabel('Expert Agreed Count')
        axes[1, 0].set_title('Expert Agreement vs Consensus')

        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('Agreement Level')
    else:
        axes[1, 0].text(0.5, 0.5, 'No consensus data available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Expert Agreement vs Consensus')

    # 5. Agreement level by count value
    axes[1, 1].scatter(agreement_analysis['most_common_counts'], agreement_analysis['agreement_levels'],
                       alpha=0.6, s=30)
    axes[1, 1].set_xlabel('Count Value')
    axes[1, 1].set_ylabel('Agreement Level')
    axes[1, 1].set_title('Agreement Level vs Count Value')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Filtering threshold visualization
    axes[1, 2].bar(['Kept Images', 'Removed Images'],
                   [sum(not x for x in agreement_analysis['meets_threshold']),
                    sum(agreement_analysis['meets_threshold'])],
                   color=['green', 'red'], alpha=0.7)
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].set_title('Filtering Results')

    # Add value labels on bars
    kept_count = sum(not x for x in agreement_analysis['meets_threshold'])
    removed_count = sum(agreement_analysis['meets_threshold'])
    axes[1, 2].text(0, kept_count + max(kept_count, removed_count) * 0.02, str(kept_count),
                    ha='center', va='bottom')
    axes[1, 2].text(1, removed_count + max(kept_count, removed_count) * 0.02, str(removed_count),
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def analyze_filtered_vs_original(original_results, filtered_results, expert_names):
    """Compare analysis results before and after filtering."""

    print(f"\n=== COMPARISON: ORIGINAL vs FILTERED DATA ===")

    # You would need to run your correction factor analysis on both datasets
    # This is a placeholder for the comparison structure

    print(f"Original dataset:")
    print(f"  Images: {original_results['summary']['original_count']}")

    print(f"Filtered dataset:")
    print(f"  Images: {filtered_results['summary']['kept_count']}")
    print(f"  Reduction: {filtered_results['summary']['filtering_rate']:.1f}%")

    print(f"\nFiltered out (high agreement images):")
    print(f"  Count: {filtered_results['summary']['removed_count']}")

    # Here you could add comparison of MAE, RMSE, etc. between datasets







