import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads

# Load the data (assuming you have them as CSV files)
# Replace these with your actual file paths
ground_truth = pd.read_csv('ground_truth.csv')
predictions = pd.read_csv('predictions.csv')

# Convert geometry columns from WKT strings to geometry objects if needed
if isinstance(ground_truth['geometry'].iloc[0], str):
    ground_truth['geometry'] = ground_truth['geometry'].apply(loads)
if isinstance(predictions['geometry'].iloc[0], str):
    predictions['geometry'] = predictions['geometry'].apply(loads)

# Count ground truth points per tile
gt_counts = ground_truth.groupby('tile_name').size().reset_index(name='ground_truth_count')

# Count prediction points per tile (using tile_name_right which matches tile_name in ground truth)
pred_counts = predictions.groupby('tile_name_right').size().reset_index(name='prediction_count')
pred_counts.rename(columns={'tile_name_right': 'tile_name'}, inplace=True)

# Merge the counts
comparison = pd.merge(gt_counts, pred_counts, on='tile_name', how='outer')

# Fill NaN values with 0 (tiles with no detections in either dataset)
comparison = comparison.fillna(0)

# Convert counts to integers
comparison['ground_truth_count'] = comparison['ground_truth_count'].astype(int)
comparison['prediction_count'] = comparison['prediction_count'].astype(int)

# Calculate the difference (predictions - ground truth)
comparison['difference'] = comparison['prediction_count'] - comparison['ground_truth_count']

# Calculate absolute difference for error metrics
comparison['absolute_difference'] = abs(comparison['difference'])

# Sort by tile name for better readability
comparison = comparison.sort_values('tile_name')

# Display the results
print("Tile-by-Tile Comparison:")
print("=" * 80)
for idx, row in comparison.iterrows():
    tile = row['tile_name']
    gt = int(row['ground_truth_count'])
    pred = int(row['prediction_count'])
    diff = int(row['difference'])

    status = "✓ Perfect" if diff == 0 else f"{'Over' if diff > 0 else 'Under'}-detected by {abs(diff)}"
    print(f"{tile}:")
    print(f"  Ground Truth: {gt} | Predictions: {pred} | Difference: {diff:+d} | {status}")
    print()

# Summary statistics
print("\nSummary Statistics:")
print("=" * 80)
print(f"Total tiles analyzed: {len(comparison)}")
print(f"Tiles with perfect match: {(comparison['difference'] == 0).sum()}")
print(f"Tiles with over-detection: {(comparison['difference'] > 0).sum()}")
print(f"Tiles with under-detection: {(comparison['difference'] < 0).sum()}")
print(f"\nTotal ground truth points: {comparison['ground_truth_count'].sum()}")
print(f"Total predicted points: {comparison['prediction_count'].sum()}")
print(f"Overall difference: {comparison['difference'].sum():+d}")
print(f"Mean absolute error per tile: {comparison['absolute_difference'].mean():.2f}")


# Optional: Calculate spatial distances between matched points
# This requires more complex matching logic
def calculate_spatial_differences(ground_truth, predictions):
    """
    Calculate nearest neighbor distances between ground truth and predicted points
    for each tile.
    """
    from scipy.spatial.distance import cdist
    import numpy as np

    results = []

    # Get unique tiles
    tiles = set(ground_truth['tile_name'].unique()) | set(predictions['tile_name_right'].unique())

    for tile in tiles:
        # Get points for this tile
        gt_tile = ground_truth[ground_truth['tile_name'] == tile]
        pred_tile = predictions[predictions['tile_name_right'] == tile]

        if len(gt_tile) == 0 or len(pred_tile) == 0:
            continue

        # Extract coordinates
        gt_coords = [[p.x, p.y] for p in gt_tile['geometry']]
        pred_coords = [[p.x, p.y] for p in pred_tile['geometry']]

        # Calculate distance matrix
        distances = cdist(gt_coords, pred_coords)

        # Find nearest prediction for each ground truth point
        min_distances = distances.min(axis=1)

        results.append({
            'tile_name': tile,
            'mean_distance': min_distances.mean(),
            'max_distance': min_distances.max(),
            'min_distance': min_distances.min(),
            'num_gt_points': len(gt_tile),
            'num_pred_points': len(pred_tile)
        })

    return pd.DataFrame(results)


# Calculate spatial differences
print("\nSpatial Distance Analysis (Ground Truth to Nearest Prediction):")
print("=" * 80)
spatial_df = calculate_spatial_differences(ground_truth, predictions)

if not spatial_df.empty:
    for idx, row in spatial_df.iterrows():
        print(f"{row['tile_name']}:")
        print(f"  Mean distance: {row['mean_distance']:.2f} units")
        print(f"  Distance range: {row['min_distance']:.2f} - {row['max_distance']:.2f} units")
        print()

    print(f"\nOverall mean distance: {spatial_df['mean_distance'].mean():.2f} units")

# Save results to CSV
comparison.to_csv('tile_comparison_results.csv', index=False)
print("\nResults saved to 'tile_comparison_results.csv'")


def create_detailed_comparison(ground_truth, predictions):
    """
    Create a detailed comparison including all point coordinates
    """
    detailed = []

    tiles = set(ground_truth['tile_name'].unique()) | set(predictions['tile_name_right'].unique())

    for tile in tiles:
        gt_tile = ground_truth[ground_truth['tile_name'] == tile]
        pred_tile = predictions[predictions['tile_name_right'] == tile]

        detailed.append({
            'tile_name': tile,
            'gt_count': len(gt_tile),
            'pred_count': len(pred_tile),
            'difference': len(pred_tile) - len(gt_tile),
            'gt_points': [(p.x, p.y) for p in gt_tile['geometry']] if len(gt_tile) > 0 else [],
            'pred_points': [(p.x, p.y) for p in pred_tile['geometry']] if len(pred_tile) > 0 else []
        })

    return pd.DataFrame(detailed)


# Create detailed comparison
detailed_df = create_detailed_comparison(ground_truth, predictions)
detailed_df.to_csv('detailed_tile_comparison.csv', index=False)
print("Detailed comparison saved to 'detailed_tile_comparison.csv'")