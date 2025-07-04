"""
Look into the hasty dataset and analyze the data.
"""
import shapely
from loguru import logger
from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

labels_path = Path("/Users/christian/data/training_data/2025-07-02")

# hacky way
# get a list of images which are in a dataset
labels_file_path = labels_path / "unzipped_hasty_annotation/labels_2025_07_03.json"
df_flat = HastyAnnotationV2.from_file(labels_file_path).get_flat_df()

# dataset_mask = df_flat['dataset_name'] == "Fer_FCD01-02-03_20122021_single_images"
# dataset_mask = df_flat['dataset_name'] == "Fer_FCD01-02-03_20122021"

#dataset_mask = df_flat['dataset_name'].isin( ["Fer_FCD01-02-03_20122021", "Fer_FCD01-02-03_20122021_single_images"] )


dataset_name_island_mapping = {
    "Floreana_22.01.21_FPC07": "Floreana",
    "Floreana_03.02.21_FMO06": "Floreana",
    "SRPB06 1053 - 1112 falcon_25.01.20": "Unknown",  # SRPB prefix unclear
    "San_STJB01_12012023": "Santiago",  # San prefix + STJ likely Santiago
    "Fer_FCD01-02-03_20122021": "Fernandina",  # Fer prefix
    "FLMO02_28012023": "Floreana",  # FL prefix + MO code
    "FLBB01_28012023": "Floreana",  # FL prefix + BB code
    "Floreana_02.02.21_FMO01": "Floreana",
    "SCris_SRIL01_04022023": "San Cristobal",  # SCris prefix
    "Fer_FPM01-02_20122023": "Fernandina",  # Fer prefix
    "SCris_SRIL02_04022023": "San Cristobal",  # SCris prefix
    "SCruz_SCM01_06012023": "Santa Cruz",  # SCruz prefix
    "San_STJB02_12012023": "Santiago",  # San prefix + STJ
    "San_STJB03_12012023": "Santiago",  # San prefix + STJ
    "San_STJB04_12012023": "Santiago",  # San prefix + STJ
    "San_STJB06_12012023": "Santiago",  # San prefix + STJ
    "SCris_SRIL04_04022023": "San Cristobal",  # SCris prefix
    "single_images": "Mixed",  # Generic category
    "FMO02": "Floreana",  # FMO code pattern
    "FMO05": "Floreana",  # FMO code pattern
    "FMO03": "Floreana",  # FMO code pattern
    "Fer_FCD01-02-03_20122021_single_images": "Fernandina",  # Fer prefix
    "Genovesa": "Genovesa",  # Explicit island name
    "FMO04": "Floreana",  # FMO code pattern
    "FPA03 condor": "Floreana",  # FPA likely Floreana + location code
    "FSCA02": "Floreana",  # F prefix + SCA code
    "floreana": "Floreana",  # Explicit island name (lowercase)
    "FPM01_24012023": "Fernandina",  # F prefix + PM code
    "Isabella": "Isabela",  # Explicit island name (corrected spelling)
    "Fer_FPE02_07052024": "Fernandina",  # Fer prefix
    "San_STJB01_10012023_DJI_0068": "Santiago"  # San prefix + STJ
}




ortho_single_images_mapping = {
    "Fer_FCD01-02-03_20122021": "DroneDeploy",
    "Fer_FPM01-02_20122023": "DroneDeploy",
    "SCruz_SCM01_06012023": "Metashape",
    "Zooniverse_expert_phase_2": "Mixed",  # Multi-island dataset
    "Zooniverse_expert_phase_3": "Mixed",
    "San_STJB03_12012023": "OpenDroneMap",
}

# remove the names that are in the mapping
# Remove the ortho datasets to get only single images datasets
dataset_name_island_mapping_single_images = {
    dataset: island for dataset, island in dataset_name_island_mapping.items()
    if dataset not in ortho_single_images_mapping
}

# pivot this dictionary to map dataset names to islands
# Pivot this dictionary to map islands to lists of dataset names
island_to_datasets = {}
for dataset_name, island in dataset_name_island_mapping_single_images.items():
    if island not in island_to_datasets:
        island_to_datasets[island] = []
    island_to_datasets[island].append(dataset_name)



# Extract specific island dataset lists
floreana_dataset_names = island_to_datasets.get("Floreana", [])
fernandina_dataset_names = island_to_datasets.get("Fernandina", [])
santiago_dataset_names = island_to_datasets.get("Santiago", [])
san_cristobal_dataset_names = island_to_datasets.get("San Cristobal", [])
santa_cruz_dataset_names = island_to_datasets.get("Santa Cruz", [])
genovesa_dataset_names = island_to_datasets.get("Genovesa", [])
isabela_dataset_names = island_to_datasets.get("Isabela", [])
mixed_dataset_names = island_to_datasets.get("Mixed", [])
unknown_dataset_names = island_to_datasets.get("Unknown", [])

df_flat = df_flat[df_flat["class_name"].isin(["iguana", "crab"])]

df_flat["island"] = df_flat["dataset_name"].map(dataset_name_island_mapping)
df_flat["source"] = df_flat["dataset_name"].map(ortho_single_images_mapping)
# everything that is not in the mapping is set to "Unknown"
df_flat['source'] = df_flat['source'].fillna('SingleImage')


dataset_mask = df_flat['island'].isin(["Floreana", "Fernandina", "Genovesa"])
df_flat = df_flat[dataset_mask]
dataset_mask = df_flat['source'].isin(["SingleImage"])
df_flat = df_flat[dataset_mask]

# Calculate bbox area from bbox_x1y1x2y2 format: [x1, y1, x2, y2]
# Parse the bbox coordinates and calculate width, height, and area
def parse_bbox_and_calculate_area(bbox: shapely.Polygon):
    try:
        area = bbox.area
        width = bbox.bounds[2] - bbox.bounds[0]  # x2 - x1
        height = bbox.bounds[3] - bbox.bounds[1]
        return area, width, height
    except Exception as e:
        logger.error(f"Error parsing bbox {bbox}: {e}")
        return 0, 0, 0



# Apply bbox parsing to create new columns
bbox_data = df_flat['bbox_polygon'].apply(parse_bbox_and_calculate_area)
df_filtered = df_flat.copy()  # Avoid SettingWithCopyWarning
df_filtered['bbox_area'] = [x[0] for x in bbox_data]
df_filtered['bbox_width'] = [x[1] for x in bbox_data]
df_filtered['bbox_height'] = [x[2] for x in bbox_data]

# Create aggregated statistics table
aggregation_stats = df_filtered.groupby(['island', 'class_name']).agg({
    'image_id': ['count', 'nunique'],  # count of annotations, unique images
    'bbox_area': ['mean', 'std', 'min', 'max'],
    'bbox_width': ['mean', 'std'],
    'bbox_height': ['mean', 'std']
}).round(2)

# Flatten column names
aggregation_stats.columns = ['_'.join(col).strip() for col in aggregation_stats.columns]

# Rename columns for clarity
column_mapping = {
    'image_id_count': 'total_annotations',
    'image_id_nunique': 'unique_images',
    'bbox_area_mean': 'avg_bbox_area',
    'bbox_area_std': 'bbox_area_std',
    'bbox_area_min': 'min_bbox_area',
    'bbox_area_max': 'max_bbox_area',
    'bbox_width_mean': 'avg_bbox_width',
    'bbox_width_std': 'bbox_width_std',
    'bbox_height_mean': 'avg_bbox_height',
    'bbox_height_std': 'bbox_height_std'
}
aggregation_stats = aggregation_stats.rename(columns=column_mapping)

# Calculate average annotations per image
if 'total_annotations' in aggregation_stats.columns and 'unique_images' in aggregation_stats.columns:
    aggregation_stats['avg_annotations_per_image'] = (
            aggregation_stats['total_annotations'] / aggregation_stats['unique_images']
    ).round(2)

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

logger.info("\nDetailed class statistics:")
for _, row in aggregation_stats.iterrows():
    island_name = row['island']
    class_name = row['class_name']
    logger.info(f"  {island_name}:")
    logger.info(f"  {class_name}:")
    logger.info(f"    - Total annotations: {int(row['total_annotations'])}")
    logger.info(f"    - Unique images: {int(row['unique_images'])}")
    logger.info(f"    - Avg annotations per image: {row['avg_annotations_per_image']:.2f}")
    logger.info(f"    - Avg bbox area: {row['avg_bbox_area']:.1f} pixels²")
    logger.info(f"    - Avg bbox width: {row['avg_bbox_width']:.1f} pixels")
    logger.info(f"    - Avg bbox height: {row['avg_bbox_height']:.1f} pixels")

logger.info("\nAggregated Statistics by Dataset and Class:")
logger.info("=" * 50)
logger.info(aggregation_stats.to_string(index=False))
latex_table = aggregation_stats.to_latex(index=False, float_format="%.2f")
with open("hasty_dataset_analysis.tex", "w") as f:
    f.write(latex_table)

# Save results to CSV for further analysis
output_path = labels_path / "dataset_analysis_summary.csv"
aggregation_stats.to_csv(output_path, index=False)
logger.info(f"\nResults saved to: {output_path}")

# Display the final aggregation table
logger.info("\nFinal Aggregation Table:")
logger.info(aggregation_stats)


# create a list of unique images from fernandina for training purposes. Sort by image name to reduce overlap and therefore leakage. Create a 50/50 split with scikit-learn.
from sklearn.model_selection import train_test_split
unique_images = df_filtered['image_id'].unique()
# sort the unique images to ensure consistent splits
unique_images = sorted(unique_images)
train_images, val_images = train_test_split(unique_images, test_size=0.5)

# TODO create this mnist style crops