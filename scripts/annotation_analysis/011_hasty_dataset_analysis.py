"""
Look into the hasty dataset and analyze the data.
"""
import pandas as pd
import shapely
from loguru import logger
from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, filter_by_class
from com.biospheredata.types.status import LabelingStatus
from scripts.training_data_preparation import dataset_configs_hasty_point_iguanas


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


def get_aggregations(df_flat):
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
    aggregation_stats = df_filtered.groupby(['island', 'class_name']).agg({
        'image_id': ['count', 'nunique'],  # count of annotations, unique images
        'bbox_area': ['mean', 'std'],
        'bbox_width': ['mean', 'std'],
        'bbox_height': ['mean', 'std']
    }).round(2)

    aggregation_stats = df_filtered.groupby(['source', 'island', 'class_name']).agg({
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


if __name__ == '__main__':
    # dataset_name = 'eikelboom2019'
    # labels_path = Path("/raid/cwinkelmann/training_data/iguana/2025_08_10_endgame")
    labels_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/IFA_training_data/")
    # labels_path = Path("/Users/christian/data/training_data/2025_08_10_endgame")

    # hacky way
    # get a list of images which are in a dataset
    # labels_file_path = labels_path / "unzipped_hasty_annotation/2025_09_19_orthomosaic_data_combined_corrections_3.json"
    labels_file_path = labels_path / "2025_10_11_labels.json"
    # "/home/cwinkelmann/work/Herdnet/data/2025_10_11/2025_09_19_orthomosaic_data_combined_corrections_4.json"
    target_labels = ['iguana_point', "iguana"]

    # labels_file_path = labels_path / "unzipped_hasty_annotation/2025_07_10_labels_final.json"
    hA = HastyAnnotationV2.from_file(labels_file_path)

    # keep only completed
    hA.images = [img for img in hA.images if img.image_status == LabelingStatus.COMPLETED.value]
    # remove all labels which are not in target_labels
    hA = filter_by_class(hA, target_labels)
    hA_stats = hA.dataset_statistics()
    df_ha_stats = pd.DataFrame(hA_stats)
    target_labels_points =  ["iguana_point"]
    target_labels_box =  ["iguana"]

    df_flat = hA.get_flat_df()
    df_flat_points = df_flat[df_flat["class_name"].isin(target_labels_points)]

    df_flat = df_flat[df_flat["class_name"].isin(target_labels_box)]

    df_flat.groupby('dataset_name').size().sort_values(ascending=False)



    dataset_name_island_mapping = {

        "SRPB06 1053 - 1112 falcon_25.01.20": "Unknown",  # SRPB prefix unclear
        "San_STJB01_12012023": "Santiago",  # San prefix + STJ likely Santiago
        "Fer_FCD01-02-03_20122021": "Fernandina",  # Fer prefix
        'ha_corrected_fer_fnj01_19122021': "Fernandina",
        'ha_fer_fnj01_19122021': "Fernandina",
        'ha_corrected_fer_fpe09_18122021': "Fernandina",
        'ha_fer_fpe09_18122021': "Fernandina",
        'ha_corrected_fer_fnd02_19122021': "Fernandina",
        'ha_fer_fnd02_19122021': "Fernandina",
        'ha_fer_fni03_04_19122021': "Fernandina",
        'ha_corrected_fer_fni03_04_19122021': "Fernandina",
        'ha_corrected_fer_fna01_02_20122021': "Fernandina",
        'ha_fer_fna01_02_20122021': "Fernandina",
        'ha_corrected_fer_fef01_02_20012023': "Fernandina",
        'ha_fer_fef01_02_20012023': "Fernandina",

        # The uncorrected labels are not added anymore
        'ha_corrected_fer_fwk01_20122021': "Fernandina",
        'ha_corrected_fer_fe01_02_20012023': "Fernandina",


        'ha_corrected_flo_flpo02_04022021': "Floreana",
        'ha_flo_flpo02_04022021': "Floreana",
        'ha_corrected_flo_flpc04_22012021': "Floreana",
        'ha_flo_flpc04_22012021': "Floreana",
        'ha_corrected_flo_flpc06_22012021': "Floreana",
        'ha_flo_flpc06_22012021': "Floreana",
        'ha_corrected_flo_flpo01_04022021': "Floreana",
        'ha_flo_flpo01_04022021': "Floreana",
        'ha_corrected_flo_flpc03_22012021': "Floreana",
        'ha_flo_flpc03_22012021': "Floreana",
        'ha_corrected_flo_flbb01_28012023': "Floreana",
        'ha_flo_flbb01_28012023': "Floreana",
        'ha_corrected_flo_flbb02_28012023': "Floreana",
        'ha_flo_flbb02_28012023': "Floreana",

        "Floreana_22.01.21_FPC07": "Floreana",
        "Floreana_03.02.21_FMO06": "Floreana",
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
        "floreana_FPE01_FECA01": "Fernandina",  # This was named incorrectly, should be Fernandina
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
        "SCris_SRIL01_04022023": "DroneDeploy",  # Orthomosaic
        "SCris_SRIL02_04022023": "DroneDeploy",  # Orthomosaic
        "SCris_SRIL04_04022023": "DroneDeploy",  # Orthomosaic, 4 iguanas

        "San_STJB01_12012023": "OpenDroneMap",  # Orthomosaic, 13
        "San_STJB02_12012023": "OpenDroneMap",  # Orthomosaic
        "San_STJB03_12012023": "OpenDroneMap",  # Orthomosaic
        "San_STJB04_12012023": "OpenDroneMap",  # Orthomosaic
        "San_STJB06_12012023": "OpenDroneMap",  # Orthomosaic

        'ha_corrected_fer_fnj01_19122021': "DroneDeploy",
        'ha_fer_fnj01_19122021': "DroneDeploy",
        'ha_corrected_fer_fpe09_18122021': "DroneDeploy",
        'ha_fer_fpe09_18122021': "DroneDeploy",
        'ha_corrected_fer_fnd02_19122021': "DroneDeploy",
        'ha_fer_fnd02_19122021': "DroneDeploy",
        'ha_fer_fni03_04_19122021': "DroneDeploy",
        'ha_corrected_fer_fni03_04_19122021': "DroneDeploy",
        'ha_corrected_fer_fna01_02_20122021': "DroneDeploy",
        'ha_fer_fna01_02_20122021': "DroneDeploy",
        'ha_corrected_fer_fef01_02_20012023': "DroneDeploy",
        'ha_fer_fef01_02_20012023': "DroneDeploy",
        'ha_corrected_fer_fwk01_20122021': "DroneDeploy",
        'ha_corrected_fer_fe01_02_20012023': "DroneDeploy",

        'ha_corrected_flo_flpo01_04022021': "DroneDeploy",
        'ha_flo_flpo01_04022021': "DroneDeploy",
        'ha_corrected_flo_flpo02_04022021': "DroneDeploy",
        'ha_flo_flpo02_04022021': "DroneDeploy",

        'ha_corrected_flo_flpc03_22012021': "DroneDeploy",
        'ha_flo_flpc03_22012021': "DroneDeploy",
        'ha_corrected_flo_flpc04_22012021': "DroneDeploy",
        'ha_flo_flpc04_22012021': "DroneDeploy",
        'ha_corrected_flo_flpc06_22012021': "DroneDeploy",
        'ha_flo_flpc06_22012021': "DroneDeploy",


        'ha_corrected_flo_flbb01_28012023': "DroneDeploy",
        'ha_flo_flbb01_28012023': "DroneDeploy",
        'ha_corrected_flo_flbb02_28012023': "DroneDeploy",
        'ha_flo_flbb02_28012023': "DroneDeploy",

    }
    df_ha_stats = df_ha_stats.T.reset_index(drop=False).rename(columns={"index": "dataset_name"})
    df_ortho_single_images_mapping = pd.DataFrame(list(ortho_single_images_mapping.items()),
                      columns=['dataset_name', 'Software'])
    df_islands_mapping   = pd.DataFrame(list(dataset_name_island_mapping.items()),
                              columns=['dataset_name', 'Island'])
    # map the dataset names to islands and if they are orthomosaics or single images
    df_ha_stats = df_ha_stats.merge(df_ortho_single_images_mapping, on='dataset_name', how='left').fillna("Single Images")
    df_ha_stats = df_ha_stats.merge(df_islands_mapping, on='dataset_name', how='left').fillna("Mixed")

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



    df_flat["island"] = df_flat["dataset_name"].map(dataset_name_island_mapping)
    df_flat["source"] = df_flat["dataset_name"].map(ortho_single_images_mapping)
    # everything that is not in the mapping is set to "Unknown"
    df_flat['source'] = df_flat['source'].fillna('SingleImage')

    df_flat_points["island"] = df_flat_points["dataset_name"].map(dataset_name_island_mapping)
    df_flat_points["source"] = df_flat_points["dataset_name"].map(ortho_single_images_mapping)
    # everything that is not in the mapping is set to "Unknown"
    df_flat_points['source'] = df_flat_points['source'].fillna('SingleImage')


    label_dataset_table_p = create_label_distribution_table(df_flat_points, group_by='dataset_name', target_labels = target_labels_points)

    label_dataset_table_box = create_label_distribution_table(df_flat, group_by='dataset_name', target_labels = ['iguana'])


    label_island_table = create_label_distribution_table(df_flat, group_by='island', target_labels = ['iguana'])
    label_source_table = create_label_distribution_table(df_flat, group_by='source', target_labels = ['iguana'])

    label_island_table_p = create_label_distribution_table(df_flat_points, group_by='island', target_labels = target_labels_points)
    label_source_table_p = create_label_distribution_table(df_flat_points, group_by='source', target_labels = target_labels_points)



    # apply this to boxes
    dataset_mask = df_flat['island'].isin(["Floreana", "Fernandina", "Genovesa"])
    df_flat = df_flat[dataset_mask]
    # dataset_mask = df_flat['source'].isin(["SingleImage"])
    # dataset_mask = df_flat['source'].isin(["SingleImage"])

    df_flat = df_flat[dataset_mask]



    aggregation_stats = get_aggregations(df_flat)

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

