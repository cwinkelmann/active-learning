"""
A hasty annotation should not contain the same image_name in a single dataset.
the image should be in the right folder
"""
import pandas as pd
from loguru import logger
from typing import Dict

from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def analyze_duplicates(hA: HastyAnnotationV2) -> Dict:
    """Analyze duplicate image_names within datasets."""

    hA_flat = hA.get_flat_df()
    datasets = hA_flat["dataset_name"].unique()

    for dataset in datasets:
        image_names = [i.image_name for i in hA.images if i.dataset_name == dataset]
        nunique = pd.Series(image_names).nunique()

        assert len(image_names) == nunique


    logger.info(f"No duplicate image_names found in {hA.project_name}")

def analyze_folder_structure(hA: HastyAnnotationV2, images_base_path: Path) -> Dict:
    """Analyze if images are in correct folder structure."""

    issues = []
    missing_files = []
    correct_structure = 0

    for image in hA.images:

        # Expected path: images_base_path / dataset_name / image_name
        expected_path = images_base_path / image.dataset_name / image.image_name

        # Check if file exists at expected location
        if expected_path.exists():
            correct_structure += 1
        else:
            logger.error(f"Image {image.image_name} not in {expected_path}")

    logger.info(f"Correct structure: {correct_structure}")

if __name__ == "__main__":
    hA_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections.json")
    hA_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_bak.json")
    images_base_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_07_10_images_final")

    hA = HastyAnnotationV2.from_file(hA_path)

    hA_flat = hA.get_flat_df()
    result = analyze_duplicates(hA)

    folder_issues = analyze_folder_structure(hA, images_base_path)

    hA_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections.json")
    hA_path_2 = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_bak.json")

    df_1 = HastyAnnotationV2.from_file(hA_path).get_flat_df()
    df_2 = HastyAnnotationV2.from_file(hA_path_2).get_flat_df()


    assert len(df_1) == len(df_2)
    pass