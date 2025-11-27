"""
A hasty annotation should not contain the same image_name in a single dataset.
the image should be in the right folder
"""
import pandas as pd
from loguru import logger
from typing import Dict

from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2



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
    hA_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_3.json")
    hA_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_2.json")
    images_base_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/2025_07_10_images_final")

    hA = HastyAnnotationV2.from_file(hA_path)

    hA_flat = hA.get_flat_df()

    folder_issues = analyze_folder_structure(hA, images_base_path)

    hA.dataset_statistics()
