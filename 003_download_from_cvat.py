from pathlib import Path

from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty
from util.HastyAnnotationV2 import hA_from_file

#analysis_date = "2024_12_09"
#base_path = Path(f"/Users/christian/data/training_data") / analysis_date
lcrop_size = 640
num = 56
analysis_date = "2024_12_14"

if __name__ == "__main__":
    """
    """
    for dset in ["train"]:
        labels_path = Path("/Users/christian/data/training_data/2024_12_11/2024_12_14/train/crops_640_num56_overlap0")
        labels_path = Path(f"/Users/christian/data/training_data/2024_12_11/val/crops_640_numNone_overlap0")

        images_path = labels_path

        dataset_name = f"experiment_active_learning_{analysis_date}_{dset}_review_{lcrop_size}_{num}"
        dataset_name = f"experiment_active_learning_{analysis_date}_{dset}_review_{lcrop_size}"
        type = "boxes"

        hA_before_path = images_path / "hasty_format.json"

        cvat2hasty(hA_before_path, dataset_name)