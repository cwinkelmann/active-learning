"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
import shapely
from loguru import logger
from pathlib import Path
from typing import Optional

from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty, download_cvat_annotations, foDataset2Hasty, \
    determine_changes
from active_learning.types.Exceptions import TooManyLabelsError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabel
from scripts.human_in_the_loop.helper import hit_cvat_download

if __name__ == "__main__":
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report.json")
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    # config = DatasetCorrectionReportConfig.load("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    # report_path_1: Path = Path("/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/report/report_copy.json")
    # report_path_2: Path = Path("/Users/christian/data/training_data/2025_07_10_refined/Floreana_detection_classic/train/report/report.json")
    # report_path_3: Path = Path("/Users/christian/data/training_data/2025_08_10_endgame/Fernandina_s_detection/train/report/report.json")
    report_path_4: Path = Path("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/Delplanque2022_512_overlap_160_ebFalse/delplanque_train/train/report/report.json")

    report_path = report_path_4
    hit_cvat_download(report_path)