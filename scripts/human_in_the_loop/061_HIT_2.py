"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
from pathlib import Path

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
    hA_updated = hit_cvat_download(report_path)