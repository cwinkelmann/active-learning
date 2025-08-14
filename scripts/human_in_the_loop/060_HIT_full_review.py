# """
# TODO: This is suposed to be run and correct all annotations not only false positives but also false negatives. To get this to work the full image needs to be gridded
# HUMAN IN THE LOOP correction of if no ground truth is available
#
# This means a model predicted iguanas and a full tile is supposed to be reviewed and false positives removed and missing annotations added.
# """
# import copy
#
# from typing import List, Optional
#
# import json
#
# import PIL.Image
# import pandas as pd
# from pathlib import Path
#
# from active_learning.analyse_detections import analyse_point_detections_greedy
# from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
# from active_learning.types.ImageCropMetadata import ImageCropMetadata
# from active_learning.util.converter import herdnet_prediction_to_hasty
# from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
# from active_learning.util.geospatial_slice import get_tiles
# from active_learning.util.image_manipulation import crop_out_individual_object
# from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file
# from examples.review_annotations import debug_hasty_fiftyone_v2
# import fiftyone as fo
# from loguru import logger
#
#
# def geospatial_inference_pipeline(orthomosaic_path: Path,
#                                   hydra_cfg: DictConfig):
#     """
#     Main function to run the geospatial inference pipeline.
#     :param orthomosaic_path: Path to the orthomosaic image.
#     :return: None
#     """
#     logger.info(f"Running geospatial inference on {orthomosaic_path}")
#     tile_images_path = orthomosaic_path.with_name(orthomosaic_path.stem + '_tiles')
#
#
# def main():
# # /Users/christian/data/training_data/2025_07_10_refined/label_correction_floreana_2025_07_10_review_hasty_corrected_formatted.json
#     base_path = Path('/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021_tiles')
#     report_path = base_path / "report"
#     config = DatasetCorrectionConfig(
#         analysis_date="2025_08_12",
#         type="points",
#         subset_base_path=base_path,
#         reference_base_path=Path(
#             '/Users/christian/data/training_data/2025_07_10_refined/unzipped_hasty_annotation'),
#         hasty_reference_annotation_name='label_correction_floreana_2025_07_10_review_hasty_corrected_formatted.json', # the full dataset annotations
#
#         detections_path=Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021_tiles/plots"),
#         correct_fp_gt=True,
#         box_size=350,
#         radius=150,
#         images_path=base_path / "Default",
#         dataset_name=f"full_label_correction_floreana_2025_07_10_train_correction",
#         output_path=base_path / "output",
#         corrected_path=base_path / "corrections"
#     )
#
#
#
#     config.output_path.mkdir(exist_ok=True)
#     config.save(report_path / f"{config.dataset_name}_config.json")
#     report_config = DatasetCorrectionReportConfig(**config.model_dump())
#     report_config.report_path = base_path / "report"
#
#     tile_images_path.mkdir(exist_ok=True, parents=True)
#
#     gdf_tiles, images_dir = get_tiles(
#         orthomosaic_path=orthomosaic_path,
#         output_dir=tile_images_path,
#         tile_size=1250
#     )
#
#     # TODO Get the predicted annotations from the GeoJSON for each tile
#
#
#
#     # TODO submit data
#
#
#
#
# if __name__ == "__main__":
#     main()
