"""
HUMAN IN THE LOOP correction of if no ground truth is available

This means a model predicted iguanas and a full tile is supposed to be reviewed and false positives removed and missing annotations added.
"""
import copy

from typing import List, Optional

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
from active_learning.config.dataset_filter import DatasetCorrectionConfig, DatasetCorrectionReportConfig
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file
from examples.review_annotations import debug_hasty_fiftyone_v2
import fiftyone as fo
from loguru import logger


def main():
# /Users/christian/data/training_data/2025_07_10_refined/label_correction_floreana_2025_07_10_review_hasty_corrected_formatted.json
    base_path = Path('/Users/christian/data/training_data/2025_07_10_refined/Floreana_detection_classic/train')
    report_path = base_path / "report"
    config = DatasetCorrectionConfig(
        analysis_date="2025_07_10",
        num=56,
        type="points",
        subset_base_path=base_path,
        reference_base_path=Path(
            '/Users/christian/data/training_data/2025_07_10_refined/unzipped_hasty_annotation'),
        hasty_reference_annotation_name='label_correction_floreana_2025_07_10_review_hasty_corrected_formatted.json', # the full dataset annotations
        hasty_ground_truth_annotation_name='hasty_format_full_size.json', # the hasty annotations ground truth we created detections for
        # herdnet_annotation_name="herdnet_format_points.csv",
        detections_path='/Users/christian/PycharmProjects/hnee/HerdNet/data/label_correction/floreana_train_inference/detections.csv',
        correct_fp_gt=True,
        box_size=350,
        radius=150,
        images_path=base_path / "Default",
        dataset_name=f"full_label_correction_floreana_2025_07_10_train_correction",
        output_path=base_path / "output",
        corrected_path=base_path / "corrections"
    )

    config.output_path.mkdir(exist_ok=True)
    config.save(report_path / f"{config.dataset_name}_config.json")
    report_config = DatasetCorrectionReportConfig(**config.model_dump())
    report_config.report_path = base_path / "report"

    loaded_config = DatasetCorrectionConfig.load(report_path / f"{config.dataset_name}_config.json")

    # df_ground_truth = pd.read_csv(config.base_path / config.herdnet_annotation_name)

    if config.hasty_ground_truth_annotation_name:
        hA_ground_truth = HastyAnnotationV2.from_file(config.subset_base_path / config.hasty_ground_truth_annotation_name)

    images = list(config.images_path.glob(f"*.{config.suffix}"))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {config.images_path}")

    df_detections = pd.read_csv(config.detections_path)

    df_detections = df_detections[df_detections.scores > 0.2]
    #df_detections = df_detections[:15] # TODO debugging


    IL_false_positive_detections = herdnet_prediction_to_hasty(df_detections, config.images_path)

    # getting this reference just for the label classes
    hA_reference = HastyAnnotationV2.from_file(config.reference_base_path / config.hasty_reference_annotation_name)
    hA_prediction = HastyAnnotationV2(
        project_name="full_tile_correction",
        images=IL_false_positive_detections,
        export_format_version="1.1",
        label_classes=hA_reference.label_classes
    )
    hA_prediction_path = config.corrected_path / f"{config.dataset_name}_predictions_hasty.json"
    report_config.hA_prediction_path = hA_prediction_path
    hA_prediction.save(hA_prediction_path)
    hA_tiled_prediction = copy.deepcopy(hA_prediction)
    hA_tiled_prediction.images = []
    all_image_mappings: List[ImageCropMetadata] = []

    # delete the dataset if it exists
    try:
        fo.delete_dataset(config.dataset_name)
    except:
        logger.warning(f"Dataset {config.dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(name=config.dataset_name)
        dataset.persistent = True

    # TODO get some sort of grid manager, create a grid, create tiles, keep only interesting tiles, create crops for each of the detections
    # TODO upload them to CVAT
    # create crops for each of the detections
    for i in IL_false_positive_detections:
        im = PIL.Image.open(config.images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")

        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          width=1500,
                                                                                          height=1500,
                                                                                          im=im,
                                                                                          output_path=config.output_path)

        hA_tiled_prediction.images.extend(cropped_annotated_images)

        samples = [fo.Sample(filepath=path) for path in images_set]
        dataset.add_samples(samples)

        # Save image mappings
        # TODO save this at the end
        for image_mapping in image_mappings:
            image_mapping.save(
                config.output_path / f"{image_mapping.parent_label_id}_metadata.json")

        all_image_mappings.extend(image_mappings)

        for cropped_annotated_image in cropped_annotated_images:
            cropped_annotated_image.save(config.output_path / f"{cropped_annotated_image.image_name}_labels.json")

        # create a polygon around each Detection
        # TODO visualise where the crops happened

        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                             detections=cropped_annotated_images)

    report_config.hA_prediction_tiled_path = config.corrected_path / f"{config.dataset_name}_tiled_hasty.json"

    hA_tiled_prediction.save(report_config.hA_prediction_tiled_path)
    report_config.save(report_path / "report.json")

    # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
    dataset.annotate(
        anno_key=config.dataset_name,
        label_field=f"detection",
        attributes=[],
        launch_editor=True,
        organization="IguanasFromAbove",
        project_name="Orthomosaic_quality_control"
    )




if __name__ == "__main__":
    main()
