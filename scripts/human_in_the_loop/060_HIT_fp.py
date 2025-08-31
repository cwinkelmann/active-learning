"""
HUMAN IN THE LOOP correction of existing annotations in Hasty format and not geospatial format.

Mainly correct for false positives in the prediction which could be false_negatives in the ground truth in point detection annotations.

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

import fiftyone as fo
from loguru import logger


def main():
    base_path = Path('/Users/christian/data/training_data/2025_08_10_endgame/Fernandina_s_detection/train')

    report_path = base_path / "report"
    config = DatasetCorrectionConfig(
        analysis_date="correcting_fernandina_s",
        dataset_name=f"fernandina_s_correction",
        num=56,
        type="points",
        subset_base_path=base_path,
        reference_base_path=Path(
            '/Users/christian/data/training_data/2025_08_10_label_correction'),
        hasty_reference_annotation_name='full_label_correction_floreana_2025_07_10_train_correction_hasty_corrected_1.json',

        # the full dataset annotations
        hasty_ground_truth_annotation_name='hasty_format_full_size.json',
        # the hasty annotations ground truth we created detections for
        herdnet_annotation_name="herdnet_format_points.csv", # should be in base_path
        detections_path='/Volumes/2TB/work/training_data_sync/herdnet/outputs/2025-08-11/09-38-28/detections.csv',
        correct_fp_gt=True,
        box_size=350,
        radius=150,
        images_path=base_path / "Default",
        output_path=base_path / "output",
        corrected_path=base_path / "corrections"
    )

    config.output_path.mkdir(exist_ok=True)
    config_path = report_path / f"{config.dataset_name}_config.json"
    config.save(config_path)
    report_config = DatasetCorrectionReportConfig(**config.model_dump())
    report_config.report_path = base_path / "report"

    loaded_config = DatasetCorrectionConfig.load(config_path)

    df_ground_truth = pd.read_csv(config.subset_base_path / config.herdnet_annotation_name)

    hA_ground_truth = HastyAnnotationV2.from_file(config.subset_base_path / config.hasty_ground_truth_annotation_name)

    images = list(config.images_path.glob(f"*.{config.suffix}"))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {config.images_path}")

    df_detections = pd.read_csv(config.detections_path)
    # df_detections.species = "iguana_point" # if that does not show up in CVAT you will have to create it manually
    df_detections = df_detections[df_detections.scores > 0.3]
    # df_detections = df_detections[:15] # TODO debugging


    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=config.radius
    )

    logger.info(f"Found {len(df_false_positives)} false positives, {len(df_true_positives)} true positives and {len(df_false_negatives)} false negatives in the detections for a Ground Truth of {len(df_ground_truth)} detections.")
    logger.info(f"This means that our error is: {(len(df_false_positives) + len(df_true_positives)) / len(df_ground_truth)} .")

    if len(df_false_positives) == 0:
        raise ValueError(f"No false positives found in the detections. Normally that would be good. But here we want to correct false positives in the ground truth. ")

    IL_false_positive_detections = herdnet_prediction_to_hasty(df_false_positives, config.images_path, hA_reference=hA_ground_truth)

    # getting this reference just for the label classes
    hA_reference = HastyAnnotationV2.from_file(config.reference_base_path / config.hasty_reference_annotation_name)
    hA_prediction = HastyAnnotationV2(
        project_name="false_positive_correction",
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
        # fo.dataset_exists(config.dataset_name)
        fo.delete_dataset(config.dataset_name)
        pass
    except:
        logger.warning(f"Dataset {config.dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(name=config.dataset_name)
        dataset.persistent = True

    # create crops for each of the detections
    for i in IL_false_positive_detections:
        im = PIL.Image.open(config.images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")

        # TODO it produces the same image multiple times, tile_22.tif_ba2ec74e-76b8-4a26-ae48-dc969f9a4dd7 tile_22.tif_96eb06c5-e0b8-4ea6-96db-138b010fdee0.jpg
        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          width=512,
                                                                                          height=512,
                                                                                          im=im,
                                                                                          output_path=config.output_path,
                                                                                          )

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

    # report_config.image_mappings = image_mappings
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
        project_name="Single_Image_FP_correction"
    )

    logger.info(f"Correct the false positives in the CVAT interface and then save the annotations. ")
    logger.info(f"After saving the annotations, you can run the next script to apply the corrections to the ground truth annotations with {config_path}")



if __name__ == "__main__":
    main()
