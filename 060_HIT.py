"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
import copy

from typing import List

import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.image_manipulation import crop_out_individual_object
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, HastyAnnotationV2, hA_from_file
from examples.review_annotations import debug_hasty_fiftyone_v2
import fiftyone as fo
from loguru import logger


def main():
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2025_02_22"
    # lcrop_size = 640
    num = 56
    type = "points"

    # test dataset
    # base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    # df_detections = pd.read_csv(
    #     '/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    # images_path = base_path  / "Default"

    base_path = Path(f'/Users/christian/data/training_data/2025_02_22_HIT/FMO02_full_orthophoto_tiles')
    df_detections = pd.read_csv(
         '/Users/christian/data/training_data/2025_02_22_HIT/06-25-46/detections.csv')

    images_path = base_path
    dataset_name = f"eal_{analysis_date}_review"


    output_path = base_path / "object_crops"
    output_path.mkdir(exist_ok=True)

    IL_all_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    # getting this reference just for the label classes
    hA_reference = hA_from_file(Path("/Users/christian/data/training_data/2024_12_09_debug/unzipped_hasty_annotation/FMO02_03_05_labels.json"))
    hA_prediction = HastyAnnotationV2(
        project_name="correction",
        images=IL_all_detections,
        export_format_version="1.1",
        label_classes=hA_reference.label_classes
    )

    hA_prediction.save(output_path / f"{dataset_name}_hasty.json")
    hA_tiled_prediction = copy.deepcopy(hA_prediction)
    hA_tiled_prediction.images = []
    all_image_mappings: List[ImageCropMetadata] = []

    # # delete the dataset if it exists
    # try:
    #     fo.delete_dataset(dataset_name)
    # except:
    #     logger.warning(f"Dataset {dataset_name} does not exist")
    # finally:
    #     # Create an empty dataset, TODO put this away so the dataset is just passed into this
    #     dataset = fo.Dataset(name=dataset_name)
    #     dataset.persistent = True

    # create crops for each of the detections
    for i in IL_all_detections:
        im = PIL.Image.open(images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")

        # TODO it produces the same image multiple times, tile_22.tif_ba2ec74e-76b8-4a26-ae48-dc969f9a4dd7 tile_22.tif_96eb06c5-e0b8-4ea6-96db-138b010fdee0.jpg
        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          width=512,
                                                                                          height=512,
                                                                                          im=im,
                                                                                          output_path=output_path)

        hA_tiled_prediction.images.extend(cropped_annotated_images)

        samples = [fo.Sample(filepath=path) for path in images_set]
        dataset.add_samples(samples)

        # Save image mappings
        # TODO save this at the end
        for image_mapping in image_mappings:
            image_mapping.save(
                output_path / f"{image_mapping.parent_label_id}_metadata.json")

        all_image_mappings.extend(image_mappings)

        for cropped_annotated_image in cropped_annotated_images:
            cropped_annotated_image.save(output_path / f"{cropped_annotated_image.image_name}_labels.json")

        # create a polygon around each Detection
        # TODO visualise where the crops happened

        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                   images_set=images_set,
                                   detections=cropped_annotated_images)

        # TODO keep the annotated images

        # TODO add the cropped annotated images to some sort of database to submit for Human in the loop
    hA_tiled_prediction.save(output_path / f"{dataset_name}_tiled_hasty.json")

    pass
    # TODO tile thes
    #
    # session = fo.launch_app(dataset)
    # session.wait()


    # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
    dataset.annotate(
        anno_key=dataset_name,
        label_field=f"detection",
        attributes=[],
        launch_editor=True,
        organization="IguanasFromAbove",
        project_name="Orthomosaic_quality_control"
    )


if __name__ == "__main__":
    main()
