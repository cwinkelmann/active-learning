"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
import json

import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.image_manipulation import crop_out_individual_object

def save_metadata_json(metadata: ImageCropMetadata, file_path: Path):
    """Save ImageCropMetadata object as a JSON file."""
    with open(file_path, "w") as f:
        json.dump(metadata.dict(), f, indent=4)

if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2024_12_09"
    # lcrop_size = 640
    num = 56
    type = "points"
    # test dataset
    base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    base_path = Path(f'/Users/christian/data/orthomosaics/tiles')
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/active_learning/tests/data/FMO02_full_orthophoto_herdnet_detections.csv')
    images_path = base_path # / "Default"
    output_path = base_path / "object_crops"
    output_path.mkdir(exist_ok=True)

    images = images_path.glob("*.tif")

    IL_all_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    # create crops for each of the detections
    for i in IL_all_detections:
        im = PIL.Image.open(images_path / i.image_name)
        # convert to RGB
        if im.mode != "RGB":
            im = im.convert("RGB")
        # TODO create a box around each Detection
        # TODO it sucks quite a lot, that I can't get my head around the fact this would work better with geospatial data.

        image_mappings, cropped_annotated_images = crop_out_individual_object(i, width=512,
                                                                                     height=512, im=im,
                                                                                     output_path=output_path)
        # Save image mappings
        for image_mapping in image_mappings:
            image_mapping.save(output_path / f"{image_mapping.parent_image}__{image_mapping.parent_label_id}_metadata.json")
        for cropped_annotated_image in cropped_annotated_images:
            cropped_annotated_image.save(output_path / f"{cropped_annotated_image.image_name}_labels.json")

        # TODO create HastyAnnotationV2 object for everythings

        # create a polygon around each Detection
        # TODO visualise where the crops happened

        # TODO keep the annotated images

        # TODO add the cropped annotated images to some sort of database to submit for Human in the loop


    pass
    # TODO tile thes
