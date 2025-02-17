"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
import PIL.Image
import pandas as pd
from pathlib import Path

from active_learning.util.Annotation import project_point_to_crop
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.image_manipulation import crop_out_individual_object, crop_polygons

if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2024_12_09"
    # lcrop_size = 640
    num = 56
    type = "points"
    # test dataset
    base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    images_path = base_path / "Default"
    output_path = base_path / "object_crops"
    output_path.mkdir(exist_ok=True)

    images = images_path.glob("*.JPG")

    IL_all_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    # create crops for each of the detections
    for i in IL_all_detections:
        im = PIL.Image.open(images_path / i.image_name)

        # TODO create a box around each Detection

        boxes, image_mappings, cropped_annotated_images = crop_out_individual_object(i, width=512,
                                                                       height=512, im=im, output_path=output_path)

        # create a polygon around each Detection
        # TODO visualise where the crops happened

        # TODO keep image mappinds

        # TODO keep the annotated images

        # TODO add the cropped annotated images to some sort of database to submit for Human in the loop


    pass
    # TODO tile thes