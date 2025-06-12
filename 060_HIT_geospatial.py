"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right.
There are two options: 1. mark it as iguanas or 2. mark it as a partial iguana

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





def main(dataset_name, analysis_date: str, output_path: Path, hA_template: HastyAnnotationV2):
    IL_all_detections = copy.deepcopy(hA_template.images)

    ## TODO interacting with that dataset is quite dangerous, so we should just create
    # delete the dataset if it exists
    try:
        dataset = fo.load_dataset(dataset_name)
    except:
        logger.warning(f"Dataset {dataset_name} does not exist")
        dataset = None


    if dataset is None:
        dataset = fo.Dataset(name=dataset_name)
        dataset.persistent = True

    # try:
    #     fo.delete_dataset(dataset_name)
    # except:
    #     logger.warning(f"Dataset {dataset_name} does not exist")
    # finally:
    #     # Create an empty dataset, TODO put this away so the dataset is just passed into this
    #     dataset = fo.Dataset(name=dataset_name)
    #     dataset.persistent = True

        images_set: List[Path] = []

        # convert each image to RGB JPEG
        for i in IL_all_detections:
            im = PIL.Image.open(images_path / i.image_name)
            # convert to RGB
            if im.mode != "RGB":
                im = im.convert("RGB")

            image_name_jpg = str(Path(i.image_name).with_suffix(".jpg"))
            i.image_name = image_name_jpg
            jpg_path = images_jpg_path / image_name_jpg
            im.save(jpg_path, "JPEG")
            # TODO persist the images
            images_set.append(jpg_path)


        samples = [fo.Sample(filepath=path) for path in images_set]
        dataset.add_samples(samples)


        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                   # images_set=images_set,
                                   detections=IL_all_detections)


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
    # Create an empty dataset, TODO put this away so the dataset is just passed into this

    # lcrop_size = 640
    num = 56
    type = "points"

    # test dataset
    # base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    # df_detections = pd.read_csv(
    #     '/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    # images_path = base_path  / "Default"

    df_detections = pd.read_csv(
        '/scripts/inferencing/detections_jpg.csv')

    # TODO get this template too to persist the converted images later
    base_path = Path(f'/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/Fer_FCD01-02-03_tiles')
    template_labels_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")

    hA_template = hA_from_file(template_labels_path)
    analysis_date = "2025_04_23"
    output_path = Path(base_path) / "output"
    dataset_name = f"eal_{analysis_date}_{base_path.name}_review"
    # replace '-' with '_'
    dataset_name = dataset_name.replace("-", "_")
    
    images_path = base_path
    images_jpg_path = images_path / "jpg"
    images_jpg_path.mkdir(parents=True, exist_ok=True)

    IL_all_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    # IL_all_detections = IL_all_detections[:1] # TODO ths

    hA_template.images = IL_all_detections
    hA_template.save(output_path / f"{dataset_name}_tiled_hasty.json")

    # TODO this is debugging code
    fo.delete_dataset(dataset_name)

    main(dataset_name=dataset_name,
         analysis_date=analysis_date,
         output_path=output_path,
         # df_detections=df_detections,
         hA_template=hA_template)
