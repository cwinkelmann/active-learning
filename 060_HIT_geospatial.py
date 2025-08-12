"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right.
There are two options: 1. mark it as iguanas or 2. mark it as a partial iguana

Then prepare the output for another training round

"""
import PIL.Image
import copy
import fiftyone as fo
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import List

from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def main(dataset_name,
         analysis_date: str,
         output_path: Path,
         hA_template: HastyAnnotationV2,
         anno_key = None):
    """
    Main function to create a FiftyOne dataset for human-in-the-loop annotation review.
    :param dataset_name:
    :param analysis_date:
    :param output_path:
    :param hA_template:
    :return:
    """

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
        anno_key=anno_key if anno_key is not None else dataset_name,
        label_field=f"detection",
        attributes=[],
        launch_editor=True,
        organization="IguanasFromAbove",
        project_name="Orthomosaic_label_correction",
        task_size=50
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

    detections_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/FCD01-07_04052024_orthomosaic_tiles_detections_pi_True.csv")
    df_detections = pd.read_csv(detections_path)

    # TODO get this template too to persist the converted images later
    base_path = Path(f'/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FCD01-07_04052024_orthomosaic_tiles/jpg')
    template_labels_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")

    hA_template = HastyAnnotationV2.from_file(template_labels_path)
    analysis_date = "2025_06_16"
    output_path = Path(base_path) / "output"
    dataset_name = f"eal_{analysis_date}_{base_path.name}_review"
    # replace '-' with '_'

    images_path = base_path
    images_jpg_path = images_path / "jpg"
    images_jpg_path.mkdir(parents=True, exist_ok=True)

    df_detections = df_detections[df_detections["scores"] > 0.5]  # filter out low confidence detections

    IL_all_detections = herdnet_prediction_to_hasty(df_detections, images_path)
    det_subset_start = 0
    det_subset = 50
    dataset_name = dataset_name.replace("-", "_")
    dataset_name = f"{dataset_name}_{det_subset}_detections"

    IL_all_detections = IL_all_detections[det_subset_start:det_subset] # TODO this is debugging code, remove later

    hA_template.images = IL_all_detections
    hA_template.save(output_path / f"{dataset_name}_tiled_hasty.json")

    # TODO this is debugging code
    try:
        fo.delete_dataset(dataset_name)
    except:
        logger.warning(f"Dataset {dataset_name} does not exist")

    main(dataset_name=dataset_name,
         analysis_date=analysis_date,
         output_path=output_path,
         # df_detections=df_detections,
         hA_template=hA_template,
         anno_key=dataset_name)
