import shutil

from pathlib import Path

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.types.Exceptions import NoLabelsError
from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage
from com.biospheredata.types.status import LabelingStatus
from com.biospheredata.visualization.visualize_result import visualise_hasty_annotation


def main(config: GeospatialDatasetCorrectionConfig,
         target_images_path: Path,
         visualisation_path: Path = None):
    """
    Merge the hasty prediction and the corrected hasty annotation into the reference annotation file
    :param config:
    :param target_images_path:
    :param visualisation_path:
    :return:
    """


    hA_prediction_path = config.hasty_intermediate_annotation_path
    if hA_prediction_path is None:
        hA_prediction_path = configs_path / f"{config.dataset_name}_intermediate_hasty.json" # in the current folder
        if hA_prediction_path.exists():
            logger.info(f"Guessing {hA_prediction_path} as hasty intermediate annotation path successful")
        else:
            raise NoLabelsError("hA_prediction_path is None")

    hA_corrected_path = config.output_path / f"{config.dataset_name}_corrected_intermediate_hasty.json"

    if not hA_corrected_path.exists():
        raise NoLabelsError(f"hA_corrected_path {hA_corrected_path} does not exist")

    hA_prediction = HastyAnnotationV2.from_file(file_path=hA_prediction_path)
    hA_correction = HastyAnnotationV2.from_file(file_path=hA_corrected_path)
    hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)

    assert len(hA_prediction.images) == len(hA_correction.images)
    # remove "_counts" from end of dataset name if present
    dataset_name = config.dataset_name.replace("_counts", "")
    hA_reference.delete_dataset(dataset_name=dataset_name)

    images = []
    uncorrected_images = []

    for pred_image in hA_prediction.images:
        logger.info(f"Processing {pred_image.image_name}, dataset {dataset_name}")
        # pred_image = hA_prediction.images[i]
        # corr_image = hA_correction.images[i]

        # be careful to match the image ids
        corr_image = hA_correction.get_image_by_id(pred_image.image_id)
        pred_image = AnnotatedImage(**pred_image.model_dump(), dataset_name=dataset_name, image_status=LabelingStatus.COMPLETED)

        if visualisation_path is not None and len(corr_image.labels) > 0:

            visualise_hasty_annotation(image=corr_image, images_path=config.image_tiles_path / "converted_tiles",
                                       output_path= visualisation_path,
                                       title = f"Prediction {dataset_name}, {pred_image.image_name}",
                                       show=False)

        pred_image.dataset_name = f"ha_{dataset_name}"
        uncorrected_images.append(pred_image)

        hA_reference.images.append(pred_image)
        target_path = target_images_path / pred_image.dataset_name
        target_path.mkdir(parents=True, exist_ok=True)

        if not (config.image_tiles_path / "converted_tiles" / pred_image.image_name).exists():
            shutil.copy(config.image_tiles_path / "converted_tiles" / pred_image.image_name,
                        target_path / pred_image.image_name)

        corr_image.dataset_name = f"ha_corrected_{dataset_name}"
        hA_reference.images.append(corr_image)
        images.append(corr_image)

        target_path = target_images_path / corr_image.dataset_name
        target_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(config.image_tiles_path / "converted_tiles" / corr_image.image_name,
                    target_images_path / corr_image.dataset_name / corr_image.image_name)

    # update the reference annotation file

    # hA_updated_path = config.hasty_reference_annotation_path.stem + f"_updated_with_{dataset_name}.json"

    # hA_reference.to_file(hA_updated_path)
    return images, uncorrected_images

if __name__ == "__main__":

    prefixes_ready_to_analyse = [
        # "flo_",
        # "fer_fni03_04_19122021",
        # "fer_fpe09_18122021",
        # "fer_fnd02_19122021",
        # "fer_fef01_02_20012023",
        "fer_fna01_02_20122021",
        # "fer_fnj01_19122021",
    ]

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')
    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected')
    base_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data")
    target_images_path = base_path / "2025_07_10_images_final"
    visualisation_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/visualisation")
    # visualisation_path = None
    annotations_to_update = Path("/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json")
    annotations_to_update = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_09_19/2025_09_19_orthomosaic_data_combined_corrections_2.json")

    hA = HastyAnnotationV2.from_file(annotations_to_update)

    for dataset_correction_config in (f for f in configs_path.glob("*_config.json") if not f.name.startswith("._")):
        if not any(dataset_correction_config.name.lower().startswith(p) for p in prefixes_ready_to_analyse):
            logger.info(f"Skipping {dataset_correction_config} as it does not match any of the prefixes")
            continue
        try:

            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)

            config.hasty_reference_annotation_path = annotations_to_update

            images, uncorrected_images = main(config, target_images_path, visualisation_path)
            hA.images.extend(images)
            # hA.images.extend(uncorrected_images)

            logger.info(f"Processed {config} config")
        except NoLabelsError as e:
            logger.warning(f"No labels found for {dataset_correction_config}, {e} skipping")
        except ValueError as e:
            if e.__str__().startswith("Dataset has no annotation run key "):
                logger.error \
                    (f"No annotation run found for {dataset_correction_config}, {e} . That should not happend if that was uploaded to cvat, skipping")


    hA.save(base_path / "2025_09_19_orthomosaic_data_combined_corrections_3.json")
    logger.info(f"Saved combined annotations to {base_path / '2025_09_19_orthomosaic_data_combined_corrections_3.json'}")