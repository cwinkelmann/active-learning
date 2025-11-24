"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
from time import sleep

import shapely
from loguru import logger
from pathlib import Path
from typing import Optional

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.reconstruct_hasty_annotation_cvat import cvat2hasty, download_cvat_annotations, foDataset2Hasty, \
    determine_changes
from active_learning.types.Exceptions import TooManyLabelsError, NoLabelsError, NoChangesDetected
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import hasty_to_shp
from active_learning.util.hit.geospatial import batched_geospatial_correction_download, merged_corrected_annotations
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabel




if __name__ == "__main__":

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected/')
    # prefixes_ready_to_analyse = [
        # "flo_",
        # "fer_fni03_04_19122021",
        # "fer_fpe09_18122021",
        # "fer_fnd02_19122021",
        # "fer_fef01_02_20012023",
        # "fer_fna01_02_20122021",
        # "fer_fnj01_19122021",
        # "fer_fpm05_24012023",
        # "fer_fe01_02_20012023",
        # "fer_fwk01_20122021",
        # "fer_fe01_02_20012023",
        # "fer_fnc01_19122021",

        # "fer_fwk02", # TODO not yet corrected

        # "isa_isvp01_27012023",
        # "isa_isvi01_27012023",

        # "isa_isvb04_27012023",
        # "isa_ispf12_13_16122021",
        # "ispvr04_17122021",

        # "ispp01_18012023",
        # "iscwn02_18012023_config",
        # "iscwn03_18012023_config",
        # "isvb04_27012023",
        # "iscas01_08012023",
        # "iscna01_02_iscnb01_02_21012023",
    # ]


    # prefixes_ready_to_analyse = [
    #     "isvb01_27012023",
    #     "isbb01_22012023"
    # ]
    # configs_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected/') # THis looks like I got a dataloss


    # prefixes_ready_to_analyse = [
    #     "fpe01",
    #     "fpe02",
    #     "fpe03",
    #     "isa_ispvr04_17122021",
    #     # "isvb01_27012023", # these don't work
    #     # "isbb01_22012023" # these don't work
    # ]
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting_2/Analysis_of_counts/all_drone_deploy_uncorrected/')

    island_short_filter = "Isa"
    island_full_name = "Isabela"
    prefixes_ready_to_analyse = None
    prefixes_ready_to_analyse = [
        "mar_mnw03_07122021"
        # "iscr02_26012023",
        # "iseb05_19012023",
        # "iseb03_19012023",
    ]
    # configs_path = Path(f"/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/correction_run_{island_full_name}")
    configs_path = Path(f"/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Analysis_of_counts_2025_11_20/all_drone_deploy_uncorrected")
    dataset_correction_configs = list(f for f in configs_path.glob("*_config.json") if not f.name.startswith("._"))
    logger.info(f"Found: {len(dataset_correction_configs)} configs in {configs_path}")

    base_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11")
    # visualisation_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_08_orthomosaic_data/visualisation")
    visualisation_path = None
    target_images_path = base_path / "unzipped_images"
    annotations_to_update = Path(
        "/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11/2025_11_12_labels_debug.json")

    for dataset_correction_config in dataset_correction_configs:

        if prefixes_ready_to_analyse is not None and not any(dataset_correction_config.name.lower().startswith(p) for p in prefixes_ready_to_analyse):
            logger.debug(f"Skipping {dataset_correction_config} as it does not match any of the prefixes")
            continue

        try:
            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)

            logger.info(f"Processing {dataset_correction_config} for dataset {config.dataset_name}")

            correction_config_path = batched_geospatial_correction_download(config, hA_to_update=annotations_to_update) # , configs_path=configs_path)

            config.hasty_reference_annotation_path = annotations_to_update
            correction_config = GeospatialDatasetCorrectionConfig.load(correction_config_path)

            correction_config.hasty_reference_annotation_path = annotations_to_update
            images, uncorrected_images = merged_corrected_annotations(config=correction_config,
                                                                      configs_path=configs_path,
                                                                      target_images_path=target_images_path,
                                                                      visualisation_path=visualisation_path)

            logger.info(f"Processed {correction_config_path} config")


        except NoLabelsError as e:
            logger.warning(f"No labels found for {dataset_correction_config}, {e} skipping")
            sleep(5)
        except NoChangesDetected as e:
            logger.warning(f"No changes detected for {dataset_correction_config}, {e} skipping")

        except ValueError as e:
            if e.__str__().startswith("Dataset has no annotation run key "):
                logger.error(f"No annotation run found for {dataset_correction_config}, {e} . That should not happend if that was uploaded to cvat, skipping")
            else:
                logger.error(f"Unexpected error with {dataset_correction_config}, {e}")

        except AttributeError as e:
            if e.__str__().startswith("'NoneType' object has no attribute 'backend' "):
                logger.error(f"No annotation run found for {dataset_correction_config}, {e} . That should not happend if that was uploaded to cvat, skipping")
            else:
                logger.error(f"Unexpected error with {dataset_correction_config}, {e}")



