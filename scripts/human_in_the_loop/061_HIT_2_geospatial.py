"""
HUMAN IN THE LOOP Download the images from the CVAT server and review them then reproject them back to original images
coordinates

Take prediction we don't have a ground truth for and double check if the prediction is right
Then prepare the output for another training round

"""
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
from active_learning.util.hit.geospatial import batched_geospatial_correction_download
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabel




if __name__ == "__main__":
    # report_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021/FLPC03_correction_config.json")
    # report_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy/Fer_FNA01_02_20122021_ds_correction_config.json")
    # dataset_correction_config = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy/flo_flbb01_28012023_counts_config.json")
    # dataset_correction_config = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy/flo_flpc06_22012021_counts_config.json")
    #
    # config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)
    # r = main(config)
    #
    # logger.info(f"Processed {r} config")

    prefixes_ready_to_analyse = [
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
        "fer_fnc01_19122021",

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
    ]
    # prefixes_ready_to_analyse = None

    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected/')
    hA_to_update = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_09_28_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_3.json")

    annotations_to_update = Path(
        "/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11/2025_11_09_labels_hn.json")

    for dataset_correction_config in (f for f in configs_path.glob("*_config.json") if not f.name.startswith("._")):

        if prefixes_ready_to_analyse is not None and not any(dataset_correction_config.name.lower().startswith(p) for p in prefixes_ready_to_analyse):
            logger.debug(f"Skipping {dataset_correction_config} as it does not match any of the prefixes")
            continue

        try:
            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)

            # config.dataset_name = "isa_isvp01_27012023_counts"

            logger.info(f"Processing {dataset_correction_config} for dataset {config.dataset_name}")
            r = batched_geospatial_correction_download(config, hA_to_update=annotations_to_update) # , configs_path=configs_path)
    
            logger.info(f"Processed {r} config")

        except NoLabelsError as e:
            logger.warning(f"No labels found for {dataset_correction_config}, {e} skipping")

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



