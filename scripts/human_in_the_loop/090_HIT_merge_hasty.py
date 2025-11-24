"""
Get the intermediate hasty formatted annotatioes an merge them in to an already existing files
"""

import shutil
from matplotlib import pyplot as plt

from pathlib import Path

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.types.Exceptions import NoLabelsError
from loguru import logger

from active_learning.util.hit.geospatial import merged_corrected_annotations
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage
from com.biospheredata.types.status import LabelingStatus
from com.biospheredata.visualization.visualize_result import visualise_hasty_annotation




if __name__ == "__main__":

    prefixes_ready_to_analyse = [
        # "flo_",
        # "fer_fni03_04_19122021",
        # "fer_fpe09_18122021",
        # "fer_fnd02_19122021",
        # "fer_fef01_02_20012023",
        # "fer_fna01_02_20122021",
        # "fer_fnj01_19122021",
        # "fer_fwk01_20122021",
        # "fer_fe01_02_20012023",
        # "fer_fwk02_03_20_21122021",
        # "fer_fnc01_19122021",
        # "fer_fnc01_19122021",
        # "isa_ispvr04_17122021",
        # "ispvr04_17122021",
        # "ispvr04_17122021",
        "isvb01_27012023"
    ]
    # prefixes_ready_to_analyse = None

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp')

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp_Isabela')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp_corr_Isabela')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting_2/Analysis_of_counts/all_drone_deploy_uncorrected/')

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp_Fernandina')
    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp_Floreana')

    base_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11")

    target_images_path = base_path / "unzipped_images"
    # visualisation_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_08_orthomosaic_data/visualisation")
    visualisation_path = None

    annotations_to_update = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11/2025_11_12_labels.json")

    hA = HastyAnnotationV2.from_file(annotations_to_update)

    for dataset_correction_config in (f for f in configs_path.glob("*_config.json") if not f.name.startswith("._")):
        if prefixes_ready_to_analyse is not None and not any(dataset_correction_config.name.lower().startswith(p) for p in prefixes_ready_to_analyse):
            logger.info(f"Skipping {dataset_correction_config} as it does not match any of the prefixes")
            continue
        try:
            logger.info(f"Processing {dataset_correction_config}")

            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)

            config.hasty_reference_annotation_path = annotations_to_update

            images, uncorrected_images = merged_corrected_annotations(config, target_images_path, visualisation_path)
            # hA.images.extend(images)
            # hA.images.extend(uncorrected_images)

            logger.info(f"Processed {config} config")
        except NoLabelsError as e:
            logger.warning(f"No labels found for {dataset_correction_config}, {e} skipping")
        except ValueError as e:
            if e.__str__().startswith("Dataset has no annotation run key "):
                logger.error \
                    (f"No annotation run found for {dataset_correction_config}, {e} . That should not happend if that was uploaded to cvat, skipping")

        print(hA.dataset_statistics())
    print(hA.dataset_statistics())


     # hA.save(base_path / "2025_09_19_orthomosaic_data_combined_corrections_5.json")
    # logger.info(f"Saved combined annotations to {base_path / '2025_09_19_orthomosaic_data_combined_corrections_5.json'}")