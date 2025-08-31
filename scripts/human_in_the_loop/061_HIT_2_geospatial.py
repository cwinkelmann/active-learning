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
from active_learning.types.Exceptions import TooManyLabelsError, NoLabelsError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.converter import hasty_to_shp
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, Keypoint, HastyAnnotationV2, \
    ImageLabel


def get_point_offset(corrected_label: ImageLabel,
                     hA_prediction_tiled: HastyAnnotationV2) -> (int, int):
    """
    Get the offset for the corrected label by comparing its current position to the original position

    :param hA_prediction_tiled:
    :param hA_prediction_tiled_corrected:
    :return:
    """
    for i, annotated_image in enumerate(hA_prediction_tiled.images):
        # read the mapping file which global coordinate
        for l in annotated_image.labels:
            if l.id == corrected_label.id:
                x_offset = corrected_label.incenter_centroid.x - l.incenter_centroid.x
                y_offset = corrected_label.incenter_centroid.y - l.incenter_centroid.y
                logger.info(f"Label {l.id} was moved by {x_offset}, {y_offset}")
                return x_offset, y_offset


    raise ValueError("Label not found")


def shift_keypoint_label(corrected_label: ImageLabel, hA_prediction: HastyAnnotationV2,
                         x_offset: Optional[int] = None, y_offset: Optional[int] = None):
    """

    :param corrected_label:
    :param hA_prediction:
    :param x_offset: move the label by this offset or delete if None
    :param y_offset: move the label by this offset or delete if None
    :return:
    """
    for i, annotated_image in enumerate(hA_prediction.images):
        # read the mapping file which global coordinate
        for l in annotated_image.labels:
            if l.id == corrected_label.id:
                if x_offset is None and y_offset is None:
                    hA_prediction.images[i].labels.remove(l)
                elif x_offset != 0 or y_offset != 0:
                    for kp in l.keypoints:
                        kp.x += int(x_offset)
                        kp.y += int(y_offset)
                else:
                    logger.info(f"Label {l.id} was not moved")




def main(config: GeospatialDatasetCorrectionConfig):


    hA_prediction_path = config.hasty_intermediate_annotation_path
    if hA_prediction_path is None:
        raise NoLabelsError("hA_prediction_path is None")
    hA_prediction = HastyAnnotationV2.from_file(file_path=hA_prediction_path)
    #
    # # the 256px crops
    # hA_prediction_tiled_path = output_path / f"{dataset_name}_tiled_hasty.json"
    # hA_prediction_tiled = HastyAnnotationV2.from_file(file_path=config.hA_prediction_tiled_path)

    hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)
    hA_reference_updated = hA_reference.copy(deep=True)
    view, dataset = download_cvat_annotations(dataset_name=config.dataset_name)

    hA_updated = foDataset2Hasty(hA_template=hA_prediction, dataset=dataset, anno_field="iguana")

    new_boxes = 0
    new_points = 0
    modified_annotated_image_names = []

    changes = determine_changes(hA_prediction, hA_updated)

    # TODO NOW two things need to happen:
    # analyse the changes. On the other hand that could happen later
    # create a new annotations from the changes and save everything
    hA_updated.save(config.output_path / f"{config.dataset_name}_corrected_intermediate_hasty.json")
    gdf_annoation = hasty_to_shp(tif_path=config.image_tiles_path, hA_reference=hA_updated)

    corrected_path = config.output_path / f"{config.dataset_name}_corrected_annotation.geojson"
    gdf_annoation.to_file(filename=corrected_path, driver="GeoJSON")
    logger.info(f"corrected file saved to : {corrected_path}")

    # Now we can project every annotation to the original orthomosaic coordinates
    for i, annotated_image in enumerate(hA_updated.images):
        point_existing = False

        for j, corrected_label in enumerate(annotated_image.labels):

            pass


    # logger.info(f"Updated {config.output_path / config.hasty_reference_annotation_name}")
    # corrected_full_path = config.output_path / f"{config.dataset_name}_hasty_corrected_1.json"
    # hA_reference_updated.save(corrected_full_path)
    # logger.info(f"Saved corrected hasty annotation to {corrected_full_path}")
    #
    # config.corrected_path = corrected_full_path
    #
    # config.save(report_path)
    # logger.info(f"Saved report config to {report_path}")

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

    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')

    for dataset_correction_config in (f for f in configs_path.glob("*_config.json") if not f.name.startswith("._")):
        try:
            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)
            r = main(config)
    
            logger.info(f"Processed {r} config")
        except NoLabelsError as e:
            logger.warning(f"No labels found for {dataset_correction_config}, {e} skipping")