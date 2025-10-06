"""
convert Hasty annotations which are most probably on a jpg to the geotiff coordinates

To do this the geotiff are required to have the same name as the jpg but with .tiff ending

TODO: use world files instead

"""
from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.util.converter import hasty_to_shp
from active_learning.util.projection import get_geotransform, pixel_to_world_point, get_orthomosaic_crs, \
    convert_jpeg_to_geotiff_coords
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file, PredictedImageLabel
import geopandas as gpd




if __name__ == '__main__':
    base_path = Path(f'/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/Fer_FCD01-02-03_tiles')
    tiff_tile_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/Fer_FCD01-02-03_tiles")

    analysis_date = "2025_04_23"
    output_path = Path(base_path) / "output"
    dataset_name = f"eal_{analysis_date}_{base_path.name}_review"
    # replace '-' with '_'
    dataset_name = dataset_name.replace("-", "_")
    hA_reference = hA_from_file(hasty_correct_file)
    # hA_reference.images = hA_reference.images[:1]
    # TODO add the two extra keypoint Classes

    prefixes_ready_to_analyse = [
        "flo_",
        "fer_fni03_04_19122021",
        "fer_fpe09_18122021",
        "fer_fnd02_19122021",
        "fer_fef01_02_20012023",
        "fer_fna01_02_20122021",
        "fer_fnj01_19122021",
    ]

    # configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')
    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp')
    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy')
    configs_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy_uncorrected')

    base_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data")
    target_images_path = base_path / "2025_07_10_images_final"
    visualisation_path = Path("/Users/christian/data/training_data/2025_09_19_orthomosaic_data/visualisation")
    # visualisation_path = None


    for dataset_correction_config in (f for f in configs_path.glob("*_config.json") if not f.name.startswith("._")):
        if not any(dataset_correction_config.name.lower().startswith(p) for p in prefixes_ready_to_analyse):
            logger.info(f"Skipping {dataset_correction_config} as it does not match any of the prefixes")
            continue
        try:
            config = GeospatialDatasetCorrectionConfig.load(dataset_correction_config)



        hA_reference = HastyAnnotationV2.from_file(hasty_correct_file)

        gdf_annoation = hasty_to_shp(tif_path=base_path, hA_reference=hA_reference)

        gdf_annoation.to_file(filename = output_path / "annotation.geojson", driver="GeoJSON")