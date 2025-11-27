"""
convert Hasty annotations which are most probably on a jpg to the geotiff coordinates

To do this the geotiff are required to have the same name as the jpg but with .tiff ending

TODO: use world files instead

"""
from pathlib import Path

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
    hasty_correct_file = Path("/Users/christian/PycharmProjects/hnee/active_learning/scripts/inferencing/Fer_FCD01-02-03_tiles/object_crops/eal_2025_04_23_Fer_FCD01_02_03_tiles_review_hasty_corrected.json")
    hA_reference = hA_from_file(hasty_correct_file)
    # hA_reference.images = hA_reference.images[:1]
    # TODO add the two extra keypoint Classes

    hA_reference

    gdf_annoation = hasty_to_shp(tif_path=base_path, hA_reference=hA_reference)

    gdf_annoation.to_file(filename = output_path / "annotation.geojson", driver="GeoJSON")