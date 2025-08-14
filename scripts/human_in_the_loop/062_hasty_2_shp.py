"""
convert Hasty annotations which are most probably on a jpg to the geotiff coordinates


"""
from pathlib import Path

from active_learning.util.converter import hasty_to_shp
from active_learning.util.projection import get_geotransform, pixel_to_world_point, get_orthomosaic_crs, \
    convert_jpeg_to_geotiff_coords
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file, PredictedImageLabel
import geopandas as gpd

# def hasty_to_shp(base_path: Path, hA_reference: HastyAnnotationV2, suffix=".tif"):
#     # TODO look at this: convert_jpeg_to_geotiff_coords from playground/052_shp2other.py
#     # convert_jpeg_to_geotiff_coords()
#     data = []
#     if len(hA_reference.images) == 0:
#         raise ValueError("No images in Hasty Annotation")
#
#     for img in hA_reference.images:
#         img_name = Path(img.image_name).with_suffix(suffix=suffix)
#         geo_transform = get_geotransform(base_path / img_name)
#         crs = get_orthomosaic_crs(base_path / img_name)
#
#         for label in img.labels:
#
#             # get the pixel coordinates
#             x, y = label.incenter_centroid.x, label.incenter_centroid.y
#             # get the world coordinates
#             p = pixel_to_world_point(geo_transform, x, y)
#             # set the new coordinates
#             # TODO add some more metadata
#             if isinstance(label, PredictedImageLabel):
#                 score = label.score
#             else:
#                 score = None
#             data.append({"img_name": img_name, "label": label.class_name ,"score": score, "geometry": p})
#
#     return gpd.GeoDataFrame(data, crs=crs)




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