import typing

from loguru import logger
from pathlib import Path
import osgeo.gdal as gdal
import geopandas as gpd

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel


def world_to_pixel(geo_transform, x, y):
    """
    Converts georeferenced coordinates to pixel coordinates.

    Parameters:
    - geo_transform: Geotransform tuple of the dataset.
    - x: Georeferenced x-coordinate (e.g., longitude).
    - y: Georeferenced y-coordinate (e.g., latitude).

    Returns:
    - (pixel_x, pixel_y): Corresponding pixel coordinates.
    """
    originX = geo_transform[0]
    pixelWidth = geo_transform[1]
    originY = geo_transform[3]
    pixelHeight = geo_transform[5]

    pixel_x = int((x - originX) / pixelWidth)
    pixel_y = int((y - originY) / pixelHeight)

    return pixel_x, pixel_y

def pixel_to_world(geo_transform, pixel_x, pixel_y):
    """
    Converts pixel coordinates to georeferenced coordinates.

    Parameters:
    - geo_transform: Geotransform tuple of the dataset.
    - pixel_x: Pixel x-coordinate (column).
    - pixel_y: Pixel y-coordinate (row).

    Returns:
    - (x, y): Corresponding georeferenced coordinates.
    """
    x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return x, y


def get_geotransform(orthomsoaic_path: Path):
    """
    :param orthomsoaic_path:
    :return:
    """
    if not Path.is_file(orthomsoaic_path):
        raise FileNotFoundError(f"File {orthomsoaic_path} does not exist")
    orthophoto_raster = gdal.Open(orthomsoaic_path)
    geo_transform = orthophoto_raster.GetGeoTransform()
    return geo_transform


def local_coordinates_to_wgs84(georeferenced_tiff_path: Path,
                               annotations: typing.List[PredictedImageLabel],
                               type_of_coordinates: AnnotationType = AnnotationType.KEYPOINT):
    """
    Converts annotation coordinates from local pixel coordinates to georeferenced WGS84 coordinates.

    Parameters:
    - georeferenced_tiff_path: Path to the georeferenced TIFF file.
    - annotations: List of PredictedImageLabel objects with local pixel coordinates.

    Returns:
    - List of annotations with updated world coordinates.
    """
    logger.info(f"Trying to open this file: {georeferenced_tiff_path}")

    if not Path.is_file(Path(georeferenced_tiff_path)):
        raise ValueError(f"File {georeferenced_tiff_path} is not there.")

    # Open the raster and get the geotransformation
    src = gdal.Open(str(georeferenced_tiff_path))
    geo_transform = src.GetGeoTransform()

    # Convert annotation coordinates
    for annotation in annotations:
        ## TODO make this work for bounding boxes, segmentation masks and points
        if type_of_coordinates == AnnotationType.KEYPOINT:
            for kp in annotation.keypoints:
                kp.x, kp.y = pixel_to_world(geo_transform, kp.x, kp.y)

        elif type_of_coordinates == AnnotationType.BOUNDING_BOX:
            raise NotImplementedError("Not implemented yet")

        elif type_of_coordinates == AnnotationType.POLYGON:
            raise NotImplementedError("Not implemented yet")

    return annotations
