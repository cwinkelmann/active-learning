import rasterio
import typing
import rasterio
from rasterio.transform import rowcol, xy
from pyproj import Transformer
from loguru import logger
from pathlib import Path
import osgeo.gdal as gdal
import geopandas as gpd
from PIL import Image
from loguru import logger

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel

def convert_gdf_to_jpeg_coords(gdf: gpd.GeoDataFrame, tiff_path: Path) -> gpd.GeoDataFrame:
    """
    Convert a GeoDataFrame with georeferenced coordinates to local JPEG image pixel coordinates.

    Parameters:
    - gdf (geopandas.GeoDataFrame): GeoDataFrame with geometries.
    - tiff_path (str): Path to the GeoTIFF file (to get georeferencing and CRS).
    - jpg_path (str): Path to the JPEG file (to get dimensions).

    Returns:
    - gdf_pixel (geopandas.GeoDataFrame): New GeoDataFrame with local pixel coordinates.
    """
    # Open the TIFF file to get its CRS and transform
    with rasterio.open(tiff_path) as dataset:
        image_crs = dataset.crs  # Get the CRS of the image
        transform = dataset.transform  # Get the affine transform
        tiff_width, tiff_height = dataset.width, dataset.height  # Get TIFF dimensions

    # Assert the GeoDataFrame CRS matches the image CRS
    assert gdf.crs == image_crs, f"CRS mismatch: GDF {gdf.crs} != TIFF {image_crs}"

    # Convert each point in the GeoDataFrame to TIFF pixel coordinates
    pixel_coords = []
    for geom in gdf.geometry:
        if geom is None:
            pixel_coords.append((None, None))
            continue

        # Convert georeferenced coordinates to pixel coordinates in TIFF
        row, col = rowcol(transform, geom.x, geom.y)
        pixel_coords.append((col, row))  # (col, row) = (x, y) in TIFF

    # Create a new DataFrame with pixel coordinates in TIFF space
    gdf_pixel = gdf.copy()
    gdf_pixel["local_x"], gdf_pixel["local_y"] = zip(*pixel_coords)

    return gdf_pixel


def convert_jpeg_to_geotiff_coords(gdf_pixel: gpd.GeoDataFrame, tiff_path: Path, jpg_path: Path) -> gpd.GeoDataFrame:
    """
    Convert local JPEG image pixel coordinates back to georeferenced coordinates in a GeoTIFF.

    Parameters:
    - gdf_pixel (geopandas.GeoDataFrame): GeoDataFrame with pixel coordinates.
    - tiff_path (str): Path to the reference GeoTIFF file.
    - jpg_path (str): Path to the JPEG file (to get dimensions).

    Returns:
    - gdf_geo (geopandas.GeoDataFrame): New GeoDataFrame with georeferenced coordinates.
    """
    # Read the JPEG dimensions
    with Image.open(jpg_path) as img:
        jpg_width, jpg_height = img.size  # (width, height)

    # Open the TIFF file to get its CRS and transform
    with rasterio.open(tiff_path) as dataset:
        image_crs = dataset.crs  # Get the CRS of the image
        transform = dataset.transform  # Get the affine transform
        tiff_width, tiff_height = dataset.width, dataset.height  # Get TIFF dimensions

    # Scale JPEG pixel coordinates to TIFF pixel coordinates
    gdf_pixel["tiff_x"] = (gdf_pixel["local_x"] / jpg_width) * tiff_width
    gdf_pixel["tiff_y"] = (gdf_pixel["local_y"] / jpg_height) * tiff_height

    # Convert TIFF pixel coordinates back to georeferenced coordinates
    geo_coords = [xy(transform, row, col) for col, row in zip(gdf_pixel["tiff_x"], gdf_pixel["tiff_y"])]
    lon, lat = zip(*geo_coords)

    # Create a new DataFrame with georeferenced coordinates
    gdf_geo = gdf_pixel.copy()
    gdf_geo["longitude"], gdf_geo["latitude"] = lon, lat
    gdf_geo = gdf_geo.set_geometry(gpd.points_from_xy(gdf_geo["longitude"], gdf_geo["latitude"]))
    gdf_geo.set_crs(image_crs, inplace=True)

    return gdf_geo


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

def project_gdfcrs(gdf: gpd.GeoDataFrame, orthomosaic_path: Path) -> gpd.GeoDataFrame:
    """
    Project a GeoDataFrame to the CRS of the orthomosaic.

    Parameters:
    - gdf (geopandas.GeoDataFrame): GeoDataFrame with geometries.
    - orthomosaic_path (str): Path to the orthomosaic GeoTIFF file.

    Returns:
    - gdf_proj (geopandas.GeoDataFrame): New GeoDataFrame with projected geometries.
    """
    # Open the orthomosaic to get its CRS
    with rasterio.open(orthomosaic_path) as dataset:
        ortho_crs = dataset.crs  # Get the CRS of the orthomosaic

    # Project the GeoDataFrame to the orthomosaic CRS
    gdf_proj = gdf.to_crs(ortho_crs)

    return gdf_proj