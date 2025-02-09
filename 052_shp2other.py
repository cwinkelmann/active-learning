"""
convert a shapefile to geojson and csv

"""
from PIL import Image
from loguru import logger
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from pyproj import Transformer

def convert_shapefile2usable(shapefile_path: Path):
    parent_path = shapefile_path.parent
    # Convert to GeoJSON
    geojson_path = parent_path / "output.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    # Convert to CSV (geometry as WKT format)
    csv_path = parent_path / "output.csv"
    gdf.to_csv(csv_path, index=False)

    print(f"GeoJSON saved to: {geojson_path}")
    print(f"CSV saved to: {csv_path}")


def convert_image(image_path) -> Path:
    """
    convert an geospatial image to a jpg
    :param image_path:
    :return:
    """
    jpg_path = image_path.with_suffix(".jpg")
    with Image.open(image_path) as img:
        img.convert("RGB").save(jpg_path, "JPEG", quality=95)

    return jpg_path


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


if __name__ == "__main__":
    # Load the Shapefile
    shapefile_path = Path("/Users/christian/data/Manual Counting/Esp_EGB04_12012021/Esp_EGB04_12012021 counts.shp")
    image_path = Path("/Users/christian/data/Manual Counting/Esp_EGB04_12012021.tif")
    gdf = gpd.read_file(shapefile_path)

    convert_shapefile2usable(shapefile_path)
    jpg_path = convert_image(image_path)

    gdf_pixel = convert_gdf_to_jpeg_coords(gdf, tiff_path=image_path)

    gdf_pixel

    gdf_geo = convert_jpeg_to_geotiff_coords(gdf_pixel, tiff_path=image_path, jpg_path=jpg_path)

    # gdf_geo coordinates should be roughly the same as gdf coordinates
    gdf_geo