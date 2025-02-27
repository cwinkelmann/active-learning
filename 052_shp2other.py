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