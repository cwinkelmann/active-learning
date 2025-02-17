"""
slice images into smaller parts
"""
import numpy as np
import osgeo.gdal
import typing

from pathlib import Path

import os
from time import sleep
import math
from loguru import logger
from osgeo import gdal
import geopandas as gpd

from active_learning.util.geospatial_image_manipulation import cut_geospatial_raster_with_grid_gdal
from active_learning.util.projection import world_to_pixel


# Function to process and potentially save a slice
def process_slice(dataset, x_offset, y_offset, x_size, y_size, output_path,
                  threshold=90, driver: osgeo.gdal.Driver = None,
                  options: typing.List[str] = []):
    """
    Processes a raster slice and saves it if it contains less than the specified
    percentage of black pixels.

    Parameters:
    - x_offset, y_offset: Pixel offsets for the slice.
    - x_size, y_size: Dimensions of the slice in pixels.
    - output_path: File path to save the slice.
    - threshold: Percentage threshold to determine if a slice is mostly black.
    """
    # Read the slice (window) into a NumPy array
    band = dataset.GetRasterBand(1)  # Assuming a single-band raster
    slice_data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)
    num_bands = 3
    # Analyze the slice for black pixels
    black_pixel_count = np.sum(slice_data == 0)
    total_pixels = slice_data.size
    black_pixel_percentage = (black_pixel_count / total_pixels) * 100

    slice_data = dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

    # Conditional writing based on the black pixel percentage
    if black_pixel_percentage < threshold:
        # Create a new raster dataset to save the slice
        # driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_path, x_size, y_size, num_bands, gdal.GDT_Byte, options=options)
        out_dataset.SetGeoTransform((
            dataset.GetGeoTransform()[0] + x_offset * dataset.GetGeoTransform()[1],
            dataset.GetGeoTransform()[1],
            0,
            dataset.GetGeoTransform()[3] + y_offset * dataset.GetGeoTransform()[5],
            0,
            dataset.GetGeoTransform()[5]
        ))
        # Write each band to the output dataset
        for band_index in range(num_bands):
            out_band = out_dataset.GetRasterBand(band_index + 1)
            out_band.WriteArray(slice_data[band_index, :, :])
            out_band.FlushCache()
        logger.info(f"Slice saved to {output_path}")
    else:
        logger.info("Slice predominantly black; not saved.")
    return output_path



class GeoSlicer():
    """
    Helper to slice geospational rasters into smaller parts
    """
    def __init__(self, base_path: Path, image_name: str, grid: gpd.GeoDataFrame, output_dir: Path):
        self.sliced_paths: typing.List[Path] = []
        self.slice_dict: typing.List[typing.Dict[Path, Path]] = []
        self.image_names: typing.List[Path]
        self.sliced_paths: typing.List[Path]
        self.grid = grid
        self.output_dir = output_dir

        self.image_name = image_name
        self.base_path = base_path

        full_image_path = self.base_path.joinpath(image_name)
        if not full_image_path.exists():
            raise FileNotFoundError(f"File {full_image_path} does not exist")
        image_size = os.path.getsize(full_image_path)  ## gdal.Open is not realizing when the file is missing. So do this before
        logger.info(f"size of the image: {image_size}")
        orthophoto_raster = gdal.Open(full_image_path)
        self.geo_transform = orthophoto_raster.GetGeoTransform()
        self.orthophoto_raster = orthophoto_raster

    def set_object_locations(self, object_locations: gpd.GeoDataFrame):
        self.object_locations = object_locations


    def slice_annotations(self, points_gdf: gpd.GeoDataFrame):
        """
        project annotation in a geospatial format to pixel coordinates
        :param points_gdf:
        :return:
        """

        # Convert all points from world to pixel coordinates
        points_gdf["pixel_x"], points_gdf["pixel_y"] = zip(
            *points_gdf.geometry.apply(lambda point: world_to_pixel(self.geo_transform, point.x, point.y)))

        projected_points_path = self.output_dir / "points_with_pixels.csv"

        # Save transformed points as a CSV or GeoJSON
        points_gdf.to_csv(projected_points_path, index=False)

        return projected_points_path

    def slice_very_big_raster(self):
        """
        slice the very big geotiff into smaller parts which can be handled by any model
        https://www.youtube.com/watch?v=H5uQ85VXttg

        :param base_path:
        :param image_name:
        :return:
        @param base_path:
        @param image_name:
        @param FORCE_UPDATE:
        @param y_size:
        @param x_size:
        """

        return cut_geospatial_raster_with_grid_gdal(raster_path=self.base_path.joinpath(self.image_name),
                                             grid_gdf=self.grid,
                                             output_dir=self.output_dir)




def get_geospatial_sliced_path(base_path, x_size, y_size):
    if x_size is not None and y_size is not None:
        sliced_path = base_path.joinpath(f"sliced_{x_size}_{y_size}px")
    else:
        sliced_path = base_path.joinpath(f"sliced")
    return sliced_path
