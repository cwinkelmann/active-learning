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

from active_learning.types.Exceptions import ProjectionError
from active_learning.util.geospatial_image_manipulation import cut_geospatial_raster_with_grid_gdal
from active_learning.util.projection import world_to_pixel





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


    def slice_annotations(self, points_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        project the points into the local pixel coordinates of the raster cell
        :param points_gdf:
        :return:
        """
        grid_index_col = "grid_id"
        # assign each point to a grid cell
        # Step 1: Assign each point to a grid cell using spatial join
        # This will match each point to the grid cell that contains it
        points_in_grid = gpd.sjoin(points_gdf, grid_gdf, how="left", predicate="within")

        # TODO check if any row "image_name_right" is None
        if points_in_grid["image_name_right"].isnull().any():
            logger.error(f"some points could not be assigned to a grid cell: {points_in_grid['image_name_right'].isnull().sum()}, THIS IS DUE TO THE fact someone messed up the shapefile or orhtomosaic projections")
            raise ProjectionError("THere is a problem with the projection")

        points_in_grid.rename(columns={"index_right": grid_index_col}, inplace=True)

        # Create a mapping from grid index to grid geometry
        grid_geom_map = grid_gdf["geometry"].to_dict()

        # Add the grid geometry to the points DataFrame
        points_in_grid["grid_geometry"] = points_in_grid[grid_index_col].map(grid_geom_map)

        # Convert all points from world to pixel coordinates
        points_in_grid["pixel_x"], points_in_grid["pixel_y"] = zip(
            *points_in_grid.geometry.apply(lambda point: world_to_pixel(self.geo_transform, point.x, point.y)))

        # Step 3: Calculate local pixel coordinates relative to the grid cell origin

        if "grid_geometry" in points_in_grid.columns:
            def calculate_local_coords(row):
                grid_cell = grid_gdf.loc[row[grid_index_col]]
                grid_minx, grid_miny, grid_maxx, grid_maxy = grid_cell.geometry.bounds

                # Convert the grid cell corners to pixel coordinates
                min_pixel_x, max_pixel_y = world_to_pixel(self.geo_transform, grid_minx,
                                                          grid_miny)  # Bottom-left corner
                max_pixel_x, min_pixel_y = world_to_pixel(self.geo_transform, grid_maxx, grid_maxy)  # Top-right corner

                # Calculate height of the grid cell in pixels
                grid_height_pixels = max_pixel_y - min_pixel_y

                # Calculate local coordinates with bottom-left origin
                local_x = row["pixel_x"] - min_pixel_x  # Distance from left edge

                # For Y, we need to flip the orientation:
                # Instead of measuring from the top, measure from the bottom
                local_y = -1 * (min_pixel_y - row["pixel_y"])  # Distance from bottom edge

                return local_x, local_y

            points_in_grid["local_pixel_x"], points_in_grid["local_pixel_y"] = zip(
                *points_in_grid.apply(calculate_local_coords, axis=1))

        return points_in_grid

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

        slices = cut_geospatial_raster_with_grid_gdal(raster_path=self.base_path.joinpath(self.image_name),
                                             grid_gdf=self.grid,
                                             output_dir=self.output_dir)
        self.slices = slices
        return slices



def get_geospatial_sliced_path(base_path, x_size, y_size):
    if x_size is not None and y_size is not None:
        sliced_path = base_path.joinpath(f"sliced_{x_size}_{y_size}px")
    else:
        sliced_path = base_path.joinpath(f"sliced")
    return sliced_path
