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
from active_learning.util.geospatial_image_manipulation import cut_geospatial_raster_with_grid_gdal, \
    create_regular_geospatial_raster_grid
from active_learning.util.projection import world_to_pixel

import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import Polygon


class ImageGrid(object):
    """ base class for creating a grid on top of images """



class GeoSpatialRasterGrid(ImageGrid):
    """
    Class for creating and managing geospatial raster grids.

    This class handles the creation of vector grids from raster data,
    with the ability to filter grids based on point data containment.
    """

    def __init__(self, raster_path=None):
        """
        Initialize the GeoSpatialRasterGrid with an optional raster path.

        Parameters:
            raster_path (Path, optional): Path to the raster image.
        """
        self.raster_path = raster_path
        self.grid_gdf = None
        self.crs = None
        self.transform = None
        self.bounds = None
        self.pixel_size_x = None
        self.pixel_size_y = None

        if raster_path:
            self._load_raster_metadata()

    def _load_raster_metadata(self):
        """Load metadata from the raster file."""
        with rasterio.open(self.raster_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds

            # Get pixel sizes (resolution in X and Y directions)
            self.pixel_size_x = self.transform.a  # Pixel width in world units
            self.pixel_size_y = abs(self.transform.e)  # Pixel height (absolute value)

    def _get_epsg_code(self):
        """Determine the EPSG code for the raster."""
        if self.crs is None:
            return None

        if self.crs.to_epsg() is None:
            # Handle specific UTM zones
            if "UTM zone 16S" in str(self.crs):
                return "EPSG:32716"
            elif "UTM zone 15S" in str(self.crs):
                return "EPSG:32715"
            else:
                # Return the WKT string if no specific handling is available
                return str(self.crs)
        else:
            return f"EPSG:{self.crs.to_epsg()}"

    def create_regular_grid(self, x_size, y_size, overlap_ratio=0.0):
        """
        Create a vector grid for the geospatial raster, slicing based on pixel sizes.

        this similar to         create_regular_geospatial_raster_grid(full_image_path=self.raster_path,
                                              x_size=x_size,
                                              y_size=y_size,
                                              overlap_ratio=0.0)

        Parameters:
            x_size (float): Width of each grid cell in pixels.
            y_size (float): Height of each grid cell in pixels.
            overlap_ratio (float): Overlap between tiles as a fraction (0 to 1).

        Returns:
            gpd.GeoDataFrame: The created grid as a GeoDataFrame.
        """
        if not self.raster_path:
            raise ValueError("Raster path not set. Either initialize with a path or set it with set_raster_path().")

        # Ensure raster metadata is loaded
        if self.transform is None:
            self._load_raster_metadata()

        # Convert to GeoDataFrame
        epsg_code = self._get_epsg_code()



        # Get bounds
        min_x, min_y, max_x, max_y = self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top

        # Compute tile sizes in world coordinates
        x_size_world = x_size * self.pixel_size_x
        y_size_world = y_size * self.pixel_size_y

        # Compute overlap in world coordinates
        x_overlap_world = x_size_world * overlap_ratio
        y_overlap_world = y_size_world * overlap_ratio

        # Compute step size (tile size minus overlap)
        step_x = x_size_world - x_overlap_world
        step_y = y_size_world - y_overlap_world

        grid_cells = []
        tiles = []
        image_name = Path(self.raster_path).stem

        # Generate grid based on pixel-based slicing
        for y in np.arange(min_y, max_y, step_y):
            for x in np.arange(min_x, max_x, step_x):
                # Define tile coordinates
                x1, y1 = x, y
                x2, y2 = x + x_size_world, y + y_size_world

                # Ensure tiles stay within bounds
                x2 = min(x2, max_x)
                y2 = min(y2, max_y)

                # Create polygon
                tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                tiles.append(tile_polygon)

                # Generate a unique tile name based on image name and coordinates
                coord_tile_name = f"{image_name}_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

                grid_cells.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "tile_name": coord_tile_name,
                    "image_name": image_name,
                })

        # Convert to GeoDataFrame
        epsg_code = self._get_epsg_code()
        self.grid_gdf = gpd.GeoDataFrame(data=grid_cells, geometry=tiles, crs=epsg_code)

        return self.grid_gdf




    def filter_by_points(self, points_gdf):
        """
        Filter the grid to include only cells that contain at least one point.

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.

        Returns:
            gpd.GeoDataFrame: Filtered grid containing only cells with points.
        """
        if self.grid_gdf is None:
            raise ValueError("Grid not created. Call create_grid() first.")

        # Ensure the CRS of points matches the grid
        if points_gdf.crs != self.grid_gdf.crs:
            points_gdf = points_gdf.to_crs(self.grid_gdf.crs)

        # Filter grid cells that contain at least one point
        # More efficient approach than the lambda function in the original code
        spatial_index = points_gdf.sindex

        # Function to check if a polygon contains any points
        def contains_points(polygon):
            possible_matches_idx = list(spatial_index.intersection(polygon.bounds))
            if len(possible_matches_idx) > 0:
                possible_matches = points_gdf.iloc[possible_matches_idx]
                return any(possible_matches.within(polygon))
            return False

        def contains_no_points(polygon):
            possible_matches_idx = list(spatial_index.intersection(polygon.bounds))
            if len(possible_matches_idx) == 0:
                # No potential matches in the bounding box, so no points in the polygon
                return True
            else:
                # Check if any of the possible matches are actually within the polygon
                possible_matches = points_gdf.iloc[possible_matches_idx]
                return not any(possible_matches.within(polygon))

        # Apply the filter
        occupied_grid_cells = self.grid_gdf[self.grid_gdf.geometry.apply(contains_points)]
        empty_gridcells = self.grid_gdf[self.grid_gdf.geometry.apply(contains_no_points)]

        return occupied_grid_cells, empty_gridcells

    def set_raster_path(self, raster_path):
        """
        Set or update the raster path and reload metadata.

        Parameters:
            raster_path (Path): Path to the raster image.
        """
        self.raster_path = raster_path
        self._load_raster_metadata()

    def get_grid(self):
        """
        Get the current grid.

        Returns:
            gpd.GeoDataFrame: The current grid.
        """
        return self.grid_gdf


# Example usage:
# grid_manager = GeoSpatialRasterGrid(Path("path/to/raster.tif"))
# grid = grid_manager.create_grid(x_size=256, y_size=256, overlap_ratio=0.2)
# points_gdf = gpd.read_file("path/to/points.shp")
# filtered_grid = grid_manager.filter_by_points(points_gdf)



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
