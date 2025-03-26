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

    def __init__(self, raster_path: Path):
        """
        Initialize the GeoSpatialRasterGrid with an optional raster path.

        Parameters:
            raster_path (Path, optional): Path to the raster image.
        """
        self.raster_path: Path = raster_path
        self.grid_gdf:  typing.Optional[gpd.GeoDataFrame] = None
        self.crs = None
        self.transform = None
        self.bounds = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.gdf_raster_mask: typing.Optional[gpd.GeoDataFrame] = None

        if raster_path:
            self._load_raster_metadata()
            self._get_extent_polygon()

    def _get_extent_polygon(self):
        """
        Calculate which part of the raster are actual pixel and which is blank space
        :return: GeoDataFrame containing polygon of the non-blank areas
        """
        if not self.raster_path:
            raise ValueError("Raster path not set. Either initialize with a path or set it with set_raster_path().")

        # Ensure raster metadata is loaded
        if self.transform is None:
            self._load_raster_metadata()

        with rasterio.open(self.raster_path) as src:
            # For large rasters, use downsampling to improve performance
            scale_factor = max(1, min(10, src.width // 1000, src.height // 1000))

            if scale_factor > 1:
                # Read at reduced resolution for large images
                data = src.read(
                    out_shape=(
                        src.count,
                        int(src.height / scale_factor),
                        int(src.width / scale_factor)
                    ),
                    resampling=rasterio.warp.Resampling.nearest
                )

                # Adjust transform for downsampled resolution
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[2]),
                    (src.height / data.shape[1])
                )
            else:
                # Read at full resolution for smaller images
                data = src.read()
                transform = src.transform

            # Create mask where any band has non-zero values
            if data.shape[0] >= 3:  # RGB or multi-band image
                mask = (data[0] != 0) | (data[1] != 0) | (data[2] != 0)
            else:  # Single-band image
                mask = data[0] != 0

            # Get shapes of all non-zero regions
            shapes = rasterio.features.shapes(
                mask.astype('uint8'),
                mask=mask,
                transform=transform
            )

            # Convert shapes to polygons
            polygons = []
            for shape, value in shapes:
                if value == 1:  # Only include non-zero areas
                    try:
                        polygons.append(Polygon(shape['coordinates'][0]))
                    except Exception:
                        # Skip invalid polygons
                        continue

            # Create a GeoDataFrame with the polygons
            if polygons:
                gdf = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)

                # Dissolve all polygons into one if multiple exist
                if len(polygons) > 1:
                    gdf = gdf.dissolve()
                    gdf = gdf.reset_index(drop=True)

                # Simplify the geometry to reduce complexity
                simplify_tolerance = self.pixel_size_x * max(1, scale_factor // 2)
                gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify_tolerance)

                self.gdf_raster_mask = gdf
                return self.gdf_raster_mask
            else:
                # Return empty GeoDataFrame if no non-zero pixels found

                self.gdf_raster_mask = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                return self.gdf_raster_mask


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

    def object_centered_grid(self, points_gdf, box_size_x, box_size_y):
        """
        Create a grid where each cell is centered on a point in the points_gdf.

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries to center grids on.
            box_size_x (float): Width of each grid cell in pixels.
            box_size_y (float): Height of each grid cell in pixels.

        Returns:
            gpd.GeoDataFrame: A grid where each cell is centered on a point.
        """
        if not self.raster_path:
            raise ValueError("Raster path not set. Either initialize with a path or set it with set_raster_path().")

        # Ensure raster metadata is loaded
        if self.transform is None:
            self._load_raster_metadata()

        # Ensure the CRS of points matches the raster
        if points_gdf.crs != self.crs:
            points_gdf = points_gdf.to_crs(self.crs)

        # Convert box size from pixels to world coordinates
        box_width_world = box_size_x * self.pixel_size_x
        box_height_world = box_size_y * self.pixel_size_y

        # Half dimensions for centering
        half_width = box_width_world / 2
        half_height = box_height_world / 2

        grid_cells = []
        tiles = []
        image_name = Path(self.raster_path).stem

        # For each point, create a centered box
        for idx, point in points_gdf.iterrows():
            # Get point coordinates
            x_center, y_center = point.geometry.x, point.geometry.y

            # Calculate box corners
            x1 = x_center - half_width
            y1 = y_center - half_height
            x2 = x_center + half_width
            y2 = y_center + half_height

            # Create polygon
            tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            tiles.append(tile_polygon)

            # Generate a unique tile name
            coord_tile_name = f"{image_name}_centered_{int(x_center)}_{int(y_center)}"

            grid_cells.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": x_center,
                "center_y": y_center,
                "tile_name": coord_tile_name,
                "image_name": image_name,
                "point_id": idx  # Store the original point ID
            })

        # Convert to GeoDataFrame
        epsg_code = self._get_epsg_code()
        centered_grid_gdf = gpd.GeoDataFrame(data=grid_cells, geometry=tiles, crs=epsg_code)

        return centered_grid_gdf

    def create_empty_regions_grid(self, points_gdf, box_size_x, box_size_y, min_distance_pixels=400):
        """
        Create a grid of rectangles that are guaranteed to be empty (no points within min_distance_pixels).

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries to avoid.
            box_size_x (float): Width of each grid cell in pixels.
            box_size_y (float): Height of each grid cell in pixels.
            min_distance_pixels (float): Minimum distance in pixels from any point to consider a region empty.

        Returns:
            gpd.GeoDataFrame: Grid of empty regions.
        """
        if not self.raster_path:
            raise ValueError("Raster path not set. Either initialize with a path or set it with set_raster_path().")

        # Ensure raster metadata is loaded
        if self.transform is None:
            self._load_raster_metadata()

        # Ensure the CRS of points matches the raster
        if points_gdf.crs != self.crs:
            points_gdf = points_gdf.to_crs(self.crs)

        # Convert pixel distances to world coordinates
        min_distance_world = min_distance_pixels * self.pixel_size_x  # Assuming square pixels or using x-resolution
        box_width_world = box_size_x * self.pixel_size_x
        box_height_world = box_size_y * self.pixel_size_y

        # Get bounds of the raster
        min_x, min_y, max_x, max_y = self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top

        # First, create a regular grid covering the entire raster
        # We'll use a non-overlapping grid as a starting point
        step_x = box_width_world
        step_y = box_height_world

        # Generate candidate grid cells
        candidate_cells = []
        candidate_geometries = []
        image_name = Path(self.raster_path).stem

        for y in np.arange(min_y, max_y, step_y):
            for x in np.arange(min_x, max_x, step_x):
                # Define tile coordinates
                x1, y1 = x, y
                x2, y2 = x + box_width_world, y + box_height_world

                # Ensure tiles stay within bounds
                x2 = min(x2, max_x)
                y2 = min(y2, max_y)

                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Create polygon
                tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                candidate_geometries.append(tile_polygon)

                # Generate a unique tile name
                coord_tile_name = f"{image_name}_empty_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

                candidate_cells.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "tile_name": coord_tile_name,
                    "image_name": image_name,
                })

        # Create a GeoDataFrame for all candidate grid cells
        candidate_gdf = gpd.GeoDataFrame(data=candidate_cells, geometry=candidate_geometries, crs=self.crs)

        # Buffer points by min_distance_world to create exclusion zones
        buffered_points = points_gdf.copy()
        buffered_points['geometry'] = buffered_points.geometry.buffer(min_distance_world)

        # Filter out any candidate cell that intersects with any buffered point
        def is_far_from_all_points(cell_geom):
            return not any(buffered_points.intersects(cell_geom))

        # Apply the filter
        empty_grid_cells = candidate_gdf[candidate_gdf.geometry.apply(is_far_from_all_points)]

        return empty_grid_cells

    def create_balanced_dataset_grids(self, points_gdf, box_size_x, box_size_y, num_empty_samples=None,
                                      min_distance_pixels=400, random_seed=42):
        """
        Create both object-centered grids and guaranteed empty grids for a balanced dataset.

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.
            box_size_x (float): Width of each grid cell in pixels.
            box_size_y (float): Height of each grid cell in pixels.
            num_empty_samples (int, optional): Number of empty samples to select. If None, will match the number of points.
            min_distance_pixels (float): Minimum distance in pixels from any point to consider a region empty.
            random_seed (int): Random seed for reproducible sampling of empty regions.

        Returns:
            tuple: (object_centered_grid, empty_regions_grid) - Two GeoDataFrames
        """
        # Get the object-centered grid
        object_grid = self.object_centered_grid(points_gdf, box_size_x, box_size_y)

        # Get the empty regions grid
        all_empty_regions = self.create_empty_regions_grid(points_gdf, box_size_x, box_size_y, min_distance_pixels)

        # Determine how many empty samples to select
        if num_empty_samples is None:
            num_empty_samples = len(object_grid)

        # If we need to sample from the empty regions
        if len(all_empty_regions) > num_empty_samples:
            # Randomly select the required number of empty regions
            empty_regions = all_empty_regions.sample(n=num_empty_samples, random_state=random_seed)
        else:
            # If we don't have enough empty regions, use all available
            empty_regions = all_empty_regions

        # Add a label column to distinguish between the two sets
        object_grid['has_object'] = True
        empty_regions['has_object'] = False

        return object_grid, empty_regions



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
