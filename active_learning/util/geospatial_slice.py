"""
slice images into smaller parts
"""
import geopandas as gpd
import multiprocessing
import numpy as np
import os
import pandas as pd
import rasterio
import typing
from loguru import logger
from osgeo import gdal
from pathlib import Path
from rtree import index
from shapely.geometry import Polygon
from tqdm import tqdm
import concurrent.futures

from active_learning.types.Exceptions import ProjectionError, NoLabelsError, NullRow
from active_learning.util.geospatial_image_manipulation import cut_geospatial_raster_with_grid_gdal
from active_learning.util.image import get_image_id

from active_learning.util.projection import world_to_pixel


class ImageGrid(object):
    """ base class for creating a grid on top of images """


class OrdinaryImageGrid(ImageGrid):
    """
    Class for creating and managing ordinary image grids.
    """


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
        self.grid_gdf: typing.Optional[gpd.GeoDataFrame] = None
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
            if data.shape[0] == 3:  # RGB or multi-band image
                mask = (data[0] != 0) | (data[1] != 0) | (data[2] != 0)
            if data.shape[0] == 4:  # RGBA
                # Use the alpha Channel if it exists
                mask = (data[3] != 0)
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
        step_x = x_size_world - x_overlap_world # + self.pixel_size_x # add pixel size to ensure no gaps
        step_y = y_size_world - y_overlap_world # + self.pixel_size_y # add pixel size to ensure annotations are in the grid

        grid_cells = []
        tiles = []
        image_name = Path(self.raster_path).stem

        # Generate grid based on pixel-based slicing
        for y in np.arange(min_y, max_y, step_y):
            for x in np.arange(min_x, max_x, step_x):
                # Define tile coordinates
                x1, y1 = x, y
                x2, y2 = x + x_size_world, y + y_size_world

                # Ensure tiles stay within bounds # TODO this probably a bad idea, because then the tiles are not the same size
                # x2 = min(x2, max_x)
                # y2 = min(y2, max_y)

                # Create polygon
                tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                tiles.append(tile_polygon)

                # Generate a unique tile name based on image name and coordinates
                coord_tile_name = f"{image_name}_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

                if coord_tile_name == "Flo_FLPO01_28012023_781121_9863875_781132_9863878":
                    """Sometimes the cell are not big enough"""
                    pass

                grid_cells.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "tile_name": coord_tile_name,
                    "image_name": image_name,
                })
            else:
                # Skip tiles that are outside the raster mask
                pass

        # Convert to GeoDataFrame
        epsg_code = self._get_epsg_code()
        self.grid_gdf = gpd.GeoDataFrame(data=grid_cells, geometry=tiles, crs=epsg_code)


        # Get indices of grid polygons that intersect with raster mask
        intersecting_indices = gpd.sjoin(
            self.grid_gdf,
            self.gdf_raster_mask,
            predicate='intersects',
            how='inner'
        ).index
        self.grid_gdf = self.grid_gdf.loc[intersecting_indices]

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

        tile_registry = {}

        # For each point, create a centered box
        for idx, point in points_gdf.iterrows():
            # Get point coordinates
            if point.geometry is None or point.geometry.is_empty:
                logger.warning(f"Skipping empty point at index {idx}.")
                continue

            x_center, y_center = point.geometry.x, point.geometry.y

            # Calculate box corners
            x1 = x_center - half_width
            y1 = y_center - half_height
            x2 = x_center + half_width
            y2 = y_center + half_height

            tile_key = (float(x_center), float(y_center))
            if tile_key not in tile_registry:
                tile_registry[tile_key] = 1
            else:
                tile_registry[tile_key] += 1
                raise ValueError(f"Multiple points at the same location: {tile_key}")

            # Create polygon
            tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            tiles.append(tile_polygon)

            # Generate a unique tile name
            coord_tile_name = f"{image_name}_centered_{float(x_center)}_{float(y_center)}"

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

    def create_empty_regions_grid_slow(self, points_gdf, box_size_x, box_size_y, min_distance_pixels=400,
                                       min_area_ratio=0.7):
        """
        Create a grid of rectangles that are guaranteed to be empty (no points within min_distance_pixels)
        and only within the non-blank areas of the raster.

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries to avoid.
            box_size_x (float): Width of each grid cell in pixels.
            box_size_y (float): Height of each grid cell in pixels.
            min_distance_pixels (float): Minimum distance in pixels from any point to consider a region empty.
            min_area_ratio (float): Minimum ratio of grid cell area that must overlap with non-blank areas.

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

        # Get the extent polygon (non-blank areas)
        extent_polygon_gdf = self._get_extent_polygon()
        if extent_polygon_gdf.empty:
            logger.warning("No non-blank areas found in the raster. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        # Get the actual non-blank area polygon
        extent_polygon = extent_polygon_gdf.iloc[0].geometry

        # Convert pixel distances to world coordinates
        min_distance_world = min_distance_pixels * self.pixel_size_x
        box_width_world = box_size_x * self.pixel_size_x
        box_height_world = box_size_y * self.pixel_size_y

        # Get bounds of the extent polygon instead of the entire raster
        min_x, min_y, max_x, max_y = extent_polygon.bounds

        # Use non-overlapping grid as a starting point
        step_x = box_width_world
        step_y = box_height_world

        # Generate candidate grid cells
        candidate_cells = []
        candidate_geometries = []
        image_name = Path(self.raster_path).stem

        for y in np.arange(min_y, max_y, step_y):
            for x in np.arange(min_x, max_x, step_x):
                # Define tile coordinates
                # logger.info(f"Processing cell at ({x}, {y})")
                x1, y1 = x, y
                x2, y2 = x + box_width_world, y + box_height_world

                # Ensure tiles stay within bounds
                x2 = min(x2, max_x)
                y2 = min(y2, max_y)

                # Create polygon
                tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                # Check if the tile is within or mostly within the non-blank area
                if tile_polygon.intersects(extent_polygon):
                    intersection = tile_polygon.intersection(extent_polygon)
                    area_ratio = intersection.area / tile_polygon.area

                    # Only include tiles that have sufficient overlap with non-blank areas
                    if area_ratio >= min_area_ratio:
                        # Calculate center
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

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
                            "area_ratio": area_ratio
                        })

        # If no candidate cells were found, return empty GeoDataFrame
        if not candidate_cells:
            logger.info("No candidate grid cells found within non-blank areas.")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        # Create a GeoDataFrame for all candidate grid cells
        candidate_gdf = gpd.GeoDataFrame(data=candidate_cells, geometry=candidate_geometries, crs=self.crs)

        # Buffer points by min_distance_world to create exclusion zones
        buffered_points = points_gdf.copy()
        buffered_points['geometry'] = buffered_points.geometry.buffer(min_distance_world)

        # Use spatial index for more efficient filtering
        buffered_points_sindex = buffered_points.sindex

        def is_far_from_all_points(cell_geom):
            # Get potential intersecting buffered points
            possible_matches_idx = list(buffered_points_sindex.intersection(cell_geom.bounds))
            if not possible_matches_idx:
                return True
            possible_matches = buffered_points.iloc[possible_matches_idx]
            return not any(possible_matches.intersects(cell_geom))

        # Apply the filter
        empty_grid_cells = candidate_gdf[candidate_gdf.geometry.apply(is_far_from_all_points)]

        return empty_grid_cells

    def create_empty_regions_grid(self, points_gdf, box_size_x, box_size_y, max_boxes=100,
                                  min_distance_pixels=400, min_area_ratio=0.7, random_seed=42):
        """
        Create a grid of rectangles that are guaranteed to be empty (no points within min_distance_pixels)
        and only within the non-blank areas of the raster.

        Parameters:
            points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries to avoid.
            box_size_x (float): Width of each grid cell in pixels.
            box_size_y (float): Height of each grid cell in pixels.
            max_boxes (int): Maximum number of boxes to generate.
            min_distance_pixels (float): Minimum distance in pixels from any point to consider a region empty.
            min_area_ratio (float): Minimum ratio of grid cell area that must overlap with non-blank areas.
            random_seed (int): Seed for random shuffling to ensure reproducibility.

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

        # Get the extent polygon (non-blank areas)
        extent_polygon_gdf = self._get_extent_polygon()
        if extent_polygon_gdf.empty:
            logger.warning("No non-blank areas found in the raster. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        # Get the actual non-blank area polygon
        extent_polygon = extent_polygon_gdf.iloc[0].geometry

        # Convert pixel distances to world coordinates
        min_distance_world = min_distance_pixels * self.pixel_size_x
        box_width_world = box_size_x * self.pixel_size_x
        box_height_world = box_size_y * self.pixel_size_y

        # Get bounds of the extent polygon
        min_x, min_y, max_x, max_y = extent_polygon.bounds

        # Create the coordinate arrays
        xs = np.arange(min_x, max_x, box_width_world)
        ys = np.arange(min_y, max_y, box_height_world)

        # Calculate total possible grid cells
        total_possible_cells = len(xs) * len(ys)
        logger.info(f"Total possible grid cells: {total_possible_cells}")

        # Create all coordinate pairs and shuffle them
        np.random.seed(random_seed)
        coordinate_pairs = []
        for x in xs:
            for y in ys:
                coordinate_pairs.append((x, y))

        np.random.shuffle(coordinate_pairs)

        # Buffer points for exclusion zones
        buffered_points = points_gdf.copy()
        buffered_points['geometry'] = buffered_points.geometry.buffer(min_distance_world)

        # Create spatial index for buffered points

        points_idx = index.Index()
        for i, point in enumerate(buffered_points.geometry):

            if point is None or point.is_empty:
                logger.warning(f"Skipping empty point at index {i}.")
            else:
                points_idx.insert(i, point.bounds)


        image_name = Path(self.raster_path).stem

        # Lists to store valid boxes
        candidate_cells = []
        candidate_geometries = []

        # Process cells until we have enough or run out of coordinates
        from tqdm import tqdm

        # We'll process all coordinates if needed
        pbar = tqdm(total=min(total_possible_cells, max_boxes * 10), desc="Checking grid cells")
        checked_cells = 0

        from shapely.geometry import box

        # Process coordinate pairs until we have enough boxes or check all coordinates
        for x, y in coordinate_pairs:
            # Check if we have enough boxes
            if len(candidate_cells) >= max_boxes:
                break

            # Update progress and counter
            checked_cells += 1
            pbar.update(1)

            # Set a maximum number of cells to check to avoid infinite loops
            if checked_cells >= min(total_possible_cells, max_boxes * 1000):
                logger.warning(f"Checked {checked_cells} cells but only found {len(candidate_cells)} valid boxes. "
                               f"Consider adjusting parameters.")
                break

            # Define tile coordinates
            x2 = min(x + box_width_world, max_x)
            y2 = min(y + box_height_world, max_y)

            # Create box
            cell_bounds = (x, y, x2, y2)
            cell_box = box(*cell_bounds)

            # Check if box intersects with the extent polygon (non-blank area)
            if not cell_box.intersects(extent_polygon):
                continue

            # Check area ratio with non-blank regions
            intersection = cell_box.intersection(extent_polygon)
            area_ratio = intersection.area / cell_box.area
            if area_ratio < min_area_ratio:
                continue

            # Check if box intersects with any buffered point using spatial index
            point_candidates = list(points_idx.intersection(cell_bounds))
            if point_candidates:
                intersects_point = False
                for i in point_candidates:
                    if buffered_points.iloc[i].geometry.intersects(cell_box):
                        intersects_point = True
                        break

                if intersects_point:
                    continue  # Skip this cell as it intersects with a buffered point

            # This cell is good - add it to our candidates
            center_x = (x + x2) / 2
            center_y = (y + y2) / 2

            candidate_geometries.append(cell_box)
            coord_tile_name = f"{image_name}_empty_{int(x)}_{int(y)}_{int(x2)}_{int(y2)}"

            candidate_cells.append({
                "x1": x,
                "y1": y,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "tile_name": coord_tile_name,
                "image_name": image_name,
                "area_ratio": area_ratio
            })

            # Update progress bar description to show progress
            pbar.set_description(f"Found {len(candidate_cells)}/{max_boxes} valid cells")

        pbar.close()

        # Report final statistics
        logger.info(f"Found {len(candidate_cells)} valid cells after checking {checked_cells} locations")

        # If no candidate cells were found, return empty GeoDataFrame
        if not candidate_cells:
            logger.warning("No empty regions found that meet the criteria. Consider relaxing constraints.")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        # Create a GeoDataFrame from our filtered cells
        empty_grid_cells = gpd.GeoDataFrame(data=candidate_cells, geometry=candidate_geometries, crs=self.crs)

        return empty_grid_cells

    def create_balanced_dataset_grids(self, points_gdf: gpd.GeoDataFrame,
                                      box_size_x: int, box_size_y: int,
                                      num_empty_samples=None,
                                      min_distance_pixels=400, random_seed=42,
                                      object_centered=True,
                                      overlap_ratio=0.0):
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

        if object_centered:
            logger.info(f"cutting objects centered grid with box size {box_size_x}x{box_size_y}")
            object_grid = self.object_centered_grid(points_gdf, box_size_x, box_size_y)
        else:
            logger.info(f"cutting objects with a regular grid {box_size_x}x{box_size_y}")
            object_grid = self.create_regular_grid(x_size=box_size_x, y_size=box_size_y, overlap_ratio=overlap_ratio)

        logger.info(f"start empty cutout")
        # Get the empty regions grid
        all_empty_regions = self.create_empty_regions_grid(points_gdf=points_gdf, box_size_x=box_size_x,
                                                           box_size_y=box_size_y,
                                                           min_distance_pixels=min_distance_pixels,
                                                           max_boxes=num_empty_samples,
                                                           min_area_ratio=0.7, random_seed=random_seed)

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


    def create_filtered_grid(self, points_gdf: gpd.GeoDataFrame,
                                      box_size_x: int, box_size_y: int,
                                      num_empty_samples=None,
                                      min_distance_pixels=400, random_seed=42,
                                      object_centered=True,
                                      overlap_ratio=0.0):
        """
        Create grids cell usable for Human in the loop Annotations

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

        if object_centered:
            logger.info(f"cutting objects centered grid with box size {box_size_x}x{box_size_y}")
            object_grid = self.object_centered_grid(points_gdf, box_size_x, box_size_y)
        else:
            logger.info(f"cutting objects with a regular grid {box_size_x}x{box_size_y}")
            object_grid = self.create_regular_grid(x_size=box_size_x, y_size=box_size_y, overlap_ratio=overlap_ratio)

        # logger.info(f"start empty cutout")
        # # Get the empty regions grid
        # all_empty_regions = self.create_empty_regions_grid(points_gdf=points_gdf, box_size_x=box_size_x,
        #                                                    box_size_y=box_size_y,
        #                                                    min_distance_pixels=min_distance_pixels,
        #                                                    max_boxes=num_empty_samples,
        #                                                    min_area_ratio=0.7, random_seed=random_seed)
        # 
        # # Determine how many empty samples to select
        # if num_empty_samples is None:
        #     num_empty_samples = len(object_grid)
        # 
        # # If we need to sample from the empty regions
        # if len(all_empty_regions) > num_empty_samples:
        #     # Randomly select the required number of empty regions
        #     empty_regions = all_empty_regions.sample(n=num_empty_samples, random_state=random_seed)
        # else:
        #     # If we don't have enough empty regions, use all available
        #     empty_regions = all_empty_regions

        # Add a label column to distinguish between the two sets
        object_grid['has_object'] = True
        # empty_regions['has_object'] = False

        return object_grid #, empty_regions


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

def _process_grid_chunk(args) -> gpd.GeoDataFrame:
    """
    Worker function to process a chunk of the grid.

    Args:
        args: Tuple containing (grid_chunk, raster_path, output_dir, crs)

    Returns:
        List of paths to the sliced raster files
    """
    assert len(args) == 4, "Expected 4 arguments: grid_chunk, raster_path, output_dir, crs"
    grid_chunk, raster_path, output_dir, crs = args

    # Convert back to GeoDataFrame if needed
    if not isinstance(grid_chunk, gpd.GeoDataFrame):
        grid_chunk = gpd.GeoDataFrame(
            grid_chunk,
            geometry='geometry',
            crs=crs
        )

    # Process this chunk
    slice = cut_geospatial_raster_with_grid_gdal(
        raster_path=raster_path,
        grid_gdf=grid_chunk,
        output_dir=output_dir
    )

    grid_chunk["slice_path"] = slice
    return grid_chunk


class GeoSlicer():
    """
    Helper to slice geospational rasters into smaller parts
    """

    def __init__(self, base_path: Path, image_name: str, grid: gpd.GeoDataFrame, output_dir: Path):
        self.gdf_slices: gpd.GeoDataFrame = None
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
        image_size = os.path.getsize(
            full_image_path)  ## gdal.Open is not realizing when the file is missing. So do this before
        logger.info(f"size of the image: {image_size // 1024}MB at {full_image_path}")
        orthophoto_raster = gdal.Open(str(full_image_path))
        self.geo_transform = orthophoto_raster.GetGeoTransform()
        self.orthophoto_raster = orthophoto_raster

    def set_object_locations(self, object_locations: gpd.GeoDataFrame):
        self.object_locations = object_locations

    def slice_annotations(self, points_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Project the points into the local pixel coordinates of the raster cell.
        This assumes each point corresponds to exactly one grid cell that was created around it.

        Parameters:
            points_gdf: GeoDataFrame of points
            grid_gdf: GeoDataFrame of grid cells

        Returns:
            GeoDataFrame with points and their local coordinates within their grid cells
        """
        assert len(points_gdf) > 0, "No points to slice"
        # Fix CRS assertion - comparing CRS objects, not lengths
        assert points_gdf.crs == grid_gdf.crs, "CRS mismatch between points and grid"
        # Assert the expected 1:1 relationship
        assert len(points_gdf) == len(
            grid_gdf), f"Points count ({len(points_gdf)}) must match grid count ({len(grid_gdf)})"

        # Instead of spatial join, use index-based matching if points and grids are already aligned
        # If points and grids have matching indices, use direct mapping
        points_in_grid = points_gdf.copy()
        points_in_grid["grid_id"] = points_gdf.index
        points_in_grid["image_name_right"] = grid_gdf["image_name"].values
        points_in_grid["grid_geometry"] = grid_gdf["geometry"].values
        points_in_grid["tile_name"] = grid_gdf["tile_name"].values

        # Check if any points couldn't be assigned to a grid cell
        if points_in_grid["image_name_right"].isnull().any():
            logger.error(f"Some points could not be assigned to a grid cell: "
                         f"{points_in_grid['image_name_right'].isnull().sum()}")
            raise ProjectionError("There is a problem with the projection")

        # Convert all points from world to pixel coordinates
        points_in_grid["pixel_x"], points_in_grid["pixel_y"] = zip(
            *points_in_grid.geometry.apply(lambda point: world_to_pixel(self.geo_transform, point.x, point.y)))

        # Calculate local pixel coordinates relative to the grid cell origin
        def calculate_local_coords(row):
            grid_idx = row["grid_id"]
            if pd.isnull(grid_idx):
                return np.nan, np.nan

            grid_cell = grid_gdf.iloc[int(grid_idx)]
            grid_minx, grid_miny, grid_maxx, grid_maxy = grid_cell.geometry.bounds

            # Convert the grid cell corners to pixel coordinates
            min_pixel_x, max_pixel_y = world_to_pixel(self.geo_transform, grid_minx,
                                                      grid_miny)  # Bottom-left corner
            max_pixel_x, min_pixel_y = world_to_pixel(self.geo_transform, grid_maxx, grid_maxy)  # Top-right corner

            # Calculate local coordinates with bottom-left origin
            local_x = row["pixel_x"] - min_pixel_x  # Distance from left edge

            # For Y, we need to flip the orientation:
            # Instead of measuring from the top, measure from the bottom
            local_y = -1 * (min_pixel_y - row["pixel_y"])  # Distance from bottom edge

            return local_x, local_y

        points_in_grid["local_pixel_x"], points_in_grid["local_pixel_y"] = zip(
            *points_in_grid.apply(calculate_local_coords, axis=1))

        return points_in_grid

    def slice_annotations_regular_grid(self, points_gdf: gpd.GeoDataFrame,
                                       grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        project the points into the local pixel coordinates of the raster cell. This assumes the grid was created around the points NOT as regular grid across the image
        :param points_gdf:
        :return:
        """

        if len(points_gdf) == 0:
            raise NoLabelsError("No points to slice")

        grid_index_col = "grid_id"
        # assign each point to a grid cell
        # Step 1: Assign each point to a grid cell using spatial join
        # This will match each point to the grid cell that contains it
        points_in_grid = gpd.sjoin(points_gdf, grid_gdf, how="left", predicate="within")

        if "image_name_right" in points_in_grid.columns and points_in_grid["image_name_right"].isnull().any():
            logger.error(
                f"some points could not be assigned to a grid cell: {points_in_grid['image_name_right'].isnull().sum()}, THIS IS DUE TO THE fact someone messed up the shapefile or orhtomosaic projections")
            raise ProjectionError("There is a problem with the projection")

        if points_in_grid["index_right"].isnull().any():
            logger.warning(f"some points could not be assigned to a grid cell: {points_in_grid['index_right'].isnull().sum()}, Could be a projection problem. It affects {len(points_in_grid[points_in_grid['index_right'].isnull()])} ")
            # remove these
            points_in_grid = points_in_grid[~points_in_grid["index_right"].isnull()]
        if points_in_grid["index_right"].isnull().all():
            raise NullRow(f"points_in_grid may not contain None/NaN because every point should be within a grid cell, but found {points_in_grid['index_right'].isnull().sum()} points without a grid cell")

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
                if local_x == max_pixel_x - min_pixel_x:
                    logger.warning(
                        f"local_x is equal to grid height in pixels for point {row['geometry']}, this might be due to rounding errors or very small grid cells")
                    local_x = local_x-1

                # For Y, we need to flip the orientation:
                # Instead of measuring from the top, measure from the bottom
                local_y = -1 * (min_pixel_y - row["pixel_y"])  # Distance from bottom edge

                if local_y == max_pixel_y - min_pixel_y:
                    logger.warning(
                        f"local_x is equal to grid height in pixels for point {row['geometry']}, this might be due to rounding errors or very small grid cells")
                    local_y = local_y - 1

                return local_x, local_y

            points_in_grid["local_pixel_x"], points_in_grid["local_pixel_y"] = zip(
                *points_in_grid.apply(calculate_local_coords, axis=1))

        return points_in_grid

    # Then modify your method to use this function
    def slice_very_big_raster(self, num_workers=None, num_chunks=10):
        """
        Slice the very big geotiff into smaller parts in parallel

        Args:
            num_workers: Number of parallel workers (default: None, uses CPU count)

        Returns:
            List of paths to the sliced raster files
        """

        # Determine number of workers
        if num_workers is None or num_workers > 32:
            num_workers = min(32, max(1, multiprocessing.cpu_count() - 1))  # Leave one CPU free and on big machines don't overdo it

        # Make sure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Split the grid into chunks for parallel processing

        if len(self.grid) < num_chunks:
            num_chunks = len(self.grid)
        # Convert to list of DataFrames for easier splitting
        indices = np.array_split(range(len(self.grid)), num_chunks)
        grid_dfs = [self.grid.iloc[idx_chunk] for idx_chunk in indices]

        # Prepare arguments for each worker
        args_list = [
            (chunk, self.base_path.joinpath(self.image_name), self.output_dir, self.grid.crs)
            for chunk in grid_dfs
        ]

        logger.info(
            f"Processing raster {self.image_name} in parallel using {num_workers} workers to process {len(self.grid)} grid cells")
        all_slices: typing.List[Path] = []

        if num_workers == 1:
            # Single process mode - useful for debugging
            for args in tqdm(args_list, desc="Processing grid chunks"):
                gdf_slices = _process_grid_chunk(args)
                all_slices.extend(gdf_slices)
        else:


            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Process chunks in parallel and collect results
                futures = {executor.submit(_process_grid_chunk, args): i for i, args in enumerate(args_list)}

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                   desc=f"Processing grid chunks with {len(futures)} tasks"):
                    try:
                        gdf_slices = future.result()
                        all_slices.append(gdf_slices)
                    except Exception as e:
                        logger.error(f"Error processing chunk {futures[future]}: {str(e)}")

        # Store the results
        gdf_all_slices = gpd.GeoDataFrame(
            pd.concat(all_slices, ignore_index=True),
            geometry='geometry',
            crs=self.grid.crs
        )
        self.gdf_slices = gdf_all_slices

        # Report statistics
        logger.info(f"Created {len(gdf_all_slices)} slices from {len(self.grid)} grid cells")

        return self.gdf_slices


def get_geospatial_sliced_path(base_path, x_size, y_size):
    if x_size is not None and y_size is not None:
        sliced_path = base_path.joinpath(f"sliced_{x_size}_{y_size}px")
    else:
        sliced_path = base_path.joinpath(f"sliced")
    return sliced_path



