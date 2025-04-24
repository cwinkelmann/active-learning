import geopandas as gpd
import json
import numpy as np
import osgeo.gdal as gdal
gdal.UseExceptions()

import rasterio
from loguru import logger
from pathlib import Path
from rasterio.mask import mask
from shapely.geometry import Polygon
from typing import List
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from pathlib import Path

from active_learning.util.projection import world_to_pixel


def get_raster_crs(raster_path: str) -> int:
    """
    Get the EPSG code of the raster's Coordinate Reference System (CRS).

    Parameters:
        raster_path (str): Path to the raster file.

    Returns:
        int: EPSG code of the raster's CRS.
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        if crs and crs.to_epsg():
            return crs.to_epsg()
        else:
            raise ValueError("Could not determine CRS for the raster.")


def safe_world_file(raster_path: Path) -> Path:
    """
    Create a world file for a raster if it does not already exist.

    Parameters:
        raster_path (Path): Path to the raster file.

    Returns:
        Path: Path to the created world file or existing one.
    """
    raster_path = Path(raster_path)
    world_file_path = raster_path.with_suffix(".wld")  # Create world file name

    # Check if world file already exists
    if world_file_path.exists():
        print(f"World file already exists: {world_file_path}")
        return world_file_path

    # Open raster to get transformation parameters
    with rasterio.open(raster_path) as src:
        transform = src.transform  # Affine transformation

        # World file format:
        world_data = [
            f"{transform.a:.8f}\n",  # Pixel size in X direction
            f"{transform.b:.8f}\n",  # Rotation (usually 0)
            f"{transform.d:.8f}\n",  # Rotation (usually 0)
            f"{transform.e:.8f}\n",  # Pixel size in Y direction (negative for North-up)
            f"{transform.xoff:.8f}\n",  # X coordinate of the upper-left corner
            f"{transform.yoff:.8f}\n"   # Y coordinate of the upper-left corner
        ]

    # Write world file
    with open(world_file_path, "w") as wf:
        wf.writelines(world_data)

    print(f"World file created: {world_file_path}")
    return world_file_path




def save_world_file_json(raster_path: Path) -> Path:
    """
    Create a JSON world file for a raster if it does not already exist.

    Parameters:
        raster_path (Path): Path to the raster file.

    Returns:
        Path: Path to the created JSON world file or existing one.
    """
    raster_path = Path(raster_path)
    json_file_path = raster_path.with_suffix(".json")  # Save as JSON

    # Check if JSON world file already exists
    if json_file_path.exists():
        print(f"JSON world file already exists: {json_file_path}")
        return json_file_path

    # Open raster to extract georeferencing metadata
    with rasterio.open(raster_path) as src:
        transform = src.transform  # Affine transformation
        metadata = {
            "pixel_size_x": transform.a,  # Pixel size in X direction
            "rotation_x": transform.b,    # Rotation (usually 0)
            "rotation_y": transform.d,    # Rotation (usually 0)
            "pixel_size_y": transform.e,  # Pixel size in Y direction (negative for North-up)
            "top_left_x": transform.xoff, # X coordinate of the upper-left corner
            "top_left_y": transform.yoff, # Y coordinate of the upper-left corner
            "crs": src.crs.to_string() if src.crs else None,  # Coordinate Reference System
            "width": src.width,  # Raster width (in pixels)
            "height": src.height # Raster height (in pixels)
        }

    # Write to JSON file
    with open(json_file_path, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, indent=4)

    print(f"JSON world file created: {json_file_path}")
    return json_file_path




def create_regular_geospatial_raster_grid(full_image_path: Path,
                                          x_size: float,
                                          y_size: float,
                                          overlap_ratio: float) -> gpd.GeoDataFrame:
    """
    Create a vector grid for a geospatial raster, slicing based on pixel sizes.

    Parameters:
        full_image_path (Path): Path to the raster image.
        x_size (float): Width of each grid cell in **pixels**.
        y_size (float): Height of each grid cell in **pixels**.
        overlap_ratio (float): Overlap between tiles as a fraction (0 to 1).

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the grid polygons.
    """
    # Open raster
    with rasterio.open(full_image_path) as src:
        transform = src.transform  # Affine transform
        crs = src.crs  # Get CRS

        if crs is not None and crs.to_epsg() is None:
            # If it's UTM zone 16S, manually set the EPSG
            if "UTM zone 16S" in str(crs):
                epsg_code = "EPSG:32716"
                pass
            # Check for UTM zone 15S
            elif "UTM zone 15S" in str(crs):
                epsg_code = "EPSG:32715"
        else:
            epsg_code = f"EPSG:{crs.to_epsg()}"

        bounds = src.bounds  # Get bounds

        # Raster extent in world coordinates (meters or degrees)
        min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top

        # Get pixel sizes (resolution in X and Y directions)
        pixel_size_x = transform.a  # Pixel width in world units (meters or degrees)
        pixel_size_y = abs(transform.e)  # Pixel height (take absolute value)

        # Compute tile sizes in world coordinates
        x_size_world = x_size * pixel_size_x
        y_size_world = y_size * pixel_size_y

        # Compute overlap in world coordinates
        x_overlap_world = x_size_world * overlap_ratio
        y_overlap_world = y_size_world * overlap_ratio

        # Compute step size (tile size minus overlap)
        step_x = x_size_world - x_overlap_world
        step_y = y_size_world - y_overlap_world

        grid_cells = []
        tiles = []
        image_name = full_image_path.stem

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

                # Alternative coordinate-based naming if preferred
                # Using world coordinates rounded to a reasonable precision
                coord_tile_name = f"{image_name}_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

                grid_cells.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "tile_name": coord_tile_name,
                                   "image_name": image_name,
                                   })

        # Convert to GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(data=grid_cells, geometry=tiles, crs=epsg_code)

    return grid_gdf





def save_grid(grid_gdf: gpd.GeoDataFrame, output_path: Path):
    """
    Save a grid to a file in Shapefile, GeoJSON, or GPKG format.

    Parameters:
        grid_gdf (gpd.GeoDataFrame): Geodataframe containing grid polygons.
        output_path (Path): Path to save the output file (with extension).

    Returns:
        Path: Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Determine file format from extension
    file_format = output_path.suffix.lower()

    if file_format == ".shp":
        grid_gdf.to_file(output_path, driver="ESRI Shapefile")
    elif file_format == ".geojson":
        grid_gdf.to_file(output_path, driver="GeoJSON")
    elif file_format == ".gpkg":
        grid_gdf.to_file(output_path, driver="GPKG")
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use .shp, .geojson, or .gpkg.")

    print(f"Grid saved successfully: {output_path}")
    return output_path



def cut_geospatial_raster_with_grid(raster_path: Path,
                                    grid_gdf: gpd.GeoDataFrame,
                                    output_dir: Path,
                                    compression: str = "JP2",
                                    quality: int = 90) -> List[Path]:
    """
    Cut a geospatial raster into tiles using a grid and save as GeoTIFF.
    FIXME: Shit, using Rasterio is a stupid idea in python

    Parameters:
        raster_path (Path): Path to the input raster.
        grid_gdf (gpd.GeoDataFrame): Geodataframe containing grid polygons.
        output_dir (Path): Directory to save output tiles.

    Returns:
        List[Path]: List of saved tile file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    saved_tiles = []

    # Open the raster
    with rasterio.open(raster_path) as src:
        for idx, row in grid_gdf.iterrows():
            tile_geom = [row.geometry]  # Convert geometry to list for rasterio.mask
            logger.info(f"Cutting tile {idx}...")
            try:
                # Clip raster using the current tile geometry
                out_image, out_transform = mask(src, tile_geom, crop=True)

                # Update metadata
                out_meta = src.meta.copy()


                if compression == "JP2":
                    out_meta.update({
                        "driver": "JP2OpenJPEG",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "lossless": False,  # Enable lossless mode
                        "compress": compression,  # Use JPEG compression,
                        "quality": quality
                    })

                    tile_filename = output_dir / f"tile_{idx}.jp2"

                else:
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,

                    })
                    out_meta["compress"] = "LZW"

                    # Define tile output path
                    tile_filename = output_dir / f"tile_{idx}.tif"

                # Save the clipped tile
                with rasterio.open(tile_filename, "w", **out_meta) as dest:
                    dest.write(out_image)

                saved_tiles.append(tile_filename)
                print(f"Saved: {tile_filename}")

            except Exception as e:
                print(f"Failed to cut tile {idx}: {e}")

    return saved_tiles



def cut_geospatial_raster_with_grid_gdal(raster_path: Path, grid_gdf: gpd.GeoDataFrame, output_dir: Path,
                                         compression: str = "geotiff", quality: int = 95) -> List[Path]:
    """
    Cut a geospatial raster into tiles using a grid and save as GeoTIFF or JPEG 2000 using GDAL.

    Parameters:
        raster_path (Path): Path to the input raster.
        grid_gdf (gpd.GeoDataFrame): Geodataframe containing grid polygons.
        output_dir (Path): Directory to save output tiles.
        compression (str): Compression type ("LZW" for GeoTIFF, "JP2" for JPEG 2000).
        quality (int): JPEG 2000 compression quality (1-100, higher is better quality).

    Returns:
        List[Path]: List of saved tile file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    saved_tiles = []

    # Open the raster with GDAL
    raster_ds = gdal.Open(str(raster_path))
    geo_transform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    res = geo_transform[1]  # resolution, GSD
    for idx, row in grid_gdf.iterrows():
        tile_geom = row.geometry  # Extract geometry for current tile

        try:
            # Define output format
            if compression == "JP2":
                driver = gdal.GetDriverByName("JP2OpenJPEG")
                # tile_filename = output_dir / f"tile_{idx}.jp2"
                tile_filename = output_dir / f"{row['tile_name']}.jp2"
                options = [
                    "QUALITY=" + str(quality),
                    "COMPRESS=JP2",
                    "TILED=YES"
                ]
            else:
                driver = gdal.GetDriverByName("GTiff")
                tile_filename = output_dir / f"tile_{idx}.tif"
                tile_filename = output_dir / f"{row['tile_name']}.tif"

                options = [
                    "COMPRESS=LZW",
                    "TILED=YES"
                ]

            if not tile_filename.exists():

                x_min, y_min, x_max, y_max = tile_geom.bounds

                # Convert georeferenced coordinates to pixel coordinates
                x_offset, y_offset = world_to_pixel(geo_transform, x_min, y_max)
                x_offset_end, y_offset_end = world_to_pixel(geo_transform, x_max, y_min)

                # Calculate the width and height in pixels
                x_size = x_offset_end - x_offset
                y_size = y_offset_end - y_offset

                return_value = gdal.Translate(destName=str(tile_filename),
                                              srcDS=raster_path,
                                              projWin=(x_min, y_max, x_max, y_min),
                                              format='GTiff',
                                              creationOptions=[
                                                  'COMPRESS=LZW',  # Apply JPEG compression
                                                  # 'JPEG_QUALITY=100',  # Set JPEG quality (1-100)
                                                  # Use YCbCr color space for better compression
                                                  'TILED=YES',  # Enable tiling for optimized access
                                                  'BLOCKXSIZE=256',  # Set tile width
                                                  'BLOCKYSIZE=256'  # Set tile height
                                              ],
                                              xRes=res,
                                              yRes=-res,
                                              )
            else:
                # Tile already exists
                pass

            saved_tiles.append(tile_filename)

        except Exception as e:
            logger.error(f"Failed to cut tile {idx}: {e}")

    return saved_tiles
