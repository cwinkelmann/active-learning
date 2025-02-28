"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
import typing

import csv
from loguru import logger

from pathlib import Path

from active_learning.util.Annotation import convert_shapefile2usable
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid, \
    save_world_file_json
from active_learning.util.geospatial_slice import GeoSlicer
from active_learning.util.image_manipulation import convert_image, convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords, convert_jpeg_to_geotiff_coords, project_gdfcrs
from com.biospheredata.converter.HastyConverter import ImageFormat
from PIL import Image

from geospatial_transformations import get_gsd, get_geotiff_compression
import geopandas as gpd

# Herdnet model


def save_tiles_to_csv(tiles, output_csv: Path, species="iguana", label=1):
    """
    Save raster tile filenames to a CSV file with additional metadata.

    Parameters:
        tiles (list): List of tile filenames (Paths or strings).
        output_csv (Path): Path to the output CSV file.
        species (str): Default species name (default: "iguana").
        label (int): Default label (default: 1).

    Returns:
        Path: Path to the saved CSV file.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Open CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["images", "x", "y", "species", "labels"])

        # Iterate through tiles and write rows
        for tile in tiles:
            filename = Path(tile).name  # Extract just the filename
            x, y = 0, 0  # Default placeholders (modify if coordinates exist)
            writer.writerow([filename, x, y, species, label])

    print(f"CSV saved to: {output_csv}")
    return output_csv





if __name__ == "__main__":
    annotations_file = None
    # annotations_file = Path('/Users/christian/data/Manual Counting/Fer_FNF02_19122021/Fer_FNF02_19122021 counts.shp')
    annotations_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNE02_19122021 counts.geojson')
    orthomosaic_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL/Fer/Fer_FNE02_19122021.tif")

    island_code = orthomosaic_path.parts[-2]
    tile_folder_name = orthomosaic_path.stem
    output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL_TILES") / island_code / tile_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = 1500

    if annotations_file is not None:
        gdf_points = gpd.read_file(annotations_file)

        gdf_points["image_name"] = orthomosaic_path.name
        # incase the orthomosaic has a different CRS than the annotations
        # gdf_points = project_gdfcrs(gdf_points, orthomosaic_path)

        # project the global coordinates to the local coordinates of the orthomosaic
        gdf_local = convert_gdf_to_jpeg_coords(gdf_points, orthomosaic_path)
        # create an ImageCollection of the annotations
        gdf_local
        # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file


    cog_compression = get_geotiff_compression(orthomosaic_path)
    logger.info(f"COG compression: {cog_compression}")

    gsd_x, gsd_y = get_gsd(orthomosaic_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning("You are either a precise pilot or you wasted quality by using drone deploy, which caps images at about 0.93cm/px, compresses images a lot throws away details")
    logger.info(f"Ground Sampling Distance (GSD): {100*gsd_x:.3f} x {100*gsd_y:.3f} cm/px")

    if annotations_file is not None:
        # TODO this is probably not necessary
        gdf_pixel = convert_gdf_to_jpeg_coords(gdf_points, tiff_path=orthomosaic_path)

        # TODO check why the expert column is empty
        gdf_pixel

        # gdf_geo = convert_jpeg_to_geotiff_coords(gdf_pixel, tiff_path=image_path, jpg_path=jpg_path)


    # Run the function
    grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(orthomosaic_path),
                                                     x_size=tile_size,
                                                     y_size=tile_size,
                                                     overlap_ratio=0.0)

    grid_gdf.to_file(output_dir / "grid.geojson", driver='GeoJSON')

    # remove grid cells which don't contain points which are saved in gdf
    grid_gdf_filtered = grid_gdf[grid_gdf.geometry.apply(lambda poly: gdf_points.geometry.within(poly).any())]

    grid_gdf_filtered.to_file(output_dir / "tile_extend.geojson", driver='GeoJSON')

    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf_filtered,
                       output_dir=output_dir)
    tiles = slicer.slice_very_big_raster()

    # for tile in tiles:
    #    save_world_file_json(tile, output_dir)

    format = ImageFormat.JPG

    # TODO remove tiles which contain no pixels
    tiles = remove_empty_tiles(tiles)

    # TODO remove tiles which contain no annotations

    converted_tiles = convert_tiles_to(tiles=tiles, format=format, output_dir=output_dir) # TODO make this into a list comprehension

    print(converted_tiles)

    if annotations_file is None:
        # This is a bit of a hack, but we can use the function to save the tiles to a CSV file
        output_csv = output_dir / "herdnet_fake.csv"

        save_tiles_to_csv(converted_tiles, output_csv, species = "iguana", label = 1)

    else:
        projected_points_path = slicer.slice_annotations(gdf_points)
