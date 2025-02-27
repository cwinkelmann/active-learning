"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
import typing

import csv
from loguru import logger

from pathlib import Path

from active_learning.util.Annotation import convert_shapefile2usable
from active_learning.util.converter import ifa_point_shapefile_to_hasty
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer
from active_learning.util.image_manipulation import convert_image, convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords, convert_jpeg_to_geotiff_coords, project_gdfcrs
from com.biospheredata.converter.HastyConverter import ImageFormat
from PIL import Image

from geospatial_transformations import convert_to_cog, get_gsd, get_geotiff_compression
import geopandas as gpd

# Herdnet model





if __name__ == "__main__":
    annotations_base_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Counts QGIS')
    orthomosaic_base_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Drone Deploy orthomosaics')

    # Counter for the number of annotations
    from collections import Counter
    stats = Counter()
    annotations = annotations_base_path.glob('**/*.shp')
    for annotataion in annotations:
        possible_orthopath = orthomosaic_base_path / f"{annotataion.parent.stem}.tif"

        if possible_orthopath.exists():
            logger.info(f"Found orthomosaic for {annotataion.parent} , {annotataion.stem}")
            stats.update(["found"])
        else:
            logger.warning(f"Could not find orthomosaic for {annotataion.parent.name}, {annotataion.stem}")
            stats.update(["NOT found"])

    logger.info(stats)
    """
    annotations_files = [
        Path('/Users/christian/data/Manual Counting/Fer_FNF02_19122021/Fer_FNF02_19122021 counts.shp'),
        # TODO look the following up on my disk
        # Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Counts QGIS/Fer_FPM05_24012023/Fer_FPM05_24012023 counts.shp'),
        Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Counts QGIS/Isa_ISPDA01_17012023/Isa_ISPDA01_17012023 counts.shp')
    ]
    orthomosaic_paths = [
        Path("/Users/christian/data/Manual Counting/Fer_FNF02_19122021/Fer_FNF02_19122021.tif")
    ]
    """

    gdfs = [gpd.read_file(annotations_file) for annotations_file in annotations_files]
    icl = [ifa_point_shapefile_to_hasty(gdf=gdf, images_path=orthomosaic_path) for gdf, orthomosaic_path in zip(gdfs, orthomosaic_paths)]

    # according to the location we can create datasets now



        # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file


    output_dir = Path("/Users/christian/data/Manual Counting/Fer_FNF02_19122021/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(orthomosaic_path).stem}_cog.tif"
    convert_to_cog(orthomosaic_path, output_file)


    original_compression = get_geotiff_compression(orthomosaic_path)
    cog_compression = get_geotiff_compression(output_file)
    logger.info(f"Original compression: {original_compression}, COG compression: {cog_compression}")

    gsd_x, gsd_y = get_gsd(output_file)
    if round(gsd_x, 8) == 0.00932648:
        logger.warning("You are either a good pilot or you wasted quality by using drone deploy, which caps images at about 0.93cm/px, compresses images a lot throws away details")
    logger.info(f"Ground Sampling Distance (GSD): {gsd_x:.2f} x {gsd_y:.2f} m/px")

    if annotations_file is not None:
        gdf_pixel = convert_gdf_to_jpeg_coords(gdf, tiff_path=output_file)

        gdf_pixel

        # gdf_geo = convert_jpeg_to_geotiff_coords(gdf_pixel, tiff_path=image_path, jpg_path=jpg_path)


    # Run the function
    grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(orthomosaic_path),
                                                     x_size=5120,
                                                     y_size=5120,
                                                     overlap_ratio=0.0)
    tiles_output_dir = output_dir / "tiles"
    tiles_output_dir.mkdir(parents=True, exist_ok=True)
    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf,
                       output_dir=tiles_output_dir)
    tiles = slicer.slice_very_big_raster()

    format = ImageFormat.JPG

    tiles = remove_empty_tiles(tiles)

    converted_tiles = convert_tiles_to(tiles=tiles, format=format, output_dir=tiles_output_dir) # TODO make this into a list comprehension

    print(converted_tiles)

    if annotations_file is None:
        # This is a bit of a hack, but we can use the function to save the tiles to a CSV file
        output_csv = output_dir / "herdnet_fake.csv"

        save_tiles_to_csv(converted_tiles, output_csv, species = "iguana", label = 1)

    else:
        raise NotImplementedError("convert the annotations to the new coordinates and save them to a CSV file")