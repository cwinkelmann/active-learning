import geopandas as gpd
from loguru import logger
from pathlib import Path

from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.image_manipulation import convert_tiles_to
from com.biospheredata.converter.HastyConverter import ImageFormat


def get_tiles(orthomosaic_path,
              output_dir,
              tile_size=1250):
    """
    Helper to to create a grid of tiles from an orthomosaic and slice it into smaller images.
    :param orthomosaic_path:
    :param output_dir:
    :param tile_size:
    :return:
    """
    logger.info(f"Tiling {orthomosaic_path} into {tile_size}x{tile_size} tiles")
    output_dir_metadata = output_dir / 'metadata'
    output_dir_jpg = output_dir / 'jpg'
    output_dir_metadata.mkdir(parents=True, exist_ok=True)
    output_dir_jpg.mkdir(parents=True, exist_ok=True)

    filename = Path(f'grid_{orthomosaic_path.with_suffix(".geojson").name}')
    if not output_dir_metadata.joinpath(filename).exists():
        grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

        grid_gdf = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0)
        grid_gdf.to_file(output_dir_metadata / filename, driver='GeoJSON')
        grid_manager.gdf_raster_mask.to_file(
            output_dir_metadata / Path(f'raster_mask_{orthomosaic_path.with_suffix(".geojson").name}'),
            driver='GeoJSON')

    else:
        logger.info(f"Grid file {filename} already exists, skipping grid creation")
        grid_gdf = gpd.read_file(output_dir_metadata / filename)

    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf,
                       output_dir=output_dir)

    gdf_tiles = slicer.slice_very_big_raster()

    converted_tiles = convert_tiles_to(tiles=list(slicer.gdf_slices.slice_path),
                                       format=ImageFormat.JPG,
                                       output_dir=output_dir_jpg)
    converted_tiles = [a for a in converted_tiles]
    logger.info(f"created {len(converted_tiles)} tiles in {output_dir_jpg}")