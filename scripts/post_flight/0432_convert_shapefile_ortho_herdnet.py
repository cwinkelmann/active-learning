"""
Convert a shapefile and orthomosaic to image file which can be to test the geospatial inference pipeline.

This check if the full orthomosaic inference aligns with with the training

TODO work in Progress
"""
import copy

import geopandas as gpd
import pandas as pd
import typing
from loguru import logger
from pathlib import Path

from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.image_manipulation import convert_tiles_to
from active_learning.util.projection import project_gdfcrs, convert_gdf_to_jpeg_coords
from active_learning.util.rename import get_site_code
from com.biospheredata.converter.HastyConverter import ImageFormat

if __name__ == "__main__":

    # orthophoto path
    train_orthophoto = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff")

    train_shapefile = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Pix4Dmatic_Orthomosaic/counts/FLPC01-07_22012021-orthomosaic counts.shp")

    tile_output_dir = Path('/Users/christian/data/Manual Counting/temp/tiff_tiles')
    jpg_tile_output_dir = Path('/Users/christian/data/Manual Counting/temp/jpg_tiles')
    jpg_tile_output_dir.mkdir(parents=True, exist_ok=True)
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    gdf_train = gpd.read_file(train_shapefile)

    gdf_train["species"] = "iguana"

    label_id_mapping = {
        "iguana": 1,
        "hard_negative": 2
    }

    gdf_train_iguanas_utm = gdf_train.to_crs(epsg=32715)

    gdf_train = gpd.GeoDataFrame(gdf_train_iguanas_utm, geometry='geometry', crs='EPSG:32715')

    logger.info(f"Combined GDF CRS: {gdf_train.crs}")
    logger.info(f"Combined shape: {gdf_train.shape}")

    # map species to labels
    gdf_train['labels'] = gdf_train['species'].map(label_id_mapping)


    gdf_local = convert_gdf_to_jpeg_coords(gdf_train, train_orthophoto)

    grid_manager = GeoSpatialRasterGrid(train_orthophoto)
    gdf_grid = grid_manager.create_regular_grid(x_size=5000, y_size=5000, overlap_ratio=0.0)

    slicer = GeoSlicer(base_path=train_orthophoto.parent,
                           image_name=train_orthophoto.name,
                           grid=gdf_grid,
                           output_dir=tile_output_dir)
    slices = slicer.slice_very_big_raster(num_chunks=len(gdf_grid), num_workers=3)

    gdf_sliced_points = slicer.slice_annotations_regular_grid(gdf_train, gdf_grid)

    converted_tiles = convert_tiles_to(tiles=list(slicer.gdf_slices.slice_path), format=ImageFormat.JPG,
                                           output_dir=jpg_tile_output_dir)
    converted_tiles = [a for a in converted_tiles]
    logger.info(f"created {len(converted_tiles)} tiles in {jpg_tile_output_dir}")

    l_herdnet = []
    for tile_name, gdf_group in gdf_sliced_points.groupby(by="tile_name"):


        gdf_group["images"] = f"{tile_name}.{ImageFormat.JPG.value}"
        gdf_herdnet = copy.copy(gdf_group[["images", "local_pixel_x", "local_pixel_y", "species", "labels"]])
        gdf_herdnet = gdf_herdnet.rename(columns={"local_pixel_x": "x", "local_pixel_y": "y"})
        # if visualise_crops:
        #     vis_filename = vis_output_dir / f"{tile_name}.jpg"
        #     logger.info(f"Visualising {tile_name}")
        #     ax_s = visualise_image(image_path=output_obj_dir / tile_image_name, show=False,
        #                            title=f"Visualisation of {len(df_herdnet)} labels in {tile_image_name}")
        #     visualise_polygons(
        #         points=[shapely.Point(x, y) for x, y in zip(df_herdnet.local_pixel_x, df_herdnet.local_pixel_y)],
        #         labels=df_herdnet["species"], ax=ax_s, show=False, linewidth=6, markersize=10,
        #         filename=vis_filename)
        #
        #     plt.close()

        l_herdnet.append(gdf_herdnet)

    df_herdnet = pd.concat(l_herdnet, axis=0)

    df_herdnet.to_csv('/Users/christian/data/Manual Counting/temp/jpg_tiles/herdnet_tiles.csv', index=False)

