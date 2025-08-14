"""
Herdnet geospatial inference script to find dot-instances of iguanas in orthomosaics.


configs based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evalustes performances etc

"""
import tempfile

from pandas import DataFrame
from typing import Any

import PIL
import albumentations as A
import geopandas as gpd
import hydra
import numpy
import numpy as np
import os
import pandas
import pandas as pd
import torch
import typing
from PIL import Image
from datetime import date
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader

from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.image_manipulation import convert_tiles_to
# from active_learning.util.geospatial_slice import get_tiles
from active_learning.util.projection import get_geotransform, pixel_to_world_point, \
    get_orthomosaic_crs
from animaloc.utils.useful_funcs import mkdir
from animaloc.vizual import draw_points, draw_text
from com.biospheredata.converter.HastyConverter import ImageFormat
from tools.inference_test import inference

Image.MAX_IMAGE_PIXELS = None  # Disable the limit of image size in PIL


def current_date():
    ''' To get current date in YYYYMMDD format '''
    today = date.today().strftime('%Y%m%d')
    return today

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

    def insert_jpg_folder(path_str):
        p = Path(path_str)
        new_dir = p.parent / "jpg"
        new_filename = p.stem + ".jpg"
        return str(new_dir / new_filename)

    gdf_tiles["jpg"] = gdf_tiles["slice_path"].apply(insert_jpg_folder)

    return gdf_tiles, output_dir_jpg



def herdnet_geospatial_inference(cfg: DictConfig,
                                 images_dir: Path,
                                 tiff_dir: Path,
                                 # gdf_tiles: gpd.GeoDataFrame,
                                 output_dir: Path,
                                 plain_inference=True,
                                 ts=256) -> tuple[DataFrame, dict[str, gpd.GeoDataFrame]]:
    """
    Inference on a geospatial dataset

    :param ts: Thumbnail size
    :param cfg:
    :param plain_inference:
    :return:
    """
    # retrieving the test part and model configuration use at the time of training
    df_detections = inference(cfg, plain_inference=plain_inference)
    plots_path = output_dir / 'plots'
    plots_path.mkdir(exist_ok=True, parents=True)

    # removing detection with Nan score,xy,
    df_detections = df_detections.dropna(subset=['scores', 'x', 'y'])

    logger.info("5) plot the detections")
    print('Exporting plots and thumbnails ...')


    dest_thumb = output_dir / 'thumbnails'
    dest_thumb.mkdir(exist_ok=True, parents=True)
    img_names = numpy.unique(df_detections['images'].values).tolist()

    dict_gdf_detections = {}

    for img_name in img_names:
        image_path = Path(images_dir) / img_name
        img = PIL.Image.open(image_path)
        if img.format != 'JPEG':
            img = img.convert("RGB")

        img_cpy = img.copy()
        pts = list(df_detections[df_detections['images'] == img_name][['y', 'x']].to_records(index=False))

        # get mask
        mask = df_detections['images'] == img_name

        # replace jpg with with tif to retrieve geospatial information
        tif_name = Path(img_name).with_suffix('.tif').name
        geo_transform = get_geotransform(Path(tiff_dir) / tif_name)
        # One-step transformation to geometry
        df_detections.loc[mask, 'geometry'] = df_detections.loc[mask].apply(
            lambda row: pixel_to_world_point(geo_transform, row.x, row.y),
            axis=1
        )

        df_detections.to_csv(output_dir / f'detections_{img_name[:-4]}.csv', index=False)

        gdf_detections = gpd.GeoDataFrame(df_detections.loc[mask],
                                          geometry='geometry',
                                          crs=get_orthomosaic_crs(Path(tiff_dir) / tif_name))

        gdf_detections.to_file(output_dir / f'detections_{img_name[:-4]}.geojson', driver='GeoJSON')
        dict_gdf_detections[str(output_dir / f'detections_{img_name[:-4]}.geojson')] =  gdf_detections

        # Visualise
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(plots_path, img_name), format="JPEG", quality=95)

        # Create and export thumbnails
        sp_score = list(
            df_detections[df_detections['images'] == img_name][['species', 'scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
            off = ts // 2
            coords = (x - off, y - off, x + off, y + off)
            if all(np.isnan(coords)):
                logger.warning(f"Coords are all NaN: {coords}, skipping")
                continue
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", position=(10, 5), font_size=int(0.08 * ts))
            thumbnail.save(dest_thumb / f"{img_name[:-4]}_{i}.JPG")

    logger.info(f'Testing done, wrote results to: {output_dir}')
    return df_detections, dict_gdf_detections


def get_config(config_name="config_2025_04_14_dla"):
    # Initialize hydra
    hydra.initialize(config_path="../configs")

    # Compose the config
    cfg = hydra.compose(config_name=config_name)

    return cfg


def geospatial_inference_pipeline(orthomosaic_path: Path,
                                  hydra_cfg: DictConfig,
                                  min_score: float = 0.0):
    """
    Main function to run the geospatial inference pipeline.
    :param orthomosaic_path: Path to the orthomosaic image.
    :return: None
    """
    logger.info(f"Running geospatial inference on {orthomosaic_path}")
    tile_images_path = orthomosaic_path.with_name(orthomosaic_path.stem + '_tiles')
    output_dir = orthomosaic_path.with_name(orthomosaic_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf_tiles, images_dir = get_tiles(
        orthomosaic_path=orthomosaic_path,
        output_dir=Path(tile_images_path),
        tile_size=hydra_cfg.datasets.test.tile_size
    )

    # ### ====== debugging ====== ####
    # images_dir = Path("/raid/cwinkelmann/mosaic_prediction/sample/FLPC01-07_22012021-orthomosaic_tiles/jpg") # TODO it should work with tiff images too
    # tile_images_path = Path("/raid/cwinkelmann/mosaic_prediction/sample/FLPC01-07_22012021-orthomosaic_tiles")
    # ### ====== debugging ====== ####

    # TODO throw the images into the config.
    hydra_cfg.datasets.test.root_dir = images_dir
    df_detections, dict_gdf_detections = herdnet_geospatial_inference(cfg=hydra_cfg,
                                                 images_dir=images_dir,
                                                 tiff_dir=Path(tile_images_path),
                                                 output_dir=output_dir,
                                                 plain_inference=True,
                                                 ts=512)

    df_detections.to_csv(output_dir / f'detections_{Path(orthomosaic_path).stem}.csv', index=False)
    df_detections = df_detections[df_detections['scores'] > min_score]  # filter out low confidence detections


    gdf_detections = gpd.GeoDataFrame(df_detections,
                                      geometry='geometry',
                                      crs=get_orthomosaic_crs(orthomosaic_path))

    gdf_detections.to_file(output_dir / f'detections_{Path(orthomosaic_path).stem}.geojson', driver='GeoJSON')
    logger.info(tile_images_path / f'detections_{Path(orthomosaic_path).stem}.geojson')

    return gdf_detections


if __name__ == '__main__':
    # run_with_config()

    # hydra_cfg = get_config(config_name="config_2025_08_08_dla34_train_val_inverted_val_corrected")
    hydra_cfg = get_config(config_name="config_2025_08_10_dinov2_floreana_fernandia_all_val_fernandina")

    # orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    # images_path = Path("/Users/christian/data/training_data/2025_02_22_HIT/FMO02_sample")
    #
    #
    # orthomosaic_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FCD01-02-03_20122021.tif')
    # orthomosaic_path = Path('/Volumes/2TB/Manual_Counting/Pix4D Orthomosaics/FNE01-02-03_19122021/exports/FNE01-02-03_19122021-orthomosaic.tiff')
    # # images_path = Path("Fer_FCD01-02-03_tiles")
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Scris_SRLN01_13012020.tif')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    #
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPC01-07_22012021/FLPC01-07_22012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #

    # demo
    orthomosaic_path = Path(
        '/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021.tif')
    gdf_predictions = geospatial_inference_pipeline(orthomosaic_path,
                                  hydra_cfg=hydra_cfg,
                                  min_score = 0.1
                                  )

    # orthomosaic_path = Path(
    #     '/raid/cwinkelmann/mosaic_prediction/FLPC01-07_22012021/FLPC01-07_22012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    # orthomosaic_path = Path(
    #     #"/home/cwinkelmann/work/active_learning/scripts/inferencing/FLPC01-07_22012021-orthomosaic_tiles/FLPC01-07_22012021-orthomosaic.tiff")
    #     "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPC01-07_22012021/FLPC01-07_22012021-orthomosaic.tiff")
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    #
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FNJ02-03-04_19122021-Pix4D-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FCD01-07_04052024_orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # # # TODO this creates problemes: all predictions have a confidence of 1.0
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Mar/MNW01-5_07122021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/SRLx01-07_05012021/exports/SRLx01-07_05012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    #
    # orthomosaic_path = Path(
    #     #"/home/cwinkelmann/work/active_learning/scripts/inferencing/FLPC01-07_22012021-orthomosaic_tiles/FLPC01-07_22012021-orthomosaic.tiff")
    #     "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPC01-07_22012021/FLPC01-07_22012021-orthomosaic.tiff")
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    #
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FNJ02-03-04_19122021-Pix4D-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FCD01-07_04052024_orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # # # TODO this creates problemes: all predictions have a confidence of 1.0
    # orthomosaic_path = Path(
    #     '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Mar/MNW01-5_07122021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/SRLx01-07_05012021/exports/SRLx01-07_05012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    #
    #
    # dd_paths = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Flo")
    # orthomosaic_list = [l for l in dd_paths.glob('*.tif') if l.is_file() and not l.name.startswith('.')]
    # for o in orthomosaic_list:
    #
    #     geospatial_inference_pipeline(o, hydra_cfg=hydra_cfg)
    #

    # from concurrent.futures import ProcessPoolExecutor
    # from pathlib import Path
    #
    # # dd_paths = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/")
    # # dd_paths = Path("/raid/cwinkelmann/Counts Nov 2022/SCruz_Mosquera_C1211122")
    # dd_paths = Path("/raid/cwinkelmann/Counts Nov 2022/cog")
    # predictions_path = dd_paths / "predictions"
    # predictions_path.mkdir(exist_ok=True, parents=True)
    # orthomosaic_list = [l for l in dd_paths.glob('*.tif') if l.is_file() and not l.name.startswith('.')]
    # # orthomosaic_list_2 = [l for l in dd_paths.glob('Flo_*.tif') if l.is_file() and not l.name.startswith('.')]
    # logger.info(f"Found: {len(orthomosaic_list)}")
    # # find predictions
    # already_finished_predictions = [l for l in predictions_path.glob('*.geojson') if
    #                                 l.is_file() and not l.name.startswith('.')]
    #
    # # remove already finished predictions from the list of orthomosaics
    # orthomosaic_list = [o for o in orthomosaic_list if not any(o.stem in p.stem for p in already_finished_predictions)]
    #
    # logger.info(f"Processing: {len(orthomosaic_list)}")
    #
    #
    # def process_orthomosaic(args):
    #     orthomosaic_path, hydra_cfg, predictions_path = args
    #     return geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)
    #
    #
    # with ProcessPoolExecutor(max_workers=7) as executor:
    #     args_list = [(o, hydra_cfg, predictions_path) for o in orthomosaic_list]
    #     results = list(executor.map(process_orthomosaic, args_list))
    #
    # for res in results:
    #     logger.info(f"Processed: {res}")


    # predictions_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/") / "predictions"
    # dd_paths = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/")
    # orthomosaic_list = [l for l in dd_paths.glob('Flo_FLPC0*.tif') if l.is_file() and not l.name.startswith('.')]
    # for o in orthomosaic_list:
    #
    #     geospatial_inference_pipeline(o, hydra_cfg=hydra_cfg)
