"""
Herdnet geospatial inference script to find dot-instances of iguanas in orthomosaics.


configs based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evalustes performances etc

"""

import PIL
import albumentations as A
import hydra
import numpy
import numpy as np
import os
import pandas
import geopandas as gpd
import pandas as pd

import torch
from PIL import Image
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
import wandb

from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.image_manipulation import remove_empty_tiles, convert_tiles_to
from active_learning.util.projection import get_geotransform, pixel_to_world_point, \
    get_orthomosaic_crs
from animaloc.data.transforms import DownSample
from animaloc.eval import PointsMetrics, BoxesMetrics
from animaloc.utils.useful_funcs import mkdir
from animaloc.vizual import draw_points, draw_text
from datetime import date, datetime
import animaloc
from com.biospheredata.converter.HastyConverter import ImageFormat

from tools.inference_test import _set_species_labels, _get_collate_fn, _build_model, _define_evaluator


Image.MAX_IMAGE_PIXELS = None  # Disable the limit of image size in PIL

def current_date():
    ''' To get current date in YYYYMMDD format '''
    today = date.today().strftime('%Y%m%d')
    return today

def get_tiles(orthomosaic_path,
              output_dir, tile_size=1250):
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

    converted_tiles = convert_tiles_to(tiles=list(slicer.gdf_slices.slice_path), format=ImageFormat.JPG,
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

# @hydra.main(config_path='../configs', config_name="config_2025_04_14_dla")
# def run_with_config(cfg: DictConfig, plain_inference = True):
#     """ wrapper to run the inference with hydra configs"""
#
#     cfg = cfg.test # TODO move this to the other configs
#     root_dir = Path(cfg.dataset.root_dir)
#     herdnet_geospatial_inference(cfg, images_dir=root_dir)


def herdnet_geospatial_inference(cfg: DictConfig,
                                images_dir: Path,
                                tiff_dir: Path,
                                # gdf_tiles: gpd.GeoDataFrame,
                                 plain_inference = True,
                                 ts = 256) -> pd.DataFrame:
    """
    Inference on a geospatial dataset

    :param cfg:
    :param plain_inference:
    :return:
    """
    # retrieving the test part of the configs

    current_directory = Path(os.getcwd())
    logger.info(f"Current directory: {current_directory}")

    down_ratio = cfg.model.kwargs.down_ratio
    device = torch.device(cfg.device_name)

    if cfg.wandb_flag:
        # Set up wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = dict(
                model = cfg.model,
                down_ratio = down_ratio,
                num_classes = cfg.dataset.num_classes,
                threshold = cfg.evaluator.threshold,
                images_dir = images_dir
                )
            )

        date = current_date()
        wandb.run.name = f'{date}_' + cfg.wandb_run + f'_RUN_{wandb.run.id}'


    cls_dict = dict(cfg.dataset.class_def)
    cls_names = list(cls_dict.values())


    # Code for the case of doing just inference TODO it seems somethings is wrong with the tiles because the result in the end is wrong
    if plain_inference:
        img_names = [i.name for i in images_dir.glob("*.jpg") if not i.name.startswith('.')]
        n = len(img_names)
        if n == 0:
            raise FileNotFoundError(f"No images found in {images_dir}.")
        test_df = pandas.DataFrame(data={'images': img_names, 'x': [0] * n, 'y': [0] * n, 'labels': [1] * n})

        test_df["species"] = "iguana"
        # if len(gdf_tiles) != len(img_names):
        #     raise ValueError("wrong tile count")
        # TODO with the FolderDataset it should be fine to pass an empty dataframe

        # TODO pass an empty csv/dataframe because theFolderDataaset should be fine with it.
        test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
            csv_file = test_df,
            root_dir = images_dir,
            albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
            end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
            )
    # TODO This part is actually correct
    else:
        test_df = pandas.read_csv(cfg.dataset.csv_file)
        _set_species_labels(cls_dict, df=test_df)

        test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
            csv_file = test_df,
            root_dir = images_dir,
            albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
            end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
            )

    # TODO a batch sampler should be possble too?
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
                                 collate_fn=_get_collate_fn(cfg))

    # Build the trained model
    logger.info('Building the trained model ...')
    model = _build_model(cfg).to(device)

    # Build the evaluator
    logger.info('Preparing for testing ...')
    anno_type = cfg.dataset.anno_type

    if anno_type == 'point':
        metrics = PointsMetrics(radius = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(iou = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    else:
        raise NotImplementedError

    logger.info(f"Define Evaluator: {anno_type}")
    evaluator = _define_evaluator(model, test_dataloader, metrics, cfg)

    # Save results
    plots_path = tiff_dir / 'plots'
    plots_path.mkdir(exist_ok=True, parents=True)

    # Start testing
    logger.info(f'Starting inferencing ...')
    out = evaluator.evaluate(viz=False, wandb_flag=cfg.wandb_flag)
    logger.info(f'Done with predictions ...')
    df_detections =  evaluator.detections

    logger.info("4) detections")
    logger.info(f"Num detections: {len(df_detections)}")
    df_detections['species'] = df_detections['labels'].map(cls_dict)

    logger.warning(f"Manually scale up the coordinates by a factor of down_ratio: {down_ratio}")
    df_detections['x'] = df_detections['x'] * down_ratio
    df_detections['y'] = df_detections['y'] * down_ratio
    df_detections.to_csv(tiff_dir / f'{tiff_dir.name}_detections_pi_{plain_inference}.csv', index=False)

    df_detections = pd.read_csv(tiff_dir / f'{tiff_dir.name}_detections_pi_{plain_inference}.csv')

    logger.info("5) plot the detections")
    print('Exporting plots and thumbnails ...')
    dest_plots = plots_path
    mkdir(dest_plots)

    dest_thumb = tiff_dir / 'thumbnails'
    dest_thumb.mkdir(exist_ok=True, parents=True)
    img_names = numpy.unique(df_detections['images'].values).tolist()

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
        geo_transform = get_geotransform(Path(tiff_dir)/ tif_name)
        # One-step transformation to geometry
        df_detections.loc[mask, 'geometry'] = df_detections.loc[mask].apply(
            lambda row: pixel_to_world_point(geo_transform, row.x, row.y),
            axis=1
        )
        gdf_detections = gpd.GeoDataFrame(df_detections.loc[mask],
                                          geometry='geometry',
                                          crs=get_orthomosaic_crs(Path(tiff_dir)/ tif_name))
        gdf_detections.to_file(dest_plots / f'detections_{img_name[:-4]}.geojson', driver='GeoJSON')

        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(dest_plots, img_name), format="JPEG", quality=95)

        # Create and export thumbnails
        sp_score = list(df_detections[df_detections['images'] == img_name][['species', 'scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
            off = ts // 2
            # TODO the fact this fails if an image is empty shows the code was never evaluated with empty images/or never predicted nothing even if the image was empty
            coords = (x - off, y - off, x + off, y + off)
            if all(np.isnan(coords)):
                logger.warning(f"Coords are all NaN: {coords}, skipping")
                continue
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", position=(10, 5), font_size=int(0.08 * ts))
            thumbnail.save(os.path.join(dest_thumb, img_name[:-4] + f'_{i}.JPG'))



    logger.info(f'Testing done, wrote results to: {os.getcwd()}')
    return df_detections




def get_config(config_name="config_2025_04_14_dla"):
    # Initialize hydra
    hydra.initialize(config_path="../configs")

    # Compose the config
    cfg = hydra.compose(config_name=config_name)

    return cfg

def geospatial_inference_pipeline(orthomosaic_path: Path, hydra_cfg):
    """
    Main function to run the geospatial inference pipeline.
    :param orthomosaic_path: Path to the orthomosaic image.
    :return: None
    """
    logger.info(f"Running geospatial inference on {orthomosaic_path}")
    tile_images_path = orthomosaic_path.with_name(orthomosaic_path.stem + '_tiles')

    tile_images_path.mkdir(exist_ok=True, parents=True)

    gdf_tiles, images_dir = get_tiles(
        orthomosaic_path=orthomosaic_path,
        output_dir=tile_images_path, tile_size=5000
    )

    #### ====== debugging ====== ####
    # images_dir = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/FLPC01-07_orthomosaic_herdnet/jpg_tiles")
    # tile_images_path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/FLPC01-07_orthomosaic_herdnet/tiff_tiles")
    #### ====== debugging ====== ####

    # hydra_cfg.dataset.root_dir = images_dir
    df_detections = herdnet_geospatial_inference(cfg=hydra_cfg,
                                              # gdf_tiles=gdf_tiles,
                                              # TODO solve the issue when tiles are empty or near ampty
                                              images_dir=images_dir,
                                              tiff_dir=tile_images_path,
                                              plain_inference=True,
                                              ts=512)

    # df_detections = pd.read_csv('detections.csv')


    gdf_detections = gpd.GeoDataFrame(df_detections,
                                      geometry='geometry',
                                      crs=get_orthomosaic_crs(orthomosaic_path))

    raise Exception(f"TODO: check if there are prediction without any geometry, label or score, if so remove them")
    raise Exception(f"add a unique id, otherwise it will be hard to track corrections later in the pipeline")

    gdf_detections.to_file(tile_images_path / f'detections_{Path(orthomosaic_path).stem}.geojson', driver='GeoJSON')
    logger.info(tile_images_path / f'detections_{Path(orthomosaic_path).stem}.geojson')


if __name__ == '__main__':
    # run_with_config()

    hydra_cfg = get_config()

    # orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    # images_path = Path("/Users/christian/data/training_data/2025_02_22_HIT/FMO02_sample")
    #
    #
    # orthomosaic_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FCD01-02-03_2021/FCD01-02-03_20122021.tif')
    # # images_path = Path("Fer_FCD01-02-03_tiles")
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)

    # orthomosaic_path = Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Scris_SRLN01_13012020.tif')
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg=hydra_cfg)


    # orthomosaic_path = Path(
    #     "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPL01-02_28012023_Tiepoints-orthomosaic.tiff")
    # geospatial_inference_pipeline(orthomosaic_path, hydra_cfg)
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



    dd_paths = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Flo")
    orthomosaic_list = [l for l in dd_paths.glob('*.tif') if l.is_file() and not l.name.startswith('.')]
    for o in orthomosaic_list:

        geospatial_inference_pipeline(o, hydra_cfg=hydra_cfg)

    agistoft_paths = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/Flo/")
    orthomosaic_list = [l for l in agistoft_paths.glob('*.tif') if l.is_file() and not l.name.startswith('.')]
    for o in orthomosaic_list:

        geospatial_inference_pipeline(o, hydra_cfg=hydra_cfg)