"""
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

import torch
from PIL import Image
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader

import animaloc
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.projection import get_geotransform, pixel_to_world_point, \
    get_orthomosaic_crs
from animaloc.data.transforms import DownSample
from animaloc.eval import PointsMetrics, BoxesMetrics
from animaloc.utils.useful_funcs import mkdir
from animaloc.vizual import draw_points, draw_text
from tools.inference_test import _set_species_labels, _get_collate_fn, _build_model, _define_evaluator

Image.MAX_IMAGE_PIXELS = None  # Disable the limit of image size in PIL

def geospatial_detection(orthomosaic_path,
                          output_dir, tile_size=2500):
    """
    Helper to to create a grid of tiles from an orthomosaic and slice it into smaller images.
    :param orthomosaic_path:
    :param output_dir:
    :param tile_size:
    :return:
    """
    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_gdf = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0)

    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf, output_dir=output_dir)
    gdf_tiles = slicer.slice_very_big_raster()

    # TODO remove the empty tiles

    return gdf_tiles

@hydra.main(config_path='../configs', config_name="config_2025_04_14_dla")
def run_with_config(cfg: DictConfig, plain_inference = True):
    """ wrapper to run the inference with hydra configs"""

    cfg = cfg.test # TODO move this to the other configs
    root_dir = Path(cfg.dataset.root_dir)
    main(cfg, root_dir)

def main(cfg: DictConfig,
         plain_inference = True,
         ts = 256) -> None:
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


    cls_dict = dict(cfg.dataset.class_def)
    cls_names = list(cls_dict.values())


    # Code for the case of doing just inference
    if plain_inference:
        img_names = [i for i in os.listdir(cfg.dataset.root_dir)
                     if i.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg', ".tiff", ".tif"))]
        n = len(img_names)
        if n == 0:
            raise FileNotFoundError(f"No images found in {cfg.dataset.root_dir}.")
        test_df = pandas.DataFrame(data={'images': img_names, 'x': [0] * n, 'y': [0] * n, 'labels': [1] * n})
        test_df["species"] = "iguana"

    else:
        test_df = pandas.read_csv(cfg.dataset.csv_file)
        _set_species_labels(cls_dict, df=test_df)

    # TODO why is this definehere and the configs is not used?
    # TODO build different dataset
    test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
        csv_file = test_df,
        root_dir = cfg.dataset.root_dir,
        albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
        end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
        )
    
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

    # Start testing
    logger.info(f'Starting inferencing ...')
    out = evaluator.evaluate(viz=False)
    logger.info(f'Done with predictions ...')

    # Save results
    plots_path = current_directory / 'plots'
    plots_path.mkdir(exist_ok=True, parents=True)

    logger.info("4) detections")
    detections =  evaluator.detections
    logger.info(f"Num detections: {len(detections)}")
    detections['species'] = detections['labels'].map(cls_dict)

    logger.warning(f"Manually scale up the coordinates by a factor of down_ratio: {down_ratio}")
    detections['x'] = detections['x'] * down_ratio
    detections['y'] = detections['y'] * down_ratio
    detections.to_csv(current_directory / 'detections.csv', index=False)

    logger.info("5) plot the detections")
    print('Exporting plots and thumbnails ...')
    dest_plots = plots_path
    mkdir(dest_plots)

    dest_thumb = current_directory / 'thumbnails'
    dest_thumb.mkdir(exist_ok=True, parents=True)
    img_names = numpy.unique(detections['images'].values).tolist()

    for img_name in img_names:
        orthomosaic_path = Path(cfg.dataset.root_dir) / img_name
        img = PIL.Image.open(orthomosaic_path)
        if img.format != 'JPEG':
            img = img.convert("RGB")

        img_cpy = img.copy()
        pts = list(detections[detections['images'] == img_name][['y', 'x']].to_records(index=False))

        # get mask
        mask = detections['images'] == img_name
        # TODO finish this
        geo_transform = get_geotransform(Path(cfg.dataset.root_dir)/ img_name)
        # One-step transformation to geometry
        detections.loc[mask, 'geometry'] = detections.loc[mask].apply(
            lambda row: pixel_to_world_point(geo_transform, row.x, row.y),
            axis=1
        )


        logger.warning(f"The coordinates are manually upscaled by a factor of down_ratio: {down_ratio}")
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(dest_plots, img_name), format="JPEG", quality=95)

        # Create and export thumbnails
        sp_score = list(detections[detections['images'] == img_name][['species', 'scores']].to_records(index=False))
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
    return detections




def get_config():
    # Initialize hydra
    hydra.initialize(config_path="../configs")

    # Compose the config
    cfg = hydra.compose(config_name="config_2025_04_14_dla")

    return cfg


if __name__ == '__main__':
    # run_with_config()

    hydra_cfg = get_config()
    images_path = Path("/Users/christian/data/training_data/2025_02_22_HIT/FMO02_sample")
    images_path = Path("Fer_FCD01-02-03_tiles")

    orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    orthomosaic_path = Path("//Users/christian/data/orthomosaics/Fer_FCD01-02-03_20122021.tif")

    # gdf_tiles = geospatial_detection(orthomosaic_path=orthomosaic_path,
    #                      output_dir=images_path)

    hydra_cfg.dataset.root_dir = images_path
    detections = main(cfg=hydra_cfg, plain_inference=True, ts=512)

    gdf_detections = gpd.GeoDataFrame(detections,
                                      geometry='geometry',
                                      crs=get_orthomosaic_crs(orthomosaic_path))

    gdf_detections.to_file(f'detections_{Path(orthomosaic_path).stem}.geojson', driver='GeoJSON')