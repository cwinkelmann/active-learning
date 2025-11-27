"""
Refactor this into a bigger class which abstracts geospatial predictions


"""
import pickle
import torch
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
# import torch
from sahi import AutoDetectionModel

# import required functions, classes
from sahi.utils.cv import read_image
from sahi.predict import get_sliced_prediction
from loguru import logger

from com.biospheredata.helper.image.identifier import get_image_id
from com.biospheredata.helper.image.manipulation.convert import convert_geotiff_to_jpg
from com.biospheredata.helper.image.manipulation.slice import slice_very_big_raster
from com.biospheredata.helper.sliced_detect_helper import coco_annotation_list_geopandas
from com.biospheredata.helper.image_annotation import AnnotationPrediction
from com.biospheredata.types.HastyAnnotation import HastyAnnotation


def filter_for_non_empty_images(base_path, image_names: list, threshold=0.9):
    """
    when the images are slices of orthomosaics there is a high probability at the boundary the image contains only a very tiny fraction of images

    @param base_path:
    @param image_names:
    @param threshold:
    @return:
    """
    from PIL import Image

    filtered_list = []

    for image_name in image_names:
        image_path = str(base_path.joinpath(image_name))
        im = Image.open(image_path)
        im2arr = np.array(im)
        tot = np.float64(np.sum(im2arr)) ## FIXME this explodes with some geotiffs
        if tot / (im2arr.size * 255) > threshold:
            logger.info(f"ratio of pixels to black: {tot / (im2arr.size * 255)}")
        else:
            filtered_list.append(image_name)
    return filtered_list


def sliced_prediction(base_path: Path,
                      image_name: str,
                      detection_model,
                      slice_height = 1280,
                      slice_width = 1280):
    """
    wrapper for SAHI object detection

    :param image_path:
    :param yolov5_model_path:
    :return:
    """
    image_path = str(base_path.joinpath(image_name))

    logger.info(f"opening image: {image_path}")
    logger.warning(f"this should be refactored into the Candidate Proposal Class")

    result = get_sliced_prediction(
        read_image(image_path),
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,  # TODO remove this
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    logger.info(f"done with prediction")
    logger.info(result.object_prediction_list)
    logger.info(f"amount of objects found: {len(result.object_prediction_list)}")

    # result.to_fiftyone_detections()


    return result


def geospatial_prediction(base_path, image_name, model_path, image_size=None, logger=None,
                          FORCE_UPDATE=False,
                          slice_size=640):
    """
    TODO refactor this so the slicing and prediction are split
    TODO move this into the class
    prediction of objects returning its geospatial position

    :param base_path:
    :param image_name:
    :param model_path:
    :param logger:
    :return:
    """
    if logger == None:
        logger = loguru.logger
    predictions = {}
    logger.warning(f"this should be refactored into the Candidate Proposal Class")
    logger.info(f"start slicing the big raster")
    ## This is needed when the input orthomosaic is just too big

    sliced_image_path, image_names, slice_dict = slice_very_big_raster(base_path, image_name,
                                                                       x_size=image_size,
                                                                       y_size=image_size, FORCE_UPDATE=FORCE_UPDATE
                                                                       )
    for sliced_jpg_image_path, sliced_tif_image_path in slice_dict.items():
        convert_geotiff_to_jpg(FORCE_UPDATE=True,
                               sliced_jpg_image_path=sliced_jpg_image_path,
                               sliced_tif_image_path=sliced_tif_image_path
                                 )

    if True: ## FIXME
        logger.info(f"predict using this model: {model_path}")
        logger.info(f"get model *.pt from {model_path}")

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov5',
            model_path=str(model_path),
            confidence_threshold=0.5,
            device=str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        )

        logger.info(f"category names: {detection_model.category_names}")

        hA = HastyAnnotation(project_name="prediction_debug",
                             create_date="2022-03-20T18:05:30Z",
                             export_date="2022-04-20T13:52:33Z",
                             detection_model=detection_model)  ## TODO don't theses hardcoded values

        for image_name, tiff_name in slice_dict.items():
            logger.info(f"predicting on this image_name: {image_name}")

            predictions[str(image_name)] = sliced_prediction(
                base_path=sliced_image_path,
                image_name=image_name,
                detection_model=detection_model,
                slice_width=slice_size,
                slice_height=slice_size,
            )

            image_id = get_image_id(base_path.joinpath(image_name))

            hA.add_image(image_id=image_id,
                         width=predictions[str(image_name)].image.width,
                         height=predictions[str(image_name)].image.height,
                         image_name=str(image_name.name),
                         dataset_name="TODO_Mission_name",
                         )

            for r in predictions[str(image_name)].object_prediction_list:
                bbox = r.bbox.to_voc_bbox()
                label_id = "06c7492c-051c-4a45-8c4b-68ffb06a1935"  ## TODO fix this
                # label_id = str(uuid.uuid1())
                hA.add_label(image_id=image_id, id=str(label_id),
                             class_name=r.category.name, bbox=bbox)
        hA.persist(sliced_image_path)

        # result.to_fiftyone_detections()

        predictions_geojson_path = sahi_predictions_to_geo_json(sliced_image_path,
                                                                predictions,
                                                                detection_model.category_names)

        return predictions_geojson_path, image_names, sliced_image_path


def sahi_predictions_to_geo_json(sliced_image_path, predictions, detection_model_category_names):
    """

    @param sliced_image_path:
    @param predictions:
    @return:
    """
    import geopandas as gpd
    import json

    predictions_json = {key: obj_pred.to_coco_annotations() for key, obj_pred in predictions.items()}

    predictions_json_path = str(sliced_image_path.joinpath('predictions.json'))
    logger.info(f"dump predictions to: {predictions_json_path}")
    logger.warning(f"this should be refactored into the Candidate Proposal Class")

    with open(predictions_json_path, 'w+') as f:
        logger.info(f"found {len(predictions_json)} predictions.")
        logger.info(f"prediction json with this content: {predictions_json}")
        json.dump(predictions_json, f)

    with open(predictions_json_path, 'r') as f:
        predictions_json = json.load(f)

    logger.info(f"convert coco annotations: {predictions_json_path}")
    annotations_paths = coco_annotation_list_geopandas(predictions_json, sliced_image_path,
                                                       detection_model_category_names)

    gpdf_annotations = []
    for annotations_path in annotations_paths:
        logger.info(f"read annotation path from : {str(annotations_path)}")
        annotations_tmp_path = gpd.read_file(annotations_path)
        # logger.info(f"annotations_tmp_path : {str(annotations_tmp_path)}")
        logger.info(f"found {annotations_tmp_path.shape[0]} records")

        gpdf_annotations.append(annotations_tmp_path)

    predictions_geojson_path = str(sliced_image_path.joinpath("predictions.geojson"))
    logger.info(f"concatenating predictions into one geojson: {predictions_geojson_path}")
    try:
        gpdf_annotation = pd.concat(gpdf_annotations)
        gpdf_annotation.to_file(predictions_geojson_path, driver="GeoJSON")
    except ValueError as e:
        logger.warning(
            f"nothing could be predicted. If you tried finding iguans in poland this might be ok. Otherwise something could be wrong with the predictor.")

        with open(predictions_geojson_path, 'w') as f:
            json.dump([], f)

    return predictions_geojson_path

