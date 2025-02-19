"""

"""
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import pytest

from active_learning.pipelines.herdnet_predict import HerdnetPredictor
from active_learning.util.converter import prediction_list_to_gdf, herdnet_prediction_to_hasty
from active_learning.util.visualisation.draw import draw_thumbnail
from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel, Keypoint

from loguru import logger
from shapely.geometry.point import Point

from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel
import geopandas as gpd

@pytest.fixture()
def orthomosaic_path():
    # TODO get the right local path
    return Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/images/orthomosaics/FMO02_full_orthophoto.tif")


@pytest.fixture()
def orthomosaic_tiles_path():
    # TODO get the right local path
    return Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/images/FMO02_full_orthophoto/tiles")


@pytest.fixture()
def herdnet_detection_path():
    herdnet_detection_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/FMO02_full_orthophoto_herdnet_detections.csv")

    return herdnet_detection_path

@pytest.fixture()
def prediction():
    herdnet_detection_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/herdnet_detections.csv")
    herdnet_detection_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/FMO02_full_orthophoto_herdnet_detections.csv")
    df_herdnet_detection = pd.read_csv(herdnet_detection_path)
    keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"

    fake_predictions = []

    for _, row in df_herdnet_detection.iterrows():
        if row["scores"] is None or np.isnan(row["scores"]):
            continue
        hkp = Keypoint(
            x=int(row["x"]),
            y=int(row["y"]),
            norder=0,
            keypoint_class_id=keypoint_class_id,
        )

        pIL = PredictedImageLabel(
            class_name=row["species"],
            score=row["scores"],
            keypoints=[hkp],
            attributes={
                "counts": [row["count_1"], row["count_2"], row["count_3"], row["count_4"], row["count_5"],
                           row["count_6"]]
            },
            kind="point", # TODO fix this right, because this "kind" should not be part of
        )

        fake_predictions.append(pIL)

    return fake_predictions

@patch.object(HerdnetPredictor, '_predict')
def test_herdnet_predict(mocked_predict, prediction, orthomosaic_path):
    mocked_predict.return_value = prediction
    output_geojson_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/output/herdnet_predictions.geojson")
    hP = HerdnetPredictor(model_path="model_path", confidence_threshold=0.3)

    hP.return_value = prediction

    predictions = hP.predict(orthomosaic_path, local=False)

    gdf = prediction_list_to_gdf(predictions)

    # Save as GeoJSON
    gdf.to_file(output_geojson_path, driver="GeoJSON")
    logger.info(f"Saved predictions as GeoJSON: {output_geojson_path}")


    assert len(prediction) == 226, "There should be three predictions"


def test_prediction_to_thumbnail(herdnet_detection_path: Path, orthomosaic_tiles_path: Path):

    df_herdnet_detection = pd.read_csv(herdnet_detection_path)
    hA_pred = herdnet_prediction_to_hasty(df_herdnet_detection, orthomosaic_tiles_path)

    for i in hA_pred:
        image_name = i.image_name
        predictions = i.labels

        # create boxes for each prediction for cropping, then we can apply the geospatial cutting


        draw_thumbnail(i, orthomosaic_tiles_path, box_size=350)

    raise NotImplementedError("This test is not implemented yet")