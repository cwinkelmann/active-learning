import shapely

from active_learning.analyse_detections import analyse_point_detections_geospatial_single_image_hungarian, \
    analyse_point_detections_geospatial_hungarian
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

@pytest.fixture
def gdf_detections():
    # Predictions
    pred_data = {
        'id': [1, 2, 3],
        'geometry': [
            Point(100.2, 100.3),  # Close to GT 1 - TRUE POSITIVE
            Point(200.1, 200.2),  # Close to GT 2 - TRUE POSITIVE
            Point(400.0, 400.0),  # No GT nearby - FALSE POSITIVE
        ],
        'labels': [1.0, 1.0, 1.0],
        'scores': [0.95, 0.92, 0.88]
    }
    gdf_pred = gpd.GeoDataFrame(pred_data, crs='EPSG:32717')

    return gdf_pred


@pytest.fixture
def gdf_ground_truth():
    """
    Simple test case: 3 GT points, 3 predictions
    - 2 perfect matches (within radius)
    - 1 false positive (no GT nearby)
    - 1 false negative (no prediction nearby)
    """
    # Ground truth points
    gt_data = {
        'id': [4, 5, 6],
        'geometry': [
            Point(100.0, 100.0),  # Will match with pred 1
            Point(200.0, 200.0),  # Will match with pred 2
            Point(300.0, 300.0),  # No match - FALSE NEGATIVE
        ],
        'labels': [1.0, 1.0, 1.0]
    }
    gdf_gt = gpd.GeoDataFrame(gt_data, crs='EPSG:32717')

    return gdf_gt


@pytest.fixture
def gdf_detections_multiple():
    # Predictions
    pred_data = {
        'id': [1, 2, 3],
        'img_name': ["a", "b", "c"],
        'geometry': [
            Point(100.2, 100.3),  # Close to GT 1 - TRUE POSITIVE
            Point(200.1, 200.2),  # Close to GT 2 - TRUE POSITIVE
            Point(400.0, 400.0),  # No GT nearby - FALSE POSITIVE
        ],
        'labels': [1.0, 1.0, 1.0],
        'scores': [0.95, 0.92, 0.88]
    }
    gdf_pred = gpd.GeoDataFrame(pred_data, crs='EPSG:32717')

    return gdf_pred


@pytest.fixture
def gdf_ground_truth_multiple():
    """
    Simple test case: 3 GT points, 3 predictions
    - 2 perfect matches (within radius)
    - 1 false positive (no GT nearby)
    - 1 false negative (no prediction nearby)
    """
    # Ground truth points
    gt_data = {
        'id': [4, 5, 6],
        'img_name': ["a", "b", "c"],
        'geometry': [
            Point(100.0, 100.0),  # Will match with pred 1
            Point(200.0, 200.0),  # Will match with pred 2
            Point(300.0, 300.0),  # No match - FALSE NEGATIVE
        ],
        'labels': [1.0, 1.0, 1.0]
    }
    gdf_gt = gpd.GeoDataFrame(gt_data, crs='EPSG:32717')

    return gdf_gt


def test_analyse_point_detections_geospatial_single_image_hungarian(gdf_detections, gdf_ground_truth):


    gdf_false_positives, gdf_true_positives, gdf_false_negatives = analyse_point_detections_geospatial_single_image_hungarian(
        gdf_detections,
        gdf_ground_truth,
        radius_m = 1,
        confidence_threshold = 0.5)

    fp_id = gdf_false_positives.id.values.tolist()
    assert fp_id == [3]

    fp_tp = gdf_true_positives.id.values.tolist()
    assert fp_tp == [1,2]

    fp_tn = gdf_false_negatives.id.values.tolist()
    assert fp_tn == [6]


def test_analyse_point_detections_geospatial_hungarian(gdf_detections_multiple, gdf_ground_truth_multiple):


    gdf_false_positives, gdf_true_positives, gdf_false_negatives = analyse_point_detections_geospatial_hungarian(
        gdf_detections_multiple,
        gdf_ground_truth_multiple,
        radius_m = 5,
        confidence_threshold = 0.5)

    fp_id = gdf_false_positives.id.values.tolist()
    assert fp_id == [3]

    fp_tp = gdf_true_positives.id.values.tolist()
    assert fp_tp == [1,2]

    fp_tn = gdf_false_negatives.id.values.tolist()
    assert fp_tn == [6]