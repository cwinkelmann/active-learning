"""
take detection and ground truth and look for the false positives and false negatives

This produces a csv file with the false positives because they are what we need to inspect

# TODO look for false negatives
Similar to Kellenberger - maybe this one Benjamin, et al. "Detecting and classifying elephants in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

"""
import sahi.prediction
from time import sleep

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from pyproj import CRS
from shapely.geometry import Point
from scipy.spatial import cKDTree

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage


def analyse_point_detections_(hA_gt: AnnotatedImage, predictions: sahi.prediction.PredictionResult):
    from shapely.geometry import box

    # TODO convert results
    df_hA_gt = pd.DataFrame([{"x": l.incenter_centroid.x, "y": l.incenter_centroid.y, "label": l.class_name} for l in hA_gt.labels])

    pred = []
    for p in predictions.object_prediction_list:
        maxx = p.bbox.maxx
        minx = p.bbox.minx
        maxy = p.bbox.maxy
        miny = p.bbox.miny
        bbox = box(minx, miny, maxx, maxy)
        score = p.score.value
        name = p.category.name

        pred.append({"x": bbox.centroid.x, "y": bbox.centroid.y, "label": name, "score": score})
    df_pred = pd.DataFrame(pred)

    return analyse_point_detections_df(df_pred, df_hA_gt)


def analyse_point_detections_df(df_detections: pd.DataFrame, df_ground_truth: pd.DataFrame):
    """
    filter for false positives and prepare data to ingest it into a review process
    :param df_detections:
    :param df_ground_truth:
    :return:
    """
    assert ["label", "score", "x", "y"] == sorted(df_detections.columns.tolist())
    assert ["label", "x", "y"] == sorted(df_ground_truth.columns.tolist())

    crs = CRS.from_proj4("+proj=cart +ellps=WGS84 +units=m +no_defs")

    # df_detections = pd.read_csv('/home/christian/hnee/HerdNet/data_iguana/val/20240824_HerdNet_results/20240824_detections.csv')
    # df_ground_truth = pd.read_csv('/home/christian/hnee/HerdNet/data_iguana/val/herdnet_format.csv')

    gdf_detections = gpd.GeoDataFrame(df_detections, geometry=gpd.points_from_xy(df_detections.x, df_detections.y))
    gdf_detections = gdf_detections.set_crs(crs)
    gdf_detections['buffer'] = gdf_detections.geometry.buffer(150)
    # df_detections_all = gdf_detections.set_geometry('buffer')
    gdf_detections_all = gdf_detections.copy()


    gdf_ground_truth = gpd.GeoDataFrame(df_ground_truth, geometry=gpd.points_from_xy(df_ground_truth.x, df_ground_truth.y))
    gdf_ground_truth_all = gdf_ground_truth.set_crs(crs)
    # gdf_ground_truth_all = gdf_ground_truth.set_geometry('geometry')

    # image_list = df_ground_truth['images'].unique()
    l_fp = []

    # for i in image_list:
    #     gdf_ground_truth = gdf_ground_truth_all[
    #         gdf_ground_truth_all['images'] == i]  # TODO proof of concept

    # gdf_detections = gdf_detections_all[gdf_detections_all['images'] == i]  # TODO proof of concept

    gt = gdf_ground_truth.copy()
    gt_coords = gt.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

    pred = gdf_detections.copy()
    pred_coords = pred.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

    gt_tree = cKDTree(gt_coords)

    # Query the nearest distance from each point in pred to any point in gt
    distances, indices = gt_tree.query(pred_coords, k=1)

    # True Positives: Detections that are within 150 units of any ground truth
    df_true_positives = gdf_detections[distances <= 150].copy()
    df_true_positives["kind"] = "true_positive"

    # Filter the points in pred where the nearest distance is greater than 150
    df_false_positves = pred[distances > 150][['x', 'y', 'label', 'score']]
    print(df_false_positves)
    l_fp.append(df_false_positves)
    df_false_positves["kind"] = "false_positive"

    # gdf_detections.drop(columns=['images', 'labels', 'species', 'count_1'], inplace=True)
    gdf_both = gpd.sjoin(gdf_ground_truth, gdf_detections, how='inner', predicate='intersects')
    gdf_both_outer = gpd.sjoin(gdf_ground_truth, gdf_detections, how='inner', predicate='intersects')

    # Reverse query: Check ground truth points that are too far from any prediction
    pred_tree = cKDTree(pred_coords)
    distances, indices = pred_tree.query(gt_coords, k=1)

    # Filter false negatives
    df_false_negatives = gdf_ground_truth[distances > 150][['x', 'y', 'label']]
    df_false_negatives['score'] = None  # No associated scores for false negatives
    print("False Negatives:")
    print(df_false_negatives)
    df_false_negatives["kind"] = "false_negative"

    #pd.concat(l_fp).to_csv('/home/christian/hnee/HerdNet/data_iguana/val/20240824_HerdNet_results/false_positives.csv', index=False)

    return pd.concat([df_true_positives, df_false_positves, df_false_negatives])