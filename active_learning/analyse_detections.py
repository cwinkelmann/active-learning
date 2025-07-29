"""
take detection and ground truth and look for the false positives and false negatives

This produces a csv file with the false positives because they are what we need to inspect

 Similar to Kellenberger - maybe this one Benjamin, et al. "Detecting and classifying elephants in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

"""
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from scipy.spatial import cKDTree

from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage


def analyse_point_detections(df_detections: pd.DataFrame, df_ground_truth: pd.DataFrame, radius=150):
    """
    @deprectaed
    Analyse detection and look into false positives, false negatives
    :param df_detections:
    :param df_ground_truth:
    :return:
    """
    raise DeprecationWarning("This function is deprecated, use analyse_point_detections_greedy instead")
    crs = CRS.from_proj4("+proj=cart +ellps=WGS84 +units=m +no_defs")

    gdf_detections = gpd.GeoDataFrame(df_detections,
                                      geometry=gpd.points_from_xy(df_detections.x, df_detections.y)).set_crs(crs)
    gdf_detections['buffer'] = gdf_detections.geometry.buffer(radius)
    # df_detections_all = gdf_detections.set_geometry('buffer')
    gdf_detections_all = gdf_detections.copy()

    gdf_ground_truth_all = gpd.GeoDataFrame(df_ground_truth,
                                            geometry=gpd.points_from_xy(df_ground_truth.x, df_ground_truth.y)).set_crs(
        crs)

    image_list = df_ground_truth['images'].unique()
    l_fp = []
    l_tp = []
    l_fn = []

    for i in image_list:
        gdf_ground_truth = gdf_ground_truth_all[gdf_ground_truth_all['images'] == i]
        gdf_detections = gdf_detections_all[gdf_detections_all['images'] == i]

        gt = gdf_ground_truth.copy()
        gt_coords = gt.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

        pred = gdf_detections.copy()
        pred_coords = pred.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

        gt_tree = cKDTree(gt_coords)

        # Query the nearest distance from each point in pred to any point in gt
        distances, indices = gt_tree.query(pred_coords, k=1)

        # Filter the points in pred where the nearest distance is greater than 150
        false_positives_mask = distances > radius

        df_false_positives = pred[false_positives_mask].copy()
        df_true_positives = pred[~false_positives_mask].copy()

        # Get unique indices of GT points that were matched (true positive matches).
        true_positive_indices = np.unique(indices[~false_positives_mask])
        df_false_negatives = gt.loc[~gt.index.isin(true_positive_indices)].copy()

        assert len(df_false_positives) + len(df_true_positives) == len(
            pred), "The sum of false positives and true positives must equal the number of predictions"
        assert len(df_false_negatives) + len(df_true_positives) == len(
            gt), "The sum of false negatives and true positives must equal the number of ground truth"
        l_fp.append(df_false_positives)
        l_tp.append(df_true_positives)
        l_fn.append(df_false_negatives)

    df_false_positives = pd.concat(l_fp)
    df_false_positives['kind'] = 'false_positive'
    df_true_positives = pd.concat(l_tp)
    df_true_positives['kind'] = 'true_positive'

    df_false_negatives = pd.concat(l_fn)
    df_false_negatives['kind'] = 'false_negative'

    return df_false_positives, df_true_positives, df_false_negatives


def analyse_point_detections_greedy(df_detections: pd.DataFrame,
                                    df_ground_truth: pd.DataFrame,
                                    radius=150,
                                    confidence_threshold=0.5) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Analyse detections and look into false positives, false negatives
    using one-to-one matching on a per-image basis.

    Each prediction can be assigned to at most one ground truth (if within the threshold)
    and vice versa. Unmatched predictions are false positives;
    unmatched ground truths are false negatives.

    Parameters:
      df_detections: pd.DataFrame
         Must include at least the columns 'x', 'y', and 'images'.
      df_ground_truth: pd.DataFrame
         Must include at least the columns 'x', 'y', and 'images'.
      radius: float, optional (default=150)
         Maximum allowed Euclidean distance to consider a prediction as matching a ground truth.

    Returns:
      A tuple of three DataFrames:
        (df_false_positives, df_true_positives, df_false_negatives)
    """
    # Define coordinate system
    crs = CRS.from_proj4("+proj=cart +ellps=WGS84 +units=m +no_defs")

    # Convert detections to GeoDataFrame and create a buffer (if you need it later)
    gdf_detections = gpd.GeoDataFrame(
        df_detections,
        geometry=gpd.points_from_xy(df_detections.x, df_detections.y)
    ).set_crs(crs)
    gdf_detections['buffer'] = gdf_detections.geometry.buffer(radius)
    gdf_detections_all = gdf_detections.copy()

    # Convert ground truth to GeoDataFrame
    gdf_ground_truth_all = gpd.GeoDataFrame(
        df_ground_truth,
        geometry=gpd.points_from_xy(df_ground_truth.x, df_ground_truth.y)
    ).set_crs(crs)

    image_list = df_ground_truth['images'].unique()

    # Containers for results
    l_fp = []  # false positives
    l_tp = []  # true positives
    l_fn = []  # false negatives
    image_errors = []

    for image in image_list:
        # Filter for the current image


        gdf_gt = gdf_ground_truth_all[gdf_ground_truth_all['images'] == image].copy()
        gdf_pred = gdf_detections_all[gdf_detections_all['images'] == image].copy()

        if len(gdf_pred) == 1:
            len_before = len(gdf_pred)
            gdf_pred = gdf_pred[~(gdf_pred['labels'].isna() & gdf_pred['scores'].isna())]
            if len(gdf_pred) < len_before:
                # logger.warning(f"Removed {len_before - len(gdf_pred)} predictions with NaN labels and scores for image {image}")
                pass
        # Convert geometries to tuples (here we use int() conversion; adjust if necessary)
        gt_coords = gdf_gt.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

        pred_coords = gdf_pred.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

        # If there are no predictions for the image, mark all ground truth as false negatives
        if len(pred_coords) == 0:
            gdf_gt['kind'] = 'false_negative'
            l_fn.append(gdf_gt)
            continue

        # Build a KD-tree on the prediction points
        pred_tree = cKDTree(pred_coords)
        # For each ground truth point, get the nearest prediction (distance and prediction index)
        distances, indices = pred_tree.query(gt_coords, k=1)

        # Build candidate matches: for each gt point, record (gt_index, distance, pred_index)
        candidate_matches = [(i_gt, d, i_pred)
                             for i_gt, (d, i_pred) in enumerate(zip(distances, indices))]

        # Sort candidate matches by distance (ascending)
        candidate_matches.sort(key=lambda t: t[1])

        # Greedily assign matches ensuring one-to-one matching
        matched_gt = set()
        matched_pred = set()
        filtered_matches = []
        for gt_idx, d, pred_idx in candidate_matches:
            if d <= radius and gt_idx not in matched_gt and pred_idx not in matched_pred:
                filtered_matches.append((gt_idx, d, pred_idx))
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)

        # For this image:
        # True positives: predictions that were matched.
        df_tp_img = gdf_pred.iloc[list(matched_pred)].copy()
        df_tp_img['kind'] = 'true_positive'

        # False positives: predictions that were not matched.
        all_pred_indices = set(range(len(gdf_pred)))
        false_positive_indices = all_pred_indices - matched_pred
        df_fp_img = gdf_pred.iloc[list(false_positive_indices)].copy()
        df_fp_img['kind'] = 'false_positive'

        # False negatives: ground truth points that were not matched.
        all_gt_indices = set(range(len(gdf_gt)))
        false_negative_indices = all_gt_indices - matched_gt
        df_fn_img = gdf_gt.iloc[list(false_negative_indices)].copy()
        df_fn_img['kind'] = 'false_negative'
        df_fn_img['scores'] = 0.0  # TODO make sure to get the scores if an prediction was below the confidence threshold

        assert len(df_tp_img) + len(df_fp_img) == len(
            gdf_pred), "The sum of false positives and true positives must equal the number of predictions"
        assert len(df_fn_img) + len(df_tp_img) == len(
            gdf_gt), "The sum of false negatives and true positives must equal the number of ground truth"

        # Accumulate the results for this image.
        l_tp.append(df_tp_img)
        l_fp.append(df_fp_img)
        l_fn.append(df_fn_img)

        err = len(df_fp_img) + len(df_tp_img) - len(gdf_gt)
        image_errors.append({"image_name": image, "err": err, "num_gt": len(gdf_gt), "num_pred": len(df_fp_img) + len(df_tp_img)})


    logger.info(f"Aggregating results over {len(image_list)} images.")
    # Calculate mean error over all images.
    df_image_errors = pd.DataFrame(image_errors)
    mean_error = df_image_errors.err.mean() if len(df_image_errors) > 0 else None
    logger.info(f"Mean Errors over all images : {mean_error}")

    # Concatenate the results from all images.
    df_false_positives = pd.concat(l_fp, ignore_index=True) if len(l_fp) > 0 else pd.DataFrame(columns=df_detections.columns)
    df_true_positives = pd.concat(l_tp, ignore_index=True) if len(l_tp) > 0 else pd.DataFrame(columns=df_detections.columns)
    df_false_negatives = pd.concat(l_fn, ignore_index=True) if len(l_fn) > 0 else pd.DataFrame(columns=df_ground_truth.columns)


    return df_false_positives, df_true_positives, df_false_negatives


def analyse_point_detection_correction(predicted: AnnotatedImage,
                                       corrected: AnnotatedImage,
                                       radius=150):
    """
    Analyse the correction of a point detector. This function compares the predicted detections with the corrected
    """

    raise NotImplementedError("This function is not implemented yet")


if __name__ == "__main__":
    pass
