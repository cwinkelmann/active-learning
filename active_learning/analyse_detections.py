"""
take detection and ground truth and look for the false positives and false negatives

This produces a csv file with the false positives because they are what we need to inspect

 Similar to Kellenberger - maybe this one Benjamin, et al. "Detecting and classifying elephants in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

"""
import typing

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import Point

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
                                    image_list: typing.List[str],
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

    # image_list = df_ground_truth['images'].unique()

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

        if len(gt_coords) == 0 and len(pred_coords) == 0:
            image_errors.append(
                {"image_name": image, "err": 0, "num_gt": 0, "num_pred": 0})
            continue


        if len(gt_coords) == 0:
            gdf_pred['kind'] = 'false_positive'
            l_fp.append(gdf_pred)
            image_errors.append(
                {"image_name": image, "err": len(gdf_pred), "num_gt": 0, "num_pred": len(gdf_pred)})
            continue

        # If there are no predictions for the image, mark all as false negatives
        if len(pred_coords) == 0:
            gdf_gt['kind'] = 'false_negative'
            l_fn.append(gdf_gt)
            image_errors.append(
                {"image_name": image, "err": -1 * len(gdf_gt), "num_gt": len(gdf_gt), "num_pred": 0})
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

    assert len(image_errors) == len(image_list), f"Number of image errors {len(image_errors)} must equal number of images {len(image_list)}"

    df_image_errors = pd.DataFrame(image_errors)
    mean_error = df_image_errors.err.mean() if len(df_image_errors) > 0 else None
    logger.info(f"Mean Errors over all images : {mean_error}")

    # Concatenate the results from all images.
    df_false_positives = pd.concat(l_fp, ignore_index=True) if len(l_fp) > 0 else pd.DataFrame(columns=df_detections.columns)
    df_true_positives = pd.concat(l_tp, ignore_index=True) if len(l_tp) > 0 else pd.DataFrame(columns=df_detections.columns)
    df_false_negatives = pd.concat(l_fn, ignore_index=True) if len(l_fn) > 0 else pd.DataFrame(columns=df_ground_truth.columns)


    return df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth_all


def analyse_point_detections_geospatial(gdf_detections: gpd.GeoDataFrame,
                                        gdf_ground_truth: gpd.GeoDataFrame,
                                        radius_m=1,
                                        tile_name="tile_name",
                                        tile_name_prediction="tile_name_right",
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

    gdf_detections['buffer'] = gdf_detections.geometry.buffer(radius_m)
    gdf_detections_all = gdf_detections.copy()


    image_list = gdf_ground_truth[tile_name].unique()

    # Containers for results
    l_fp = []  # false positives
    l_tp = []  # true positives
    l_fn = []  # false negatives
    image_errors = []

    for image in image_list:
        # Filter for the current image

        gdf_gt = gdf_ground_truth[gdf_ground_truth[tile_name] == image].copy()
        gdf_pred = gdf_detections_all[gdf_detections_all[tile_name_prediction] == image].copy()

         # TODO put analyse_point_detections_geospatial_single_image in here


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
            if d <= radius_m and gt_idx not in matched_gt and pred_idx not in matched_pred:
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
    df_false_positives = pd.concat(l_fp, ignore_index=True) if len(l_fp) > 0 else pd.DataFrame(columns=gdf_detections.columns)
    df_true_positives = pd.concat(l_tp, ignore_index=True) if len(l_tp) > 0 else pd.DataFrame(columns=gdf_detections.columns)
    df_false_negatives = pd.concat(l_fn, ignore_index=True) if len(l_fn) > 0 else pd.DataFrame(columns=gdf_ground_truth.columns)


    return df_false_positives, df_true_positives, df_false_negatives


def analyse_point_detections_geospatial_single_image(gdf_detections: gpd.GeoDataFrame,
                                        gdf_ground_truth: gpd.GeoDataFrame,
                                        radius_m=1,
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

    gdf_detections['buffer'] = gdf_detections.geometry.buffer(radius_m)

    gdf_gt = gdf_ground_truth.copy()
    if len(gdf_gt) == 0:
        logger.warning(f"No ground truth points provided existing, which means all predictions are false positives.")
        df_false_negatives = gdf_gt
        df_false_positives = gdf_detections.copy()
        df_false_positives['kind'] = 'false_positive'
        df_true_positives = gpd.GeoDataFrame(columns=gdf_detections.columns)
        return df_false_positives, df_true_positives, df_false_negatives
    gdf_pred = gdf_detections.copy()

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
        df_false_negatives = gdf_gt
        df_false_positives = df_true_positives = gpd.GeoDataFrame(columns=gdf_detections.columns)
        return df_false_positives, df_true_positives, df_false_negatives

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
        if d <= radius_m and gt_idx not in matched_gt and pred_idx not in matched_pred:
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
    df_true_positives = df_tp_img
    df_false_positives = df_fp_img
    df_false_negatives = df_fn_img

    err = len(df_fp_img) + len(df_tp_img) - len(gdf_gt)

    return df_false_positives, df_true_positives, df_false_negatives


def analyse_multiple_user_point_detections_geospatial(dict_gdf_detections: typing.Dict[str, gpd.GeoDataFrame],
                                                      radius_m=1, N_agree=2) -> typing.Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Find marks on a map which more than N people agree on.

    Parameters:
      dict_gdf_detections: Dict[str, gpd.GeoDataFrame]
         Dictionary which contains user_name and the geodataframe of their predictions.
      radius_m: float, optional (default=1)
         Maximum allowed Euclidean distance to consider points as agreeing.
      N_agree: int, optional (default=2)
         Minimum number of users that must agree for a point to be considered agreement.

    Returns:
      A tuple of two GeoDataFrames:
        (gdf_agreement, gdf_disagreement, gpd.GeoDataFrame)
    """

    if not dict_gdf_detections:
        raise ValueError("dict_gdf_detections cannot be empty")

    if N_agree < 2:
        raise ValueError("N_agree must be at least 2")

    # Combine all user detections into a single GeoDataFrame
    combined_detections = []
    for user_name, gdf in dict_gdf_detections.items():
        if len(gdf) > 0:
            gdf_copy = gdf.copy()
            gdf_copy['user'] = user_name
            gdf_copy['original_index'] = gdf_copy.index
            combined_detections.append(gdf_copy)

    if not combined_detections:
        # Return empty GeoDataFrames with expected structure
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'user', 'cluster_id', 'users_agreeing', 'centroid_geometry'])
        return empty_gdf, empty_gdf

    # Combine all detections
    all_detections = pd.concat(combined_detections, ignore_index=True)
    all_detections = gpd.GeoDataFrame(all_detections)

    # Extract coordinates for spatial analysis
    coords = [(geom.x, geom.y) for geom in all_detections.geometry]

    if len(coords) == 0:
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'user', 'cluster_id', 'users_agreeing', 'centroid_geometry'])
        return empty_gdf, empty_gdf

    # Build KDTree for efficient spatial queries
    tree = cKDTree(coords)

    # Find clusters of nearby points
    clusters = []
    visited = set()

    for i, coord in enumerate(coords):
        if i in visited:
            continue

        # Find all points within radius
        nearby_indices = tree.query_ball_point(coord, radius_m)

        # Get unique users in this cluster
        cluster_users = set()
        cluster_points = []

        for idx in nearby_indices:
            if idx not in visited:
                user = all_detections.iloc[idx]['user']
                cluster_users.add(user)
                cluster_points.append(idx)
                visited.add(idx)

        if len(cluster_points) > 0:
            clusters.append({
                'point_indices': cluster_points,
                'users': cluster_users,
                'user_count': len(cluster_users)
            })

    # Process clusters to create agreement and disagreement datasets
    agreement_data = []
    disagreement_data = []
    agreement_locations_data = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_points = [all_detections.iloc[i] for i in cluster['point_indices']]

        if cluster['user_count'] >= N_agree:
            # Agreement: calculate centroid of all points in cluster
            cluster_coords = [(pt.geometry.x, pt.geometry.y) for pt in cluster_points]
            centroid_x = np.mean([c[0] for c in cluster_coords])
            centroid_y = np.mean([c[1] for c in cluster_coords])
            centroid_geom = Point(centroid_x, centroid_y)

            # Add all points from this cluster to agreement data
            for pt in cluster_points:
                agreement_data.append({
                    **pt.to_dict(),
                    'cluster_id': cluster_id,
                    'users_agreeing': list(cluster['users']),
                    'user_count': cluster['user_count'],
                    'centroid_geometry': centroid_geom
                })
            # Create agreement location entry (one per cluster)
            agreement_locations_data.append({
                'geometry': centroid_geom,
                'cluster_id': cluster_id,
                'users_agreeing': list(cluster['users']),
                'user_count': cluster['user_count'],
                'contributing_points': len(cluster['point_indices'])
            })
        else:
            # Disagreement: not enough users agree
            for pt in cluster_points:
                disagreement_data.append({
                    **pt.to_dict(),
                    'cluster_id': cluster_id,
                    'users_agreeing': list(cluster['users']),
                    'user_count': cluster['user_count'],
                    'centroid_geometry': None
                })

    # Create result GeoDataFrames
    if agreement_data:
        gdf_agreement = gpd.GeoDataFrame(agreement_data)
        gdf_agreement = gdf_agreement.set_crs(all_detections.crs)
    else:
        gdf_agreement = gpd.GeoDataFrame(columns=list(all_detections.columns) +
                                                 ['cluster_id', 'users_agreeing', 'user_count', 'centroid_geometry'])
        if hasattr(all_detections, 'crs'):
            gdf_agreement = gdf_agreement.set_crs(all_detections.crs)

    if disagreement_data:
        gdf_disagreement = gpd.GeoDataFrame(disagreement_data)
        gdf_disagreement = gdf_disagreement.set_crs(all_detections.crs)
    else:
        gdf_disagreement = gpd.GeoDataFrame(columns=list(all_detections.columns) +
                                                    ['cluster_id', 'users_agreeing', 'user_count', 'centroid_geometry'])
        if hasattr(all_detections, 'crs'):
            gdf_disagreement = gdf_disagreement.set_crs(all_detections.crs)

    # Create agreement locations GeoDataFrame (one row per agreement cluster)
    if agreement_locations_data:
        gdf_agreement_locations = gpd.GeoDataFrame(agreement_locations_data)
        gdf_agreement_locations = gdf_agreement_locations.set_crs(all_detections.crs)
    else:
        gdf_agreement_locations = gpd.GeoDataFrame(
            columns=['geometry', 'cluster_id', 'users_agreeing', 'user_count', 'contributing_points'])
        if hasattr(all_detections, 'crs'):
            gdf_agreement_locations = gdf_agreement_locations.set_crs(all_detections.crs)

    return gdf_agreement, gdf_disagreement, gdf_agreement_locations


def analyse_multiple_user_point_detections_geospatial_v2(dict_gdf_detections: typing.Dict[str, gpd.GeoDataFrame],
                                                      radius_m=1, N_agree=2) -> (gpd.GeoDataFrame, gpd.GeoDataFrame,
                                                                                 gpd.GeoDataFrame):
    """
    Find marks on a map which more than N people agree on.

    Parameters:
      dict_gdf_detections: Dict[str, gpd.GeoDataFrame]
         Dictionary which contains user_name and the geodataframe of their predictions.
      radius_m: float, optional (default=1)
         Maximum allowed Euclidean distance to consider points as agreeing.
      N_agree: int, optional (default=2)
         Minimum number of users that must agree for a point to be considered agreement.

    Returns:
      A tuple of three GeoDataFrames:
        (gdf_agreement, gdf_disagreement, gdf_agreement_locations)
        - gdf_agreement: Individual points that belong to agreeing clusters
        - gdf_disagreement: Individual points that don't meet agreement threshold
        - gdf_agreement_locations: One row per agreement location with centroid geometry and user count
    """

    if not dict_gdf_detections:
        raise ValueError("dict_gdf_detections cannot be empty")

    if N_agree < 1:
        raise ValueError("N_agree must be at least 1")

    # Combine all user detections into a single GeoDataFrame
    combined_detections = []
    for user_name, gdf in dict_gdf_detections.items():
        if len(gdf) > 0:
            gdf_copy = gdf.copy()
            gdf_copy['user'] = user_name
            gdf_copy['original_index'] = gdf_copy.index
            combined_detections.append(gdf_copy)

    if not combined_detections:
        # Return empty GeoDataFrames with expected structure
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'user', 'cluster_id', 'users_agreeing', 'centroid_geometry'])
        empty_locations_gdf = gpd.GeoDataFrame(
            columns=['geometry', 'cluster_id', 'users_agreeing', 'user_count', 'contributing_users'])
        return empty_gdf, empty_gdf, empty_locations_gdf

    # Combine all detections
    all_detections = pd.concat(combined_detections, ignore_index=True)
    all_detections = gpd.GeoDataFrame(all_detections)

    # remove None geometries
    if all_detections.geometry.isnull().any():
        logger.warning(f"There are {all_detections.geometry.isnull().sum()} detections with null geometry, these will be ignored but you should check why they are there, because that is actually impossible.")
    all_detections = all_detections[all_detections.geometry.notnull()].reset_index(drop=True)

    # Extract coordinates for spatial analysis
    coords = [(geom.x, geom.y) for geom in all_detections.geometry]
    if len(coords) == 0:
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'user', 'cluster_id', 'users_agreeing', 'centroid_geometry'])
        empty_locations_gdf = gpd.GeoDataFrame(
            columns=['geometry', 'cluster_id', 'users_agreeing', 'user_count', 'contributing_users'])
        return empty_gdf, empty_gdf, empty_locations_gdf

    # Build KDTree for efficient spatial queries
    tree = cKDTree(coords)

    # Find clusters of nearby points
    clusters = []
    visited = set()

    for i, coord in enumerate(coords):
        if i in visited:
            continue

        # Find all points within radius
        nearby_indices = tree.query_ball_point(coord, radius_m)

        # Mark all points in this cluster as visited
        cluster_points = []
        for idx in nearby_indices:
            if idx not in visited:
                cluster_points.append(idx)
                visited.add(idx)

        if len(cluster_points) > 0:
            clusters.append({
                'point_indices': cluster_points,
            })

    # Filter each cluster to ensure one point per user per cluster
    filtered_clusters = []
    for cluster in clusters:
        user_to_best_point = {}

        # For each user in this cluster, keep only one point (first encountered)
        # Could be enhanced to keep "best" point based on confidence score
        for idx in cluster['point_indices']:
            user = all_detections.iloc[idx]['user']
            if user not in user_to_best_point:
                user_to_best_point[user] = idx

        # Create filtered cluster
        filtered_points = list(user_to_best_point.values())
        filtered_users = set(user_to_best_point.keys())

        if len(filtered_points) > 0:
            filtered_clusters.append({
                'point_indices': filtered_points,
                'users': filtered_users,
                'user_count': len(filtered_users)
            })

    clusters = filtered_clusters

    # Process clusters to create agreement and disagreement datasets
    agreement_data = []
    disagreement_data = []
    agreement_locations_data = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_points = [all_detections.iloc[i] for i in cluster['point_indices']]

        if cluster['user_count'] >= N_agree:
            # Agreement: calculate centroid of all points in cluster
            cluster_coords = [(pt.geometry.x, pt.geometry.y) for pt in cluster_points]
            centroid_x = np.mean([c[0] for c in cluster_coords])
            centroid_y = np.mean([c[1] for c in cluster_coords])
            centroid_geom = Point(centroid_x, centroid_y)

            # Add all points from this cluster to agreement data
            for pt in cluster_points:
                agreement_data.append({
                    **pt.to_dict(),
                    'cluster_id': cluster_id,
                    'users_agreeing': list(cluster['users']),
                    'user_count': cluster['user_count'],
                    'centroid_geometry': centroid_geom
                })

            # Create agreement location entry (one per cluster)
            agreement_locations_data.append({
                'geometry': centroid_geom,
                'cluster_id': cluster_id,
                'users_agreeing': list(cluster['users']),
                'user_count': cluster['user_count'],
                'contributing_users': cluster['user_count']  # Number of users contributing to this location
            })

        else:
            # Disagreement: not enough users agree
            for pt in cluster_points:
                disagreement_data.append({
                    **pt.to_dict(),
                    'cluster_id': cluster_id,
                    'users_agreeing': list(cluster['users']),
                    'user_count': cluster['user_count'],
                    'centroid_geometry': None
                })

    # Create result GeoDataFrames
    if agreement_data:
        gdf_agreement = gpd.GeoDataFrame(agreement_data)
        gdf_agreement = gdf_agreement.set_crs(all_detections.crs)
    else:
        gdf_agreement = gpd.GeoDataFrame(columns=list(all_detections.columns) +
                                                 ['cluster_id', 'users_agreeing', 'user_count', 'centroid_geometry'])
        if hasattr(all_detections, 'crs'):
            gdf_agreement = gdf_agreement.set_crs(all_detections.crs)

    if disagreement_data:
        gdf_disagreement = gpd.GeoDataFrame(disagreement_data)
        gdf_disagreement = gdf_disagreement.set_crs(all_detections.crs)
    else:
        gdf_disagreement = gpd.GeoDataFrame(columns=list(all_detections.columns) +
                                                    ['cluster_id', 'users_agreeing', 'user_count', 'centroid_geometry'])
        if hasattr(all_detections, 'crs'):
            gdf_disagreement = gdf_disagreement.set_crs(all_detections.crs)

    # Create agreement locations GeoDataFrame (one row per agreement cluster)
    if agreement_locations_data:
        gdf_agreement_locations = gpd.GeoDataFrame(agreement_locations_data)
        gdf_agreement_locations = gdf_agreement_locations.set_crs(all_detections.crs)
    else:
        gdf_agreement_locations = gpd.GeoDataFrame(
            columns=['geometry', 'cluster_id', 'users_agreeing', 'user_count', 'contributing_users'])
        if hasattr(all_detections, 'crs'):
            gdf_agreement_locations = gdf_agreement_locations.set_crs(all_detections.crs)

    return gdf_agreement, gdf_disagreement, gdf_agreement_locations





def get_agreement_summary(gdf_agreement: gpd.GeoDataFrame,
                          gdf_disagreement: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generate a summary of agreement statistics.

    Returns:
        pd.DataFrame with summary statistics
    """

    summary_stats = {
        'total_agreement_points': len(gdf_agreement),
        'total_disagreement_points': len(gdf_disagreement),
        'total_points': len(gdf_agreement) + len(gdf_disagreement),
    }

    if len(gdf_agreement) > 0:
        # Get unique clusters (agreement locations)
        unique_clusters = gdf_agreement['cluster_id'].nunique()
        summary_stats['unique_agreement_locations'] = unique_clusters

        # User agreement statistics
        if 'user_count' in gdf_agreement.columns:
            summary_stats['avg_users_per_agreement'] = gdf_agreement.groupby('cluster_id')['user_count'].first().mean()
            summary_stats['max_users_per_agreement'] = gdf_agreement.groupby('cluster_id')['user_count'].first().max()
            summary_stats['min_users_per_agreement'] = gdf_agreement.groupby('cluster_id')['user_count'].first().min()
    else:
        summary_stats['unique_agreement_locations'] = 0
        summary_stats['avg_users_per_agreement'] = 0
        summary_stats['max_users_per_agreement'] = 0
        summary_stats['min_users_per_agreement'] = 0

    if len(gdf_agreement) + len(gdf_disagreement) > 0:
        summary_stats['agreement_rate'] = len(gdf_agreement) / (len(gdf_agreement) + len(gdf_disagreement))
    else:
        summary_stats['agreement_rate'] = 0

    return summary_stats




###########





if __name__ == "__main__":
    pass
