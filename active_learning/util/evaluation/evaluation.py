"""
Take Detections from a model and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
import numpy as np
import typing

import PIL.Image
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_greedy
# import pytest

from active_learning.util.converter import herdnet_prediction_to_hasty, _create_keypoints_s, _create_boxes_s, \
    _create_fake_boxes
from active_learning.util.image_manipulation import crop_out_images_v3
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, ImageLabelCollection, AnnotatedImage, \
    HastyAnnotationV2


def evaluate_predictions(
        # images_set: typing.List[Path],
        sample: fo.Sample,
        # images_dir: Path,
        predictions: typing.List[ImageLabelCollection] = None,
        type: AnnotationType = AnnotationType.KEYPOINT,
        sample_field_name="prediction") -> fo.Sample:
    """
    Evaluate the predictions
    :return:
    """
    image_name = sample.filename

    # for image_path in images_set:
    filtered_predictions = [p for p in predictions if p.image_name == image_name]
    if len(filtered_predictions) == 1:

        hA_image_pred = filtered_predictions[0]

        if type == AnnotationType.KEYPOINT:
            # keypoints_pred = _create_fake_boxes(hA_image=hA_image_pred) # TODO this was a bit of hack to visualise better in fiftyone
            keypoints_pred = _create_keypoints_s(hA_image=hA_image_pred)
        elif type == AnnotationType.BOUNDING_BOX:
            boxes_pred = _create_boxes_s(hA_image=hA_image_pred)
        elif type == AnnotationType.POLYGON:
            raise NotImplementedError("Polygons are not yet implemented")
            #polygons = _create_polygons_s(hA_image=hA_image)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")
        # FIXME, if this function is called multiple times, the same image will be added multiple times


        if type == AnnotationType.KEYPOINT:
            # sample[sample_field_name] = fo.Detections(detections=keypoints_pred)
            sample[sample_field_name] = fo.Keypoints(keypoints=keypoints_pred)
        elif type == AnnotationType.BOUNDING_BOX:
            sample[sample_field_name] = fo.Detections(detections=boxes_pred)
        elif type == AnnotationType.POLYGON:
            raise NotImplementedError("Polygons are not yet implemented")
            # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")

        if hasattr(hA_image_pred, "tags"):
            sample.tags = hA_image_pred.tags
        sample["hasty_image_id"] = hA_image_pred.image_id
        sample["hasty_image_name"] = hA_image_pred.image_name

        # logger.info(f"Added {image_name} to the dataset")

        sample.save()
    else:
        logger.error(f"There should be one single image left, but {len(filtered_predictions)} are left.")

    return sample


def evaluate_ground_truth(
        sample: fo.Sample,
        ground_truth_labels: typing.List[AnnotatedImage] = None,
        type: AnnotationType = AnnotationType.KEYPOINT, sample_field_name="ground_truth", ):
    """
    add ground truth to fiftyOne sample    :return:
    """
    # dataset = fo.load_dataset(dataset_name) # loading a bad idea because the single source of truth is the hasty annotations
    image_name = sample.filename

    hA_gt_sample = [i for i in ground_truth_labels if i.image_name == sample.filename]
    assert len(hA_gt_sample) == 1, f"There should be one single image left, but {len(hA_gt_sample)} are left."

    hA_image = hA_gt_sample[0]

    if type == AnnotationType.KEYPOINT:
        keypoints = _create_keypoints_s(hA_image=hA_image)
    elif type == AnnotationType.BOUNDING_BOX:
        boxes = _create_boxes_s(hA_image=hA_image)
    elif type == AnnotationType.POLYGON:
        raise NotImplementedError("Polygons are not yet implemented")
        #polygons = _create_polygons_s(hA_image=hA_image)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    sample.tags=hA_image.tags
    sample["hasty_image_id"] = hA_image.image_id
    sample["hasty_image_name"] = hA_image.image_name

    if type == AnnotationType.KEYPOINT:
        sample["ground_truth"] = fo.Keypoints(keypoints=keypoints)
    elif type == AnnotationType.BOUNDING_BOX:
        sample["ground_truth"] = fo.Detections(detections=boxes)
    elif type == AnnotationType.POLYGON:
        raise NotImplementedError("Polygons are not yet implemented")
        # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    # logger.info(f"Added {image_name} to the dataset")

    sample.save()

    return sample


def evaluate_in_fifty_one(dataset_name: str, images_set: typing.List[Path],
                          hA_ground_truth: HastyAnnotationV2,
                          IL_fp_detections: typing.List[ImageLabelCollection],
                          IL_fn_detections: typing.List[ImageLabelCollection],
                          IL_tp_detections: typing.List[ImageLabelCollection],
                          type: AnnotationType =AnnotationType.KEYPOINT):
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True
        # fo.list_datasets()

    dataset = fo.Dataset.from_images([str(i) for i in images_set])
    dataset.persistent = True

    for sample in dataset:
        # create dot annotations
        sample = evaluate_ground_truth(
            ground_truth_labels=hA_ground_truth.images,
            sample=sample,
            sample_field_name="ground_truth_points",
            # images_set=images_set,
            type=type,
        )

        sample = evaluate_predictions(
            predictions=IL_fp_detections,
            sample=sample,
            sample_field_name="false_positives",
            # images_set=images_set,
            type=type,
        )
        sample = evaluate_predictions(
            predictions=IL_fn_detections,
            sample=sample,
            sample_field_name="false_negatives",
            type=type,
        )
        sample = evaluate_predictions(
            predictions=IL_tp_detections,
            sample=sample,
            sample_field_name="true_positives",
            type=type,
        )

        dataset.add_sample(sample)


    # ## TODO fix the dataset fields
    # evaluation_results = dataset.evaluate_detections(
    #     pred_field="true_positives",
    #     gt_field="ground_truth_points",
    #     eval_key="point_eval",
    #     eval_type="keypoint",
    #     distance=10.0,  # Adjust based on your specific requirements
    #     classes=None  # Specify classes if applicable
    # )
    # precision = evaluation_results.metrics()["precision"]
    # recall = evaluation_results.metrics()["recall"]
    # f1_score = evaluation_results.metrics()["f1"]
    #
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"F1 Score: {f1_score:.2f}")
    #
    # evaluation_results.print_report()

    session = fo.launch_app(dataset)
    session.wait()


def submit_for_cvat_evaluation(dataset: fo.Dataset,
                               # images_set: typing.List[Path],
                          detections: typing.List[ImageLabelCollection],
                          type=AnnotationType.KEYPOINT):
    """
    @Deprecated
    :param dataset_name:
    :param images_set:
    :param detections:
    :param type:
    :return:
    """


    for sample in dataset:
        # create dot annotations
        sample = evaluate_predictions(
            predictions=detections,
            sample=sample,
            sample_field_name="detection",
            # images_set=images_set,
            type=type,
        )
        sample.save()

    return dataset



def submit_for_roboflow_evaluation(dataset_name: str, images_set: typing.List[Path],
                          detections: typing.List[ImageLabelCollection],
                          type="points"):
    raise NotImplementedError("Roboflow is not yet implemented")


class Evaluator():

    def __init__(self, df_detections, df_ground_truth, radius=150):
        self.df_detections = df_detections
        self.df_ground_truth = df_ground_truth
        self.radius = radius

        self.df_false_positives, self.df_true_positives, self.df_false_negatives = analyse_point_detections_greedy(
            df_detections=self.df_detections,
            df_ground_truth=self.df_ground_truth,
            radius=self.radius
        )
        self.precision_all = self.precision(self.df_true_positives, self.df_false_positives)
        self.recall_all = self.recall(self.df_true_positives, self.df_false_negatives)
        self.f1_all = self.f1(self.precision_all, self.recall_all)

    def precision(self, df_true_positives, df_false_positives):
            if len(df_true_positives) + len(df_false_positives) == 0:
                return 0.0
            return len(df_true_positives) / (len(df_true_positives) + len(df_false_positives))

    def recall(self, df_true_positives, df_false_negatives):
        if len(df_true_positives) + len(df_false_negatives) == 0:
            return 0.0
        return len(df_true_positives) / (len(df_true_positives) + len(df_false_negatives))

    def f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


    def get_precision_recall_f1(self, df_detections):

        df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
            df_detections=df_detections,
            df_ground_truth=self.df_ground_truth,
            radius=self.radius
        )

        precision = self.precision(df_true_positives, df_false_positives)
        recall = self.recall(df_true_positives, df_false_negatives)
        f1 = self.f1(precision, recall)

        return precision, recall, f1


    def get_precition_recall_curve(self, values: typing.List[float] = None, range_start=0, range_end=1.0, step=0.05):
        results = []
        all_errors = []

        for confidence_threshold in values:
            df_detections = self.df_detections[self.df_detections.scores >= confidence_threshold]

            df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
                df_detections=df_detections,
                df_ground_truth=self.df_ground_truth,
                radius=self.radius
            )

            precision = self.precision(df_true_positives, df_false_positives)
            recall = self.recall(df_true_positives, df_false_negatives)
            f1 = self.f1(precision, recall)

            errors = self.calculate_error_metrics(df_true_positives, df_false_positives, df_false_negatives)

            all_errors.append(errors)
            d=[confidence_threshold, precision, recall, f1]
            results.append(d)

        df_results = pd.DataFrame(results, columns=["confidence_threshold", "precision", "recall", "f1"])
        df_errors = pd.DataFrame(all_errors)


        return pd.concat([df_results, df_errors], axis=1)

    def calculate_error_metrics(self, df_true_positives: pd.DataFrame,
                                df_false_positives: pd.DataFrame,
                                df_false_negatives: pd.DataFrame):

        # Get the counting errors (your existing function)
        diffs = self.get_counting_errors(df_true_positives, df_false_positives, df_false_negatives)
        errors = np.array(diffs)

        # Calculate all error metrics with numpy
        mean_error = np.mean(errors)  # Mean Error (ME)
        mean_absolute_error = np.mean(np.abs(errors))  # Mean Absolute Error (MAE)
        mean_squared_error = np.mean(errors ** 2)  # Mean Squared Error (MSE)
        root_mean_squared_error = np.sqrt(mean_squared_error)  # RMSE (bonus)

        return {
            'mean_error': mean_error,
            'mean_absolute_error': mean_absolute_error,
            'mean_squared_error': mean_squared_error,
            'root_mean_squared_error': root_mean_squared_error,
            'total_images': len(errors)
        }

    def get_counting_errors(self, df_true_positives: pd.DataFrame,
                            df_false_positives: pd.DataFrame, df_false_negatives: pd.DataFrame):

        image_list = self.df_ground_truth['images'].unique()

        tp_counts = df_true_positives['images'].value_counts().reindex(image_list, fill_value=0).values
        fp_counts = df_false_positives['images'].value_counts().reindex(image_list, fill_value=0).values
        fn_counts = df_false_negatives['images'].value_counts().reindex(image_list, fill_value=0).values
        gt_counts = self.df_ground_truth['images'].value_counts().reindex(image_list, fill_value=0).values

        total_counts = tp_counts + fp_counts + fn_counts
        mismatched = total_counts != gt_counts

        if np.any(mismatched):
            # Only loop for warnings (much fewer iterations)
            mismatched_images = image_list[mismatched]
            for i, image in enumerate(mismatched_images):
                idx = np.where(image_list == image)[0][0]
                # logger.warning(f"Counting error in image {image}: "
                #                f"TP: {tp_counts[idx]}, FP: {fp_counts[idx]}, FN: {fn_counts[idx]}, "
                #                f"GT: {gt_counts[idx]}")

            # Vectorized calculation of diffs
        diffs = fp_counts - fn_counts

        return diffs.tolist()