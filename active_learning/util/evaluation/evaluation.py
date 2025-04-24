"""
Take Detections from a model and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
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
    TODO get the comments right. This function
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
                          type:AnnotationType =AnnotationType.KEYPOINT):
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