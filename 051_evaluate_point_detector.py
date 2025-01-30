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
from active_learning.util.visualisation.draw import draw_text
from com.biospheredata.converter.Annotation import project_point_to_crop
from com.biospheredata.image.image_manipulation import crop_out_images_v3

import shapely
import fiftyone as fo
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, ImageLabelCollection, AnnotatedImage


def evaluate_predictions(
        # images_set: typing.List[Path],
        sample: fo.Sample,
        # images_dir: Path,
        predictions: typing.List[ImageLabelCollection] = None,
        type="points", sample_field_name="prediction", ):
    """
    Evaluate the predictions
    :return:
    """
    image_name = sample.filename

    # for image_path in images_set:
    filtered_predictions = [p for p in predictions if p.image_name == image_name]
    if len(filtered_predictions) == 1:

        hA_image_pred = filtered_predictions[0]

        if type == "points":
            keypoints_pred = _create_fake_boxes(hA_image=hA_image_pred)
        elif type == "boxes":
            boxes_pred = _create_boxes_s(hA_image=hA_image_pred)
        elif type == "polygons":
            raise NotImplementedError("Polygons are not yet implemented")
            #polygons = _create_polygons_s(hA_image=hA_image)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")
        # FIXME, if this function is called multiple times, the same image will be added multiple times


        if type == "points":
            sample[sample_field_name] = fo.Detections(detections=keypoints_pred)
        elif type == "boxes":
            sample[sample_field_name] = fo.Detections(detections=boxes_pred)
        elif type == "polygons":
            raise NotImplementedError("Polygons are not yet implemented")
            # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")

        # logger.info(f"Added {image_name} to the dataset")

        sample.save()
    else:
        logger.warning(f"There should be one single image left, but {len(filtered_predictions)} are left.")

    return sample


def evaluate_ground_truth(
        # images_set: typing.List[Path],
        sample: fo.Sample,
        # images_dir: Path,
        ground_truth_labels: typing.List[AnnotatedImage] = None,
        type="points", sample_field_name="ground_truth", ):
    """
    Evaluate the predictions against the ground truth in
    :return:
    """
    # dataset = fo.load_dataset(dataset_name) # loading a bad idea because the single source of truth is the hasty annotations
    image_name = sample.filename

    hA_gt_sample = [i for i in ground_truth_labels if i.image_name == sample.filename]
    assert len(hA_gt_sample) == 1, f"There should be one single image left, but {len(hA_gt_sample)} are left."

    hA_image = hA_gt_sample[0]

    if type == "points":
        keypoints = _create_keypoints_s(hA_image=hA_image)
    elif type == "boxes":
        boxes = _create_boxes_s(hA_image=hA_image)
    elif type == "polygons":
        raise NotImplementedError("Polygons are not yet implemented")
        #polygons = _create_polygons_s(hA_image=hA_image)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    sample.tags=hA_image.tags
    sample["hasty_image_id"] = hA_image.image_id
    sample["hasty_image_name"] = hA_image.image_name

    if type == "points":
        sample["ground_truth"] = fo.Keypoints(keypoints=keypoints)
    elif type == "boxes":
        sample["ground_truth"] = fo.Detections(detections=boxes)
    elif type == "polygons":
        raise NotImplementedError("Polygons are not yet implemented")
        # sample['ground_truth_polygons'] = fo.Polylines(polyline=polygons)
    else:
        raise ValueError("Unknown type, use 'boxes' or 'points'")

    # logger.info(f"Added {image_name} to the dataset")

    sample.save()

    return sample


def create_box_around(point: shapely.geometry.Point, box_width: float, box_height: float) -> shapely.geometry.Polygon:
    """
    Create a rectangular box (polygon) centered at a given point.

    Parameters:
        point: Shapely Point around which to create the box.
        box_width: Total width of the box.
        box_height: Total height of the box.

    Returns:
        A Shapely Polygon representing the box.
    """
    x, y = point.x, point.y
    half_w = box_width / 2.0
    half_h = box_height / 2.0
    # shapely.geometry.box(minx, miny, maxx, maxy)
    return shapely.geometry.box(x - half_w, y - half_h, x + half_w, y + half_h)

def draw_thumbnail(df, i, suffix):
    ts_path = images_path.parent / f"thumbnails_{suffix}"
    ts_path.mkdir(exist_ok=True)

    if len(df) > 0:
        box_polygons = [create_box_around(point, box_size, box_size) for point in df.geometry]
        df_fp_list = df.to_dict(orient="records")
        crops = crop_out_images_v3(image=PIL.Image.open(i), rasters=box_polygons)
        # TODO add every point to theses boxes, some might not be in the center.
        projected_points = [project_point_to_crop(point, crop_box)
                            for point, crop_box in zip(df.geometry, box_polygons)]

        for idx, (crop, point) in enumerate(zip(crops, projected_points)):
            crop = draw_text(crop, f"{df_fp_list[idx].get('species', '')} | {df_fp_list[idx].get('scores')}%",
                             position=(10, 5), font_size=int(0.08 * box_size))
            crop.save(ts_path / f"{Path(i.name).stem}_{suffix}_{idx}.JPG")
            # visualise_image(image=crop, show=True)


if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this
    analysis_date = "2024_12_09"
    # lcrop_size = 640
    num = 56
    type = "points"

    # test dataset
    base_path = Path(f'/Users/christian/data/training_data/{analysis_date}_debug/test/')
    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    images_path = base_path / "Default"

    # IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)
    hA_ground_truth = base_path / 'hasty_format_iguana_point.json'

    df_ground_truth = pd.read_csv(base_path / 'herdnet_format.csv')

    # # val dataset
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/19-17-14/detections.csv')
    # df_ground_truth = pd.read_csv('/Users/christian/data/training_data/2024_12_09_debug/val/herdnet_format.csv')
    # images_path = Path("/Users/christian/data/training_data/2024_12_09_debug/val/Default")

    hA_ground_truth = hA_from_file(hA_ground_truth)
    images = images_path.glob("*.JPG")

    # df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections(df_detections, df_ground_truth)
    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(
        df_detections=df_detections, df_ground_truth=df_ground_truth, radius=150)
    df_concat = pd.concat([df_false_positives, df_true_positives, df_false_negatives])


    IL_all_detections = herdnet_prediction_to_hasty(df_concat, images_path)
    IL_fp_detections = herdnet_prediction_to_hasty(df_false_positives, images_path)
    IL_tp_detections = herdnet_prediction_to_hasty(df_true_positives, images_path)
    IL_fn_detections = herdnet_prediction_to_hasty(df_false_negatives, images_path)


    dataset_name = f"eal_{analysis_date}_review"

    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")


    box_size = 350

    ## TODO Skipped Thumbnail create
    for i in images:
        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]

        draw_thumbnail(df_fp, i, "fp")
        draw_thumbnail(df_fn, i, "fn")
        draw_thumbnail(df_tp, i, "tp")



    images_set = [images_path / i.image_name for i in hA_ground_truth.images]

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