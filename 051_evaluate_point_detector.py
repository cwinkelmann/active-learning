"""
Take Detections from a model and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
import PIL.Image
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections, \
    analyse_point_detections_greedy
# import pytest

from active_learning.analyse_detections_yolo import analyse_point_detections_
from active_learning.util.visualisation.draw import draw_text
from com.biospheredata.converter.Annotation import project_point_to_crop
from com.biospheredata.image.image_manipulation import crop_out_images_v3

import shapely

from util.util import visualise_image


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

    df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-15/16-14-19/detections.csv')
    df_ground_truth = pd.read_csv('/Users/christian/data/training_data/2024_12_09_debug/test/herdnet_format.csv')
    images_path = Path("/Users/christian/data/training_data/2024_12_09_debug/test/Default")
    images = images_path.glob("*.JPG")

    # df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections(df_detections, df_ground_truth)
    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_greedy(df_detections=df_detections, df_ground_truth=df_ground_truth, radius=150)


    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")

    box_size = 250

    for i in images:
        df_fp = df_false_positives[df_false_positives.images == i.name]
        df_tp = df_true_positives[df_true_positives.images == i.name]
        df_fn = df_false_negatives[df_false_negatives.images == i.name]

        draw_thumbnail(df_fp, i, "fp")
        draw_thumbnail(df_fn, i, "fn")
        draw_thumbnail(df_tp, i, "tp")




    # try:
    #     dataset = fo.load_dataset("test_evaluations")
    #     dataset.delete()
    # except:
    #     pass
    #
    # dataset = fo.Dataset("test_evaluations")
    # dataset.persistent = True
    #
    #
    # for i in images:
    #
    #     debug_hasty_fiftyone_v3(image_path=i, df_labels=df_false_positives, dataset=dataset)
    #
    # dataset.save()
    # session = fo.launch_app(dataset, port=5151)
    # session.refresh()
    # session.wait()