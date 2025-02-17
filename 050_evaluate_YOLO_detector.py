"""
Take Detections from a model and display them in a FiftyOne Dataset

TODO: why are there missing objects
"""
from pathlib import Path

# import pytest

from active_learning.analyse_detections_yolo import analyse_point_detections_
from active_learning.pipelines.yolo_predict import YoloPredictor
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file
from examples.review_annotations import debug_hasty_fiftyone_v3
import fiftyone as fo

def analyse_detections_yolo(image_path: Path, dataset: fo.Dataset):
    """
    filter for false positives and prepare data to ingest it into a review process
    :return:
    """
    from sahi.utils.ultralytics import (
        download_yolo11n_model, download_yolo11n_seg_model,
        # download_yolov8n_model, download_yolov8n_seg_model
    )


    hA_gt = hA_from_file(Path("/Users/christian/data/training_data/2024_12_16/train/hasty_format_iguana.json"))
    hA_gt = hA_from_file(Path("/Users/christian/data/training_data/2024_12_16/val/hasty_format_iguana.json"))
    # arrange an instance segmentation model for test
    # image_path = Path("/Users/christian/data/training_data/2024_12_16/train/Default/FMO05___DJI_0591.JPG")

    image_gt = [i for i in hA_gt.images if i.image_name == image_path.name][0]

    yolo11n_model_path = "/Users/christian/Downloads/best.pt"


    yp = YoloPredictor(yolo11n_model_path=yolo11n_model_path, confidence_threshold=0.5)
    results = yp.sliced_predict(image_path)

    r = analyse_point_detections_(hA_gt=image_gt, predictions=results)

    # assert r[r.kind == "true_positive"].shape[0] == 3, "There should be three true positives"
    # assert r[r.kind == "false_negative"].shape[0] == 10, "10 Objects were missed"

    # assert r.to_dict(orient="records") == {} # TODO


    dataset = debug_hasty_fiftyone_v3(image_path=image_path, df_labels=r, dataset=dataset)

    return dataset


if __name__ == "__main__":
    # Create an empty dataset, TODO put this away so the dataset is just passed into this

    try:
        dataset = fo.load_dataset("test_evaluations")
        dataset.delete()
    except:
        pass

    dataset = fo.Dataset("test_evaluations")
    dataset.persistent = True

    images = Path("/Users/christian/data/training_data/2024_12_16/test/Default").glob("*.JPG")
    for i in images:
        # = Path("/Users/christian/data/training_data/2024_12_16/val/Default/FMO03___DJI_0432.JPG")
        dataset = analyse_detections_yolo(i, dataset)
    dataset.save()
    session = fo.launch_app(dataset, port=5151)
    session.refresh()
    session.wait()