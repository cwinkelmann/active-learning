"""
inspect annotations and correct them using cvat

"""
from pathlib import Path
import fiftyone as fo
import pandas as pd

from active_learning.util.evaluation.evaluation import evaluate_predictions
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file
from examples.review_annotations import debug_hasty_fiftyone


# start with yolo

# 1. get the images from the hasty dataset
# 2. create this grid of images and remove iguanas which are on the edge



def create_fo_ds(dset, dataset_name, type, base_path, crop_size):


    labels_path = base_path / dset
    full_images_path = labels_path
    images_path = labels_path / f"crops_{crop_size}"
    images_path.mkdir(exist_ok=True, parents=True)

    hA = hA_from_file(images_path / "hasty_format.json")
    hA.images = [i for i in hA.images if len(i.labels) > 0]

    images_set = [images_path / i.image_name for i in hA.images]

    # TODO display this in fiftyOne and CVAT
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass

    # create dot annotations
    dataset = debug_hasty_fiftyone(
        annotated_images=hA.images,
        images_set=images_set,
        dataset_name=dataset_name,
        type=type,
    )

    return dataset

def create_fo_ds_v2(dset,
                    dataset_name,
                    type,
                    images_path,
                    hA_path: Path,
                    objects_threshold=-1):
    """

    :param dset:
    :param dataset_name:
    :param type:
    :param base_path:
    :param crop_size:
    :return:
    """

    hA_ground_truth = hA_from_file(hA_path)
    hA_ground_truth.images = [i for i in hA_ground_truth.images if len(i.labels) > objects_threshold]

    images_set = [images_path / i.image_name for i in hA_ground_truth.images]

    # create dot annotations
    dataset = evaluate_predictions(
        ground_truth_labels=hA_ground_truth.images,
        prediction_labels=hA_ground_truth.images,
        images_set=images_set,
        dataset_name=dataset_name,
        type=type,
    )

    return dataset



if __name__ == "__main__":
    # /Users/christian/data/training_data/2024_12_11/train/crops_640_num1_overlap0
    analysis_date = "2024_12_16"
    lcrop_size = 640
    num = 56

    just_check_fiftyone = True

# '/Users/christian/data/training_data/2024_12_11/2024_12_14/train/crops_640_num1_overlap0/crops_640
    dset = "train"


    # lbase_path = Path(f"/Users/christian/data/training_data/2024_12_11/{analysis_date}/{dset}/crops_{lcrop_size}_num{num}_overlap0")
    label_base_path = Path(f"/Users/christian/data/training_data/{analysis_date}/{dset}/crops_640_num1_overlap0")

    df_point_prediction = pd.DataFrame("/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2025-01-22/11-21-30/detections.csv")

    # lbase_path = Path(f"/Users/christian/data/training_data/2024_12_16/val/crops_640_numNone_overlap0")

    dataset_name = f"eal_{analysis_date}_{dset}_review_{lcrop_size}"
    type = "boxes"

    ## TODO display this in fiftyOne and CVAT
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass

    hA_ground_truth = hA_from_file(label_base_path / "hasty_format.json")

    images_set = [label_base_path / i.image_name for i in hA_ground_truth.images]

    # create dot annotations
    ds = evaluate_predictions(
        ground_truth_labels=hA_ground_truth.images,
        df_point_prediction=df_point_prediction,
        images_set=images_set,
        dataset_name=dataset_name,
        type=type,
    )

    results = ds.evaluate_detections(
        pred_field="predictions_boxes",
        gt_field="ground_truth_boxes",
        eval_key="eval",
    )

    results.print_report()

    # #ds = create_fo_ds(dset, dataset_name, type=type, base_path=lbase_path, crop_size=lcrop_size)
    # ds = create_fo_ds_v2(dset,
    #                      dataset_name,
    #                      type=type,
    #                      images_path=lbase_path,
    #                      hA_path=lbase_path / "hasty_format.json",
    #                      objects_threshold=0)

    ## fiftyOne check
    type = "boxes"
    session = fo.launch_app(ds)
    session.wait()

    # CVAT correction
    ds.annotate(
        anno_key=dataset_name,
        label_field=f"ground_truth_{type}",
        attributes=["iscrowd", "sth", "custom_attribute", "tags"],
        launch_editor=True,
    )

    # download the corrected dataset

