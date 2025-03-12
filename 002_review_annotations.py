"""
inspect annotations and correct them using cvat, fifty one or roboflow

"""
from pathlib import Path
import fiftyone as fo

from active_learning.util.visualisation.fiftyone import hastyAnnotation2fiftyOne
from com.biospheredata.converter.HastyConverter import AnnotationType
from examples.review_annotations import debug_hasty_fiftyone, debug_hasty_fiftyone_v2
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file

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

    hA = hA_from_file(hA_path)
    hA.images = [i for i in hA.images if len(i.labels) > objects_threshold]

    images_set = [images_path / i.image_name for i in hA.images]

    # TODO use code from this repo, evaluate_predictions is very simiar to this
    # create dot annotations
    dataset = debug_hasty_fiftyone_v2(
        annotated_images=hA.images,
        images_set=images_set,
        dataset_name=dataset_name,
        type=type,
    )

    return dataset



if __name__ == "__main__":
    # /Users/christian/data/training_data/2024_12_11/train/crops_640_num1_overlap0

    just_check_fiftyone = True
    #eal_2024_12_09_review_hasty_corrected_jpg.json

    analysis_date = "2024_12_09"
    type = "points"
    dataset_name = f"eal_{analysis_date}_review"
    base_path = Path(f'/Users/christian/data/orthomosaics/tiles')
    # label_path = Path("/Users/christian/data/orthomosaics/tiles/object_crops/eal_2024_12_09_review_hasty.json") # wrong
    label_path = Path("/Users/christian/data/orthomosaics/tiles/object_crops/eal_2024_12_09_review_hasty_corrected_jpg.json") # corrected
    images_path = base_path

    output_path = base_path / "object_crops"

    hA = hA_from_file(label_path)


    ## TODO display this in fiftyOne and CVAT
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass

    ds = hastyAnnotation2fiftyOne(hA, images_path, annoation_type=AnnotationType.KEYPOINT)

    if just_check_fiftyone:
        session = fo.launch_app(ds)
        session.wait()

    else:
        # CVAT correction
        ds.annotate(
            anno_key=dataset_name,
            label_field=f"ground_truth_{type}",
            attributes=["iscrowd", "sth", "custom_attribute", "tags"],
            launch_editor=True,
        )


