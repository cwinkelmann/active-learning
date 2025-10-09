import json
import pandas as pd

import tempfile

import pytest
import numpy as np
from pathlib import Path

import shapely
from shapely.geometry import Polygon

from active_learning.util.converter import coco2hasty
from com.biospheredata.converter.HastyConverter import HastyConverter
from active_learning.filter import ImageFilterConstantNum, sample_coco
from active_learning.types.filter import SampleStrategy
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2


@pytest.fixture
def iSAID_annotations_path():
    return Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/iSAID/train/Annotations/iSAID_train.json")

@pytest.fixture
def hA():
    return hA_from_file(
        file_path=Path(__file__).parent / "data/annotations/annotations_FMO04_DJI_0049.JPG.json")

@pytest.fixture
def hA_segment():
    return hA_from_file(
        file_path=Path(__file__).parent / "data/hasty_format_crops_segments.json")

@pytest.fixture
def template_image_path():
    return Path(__file__).parent / "data/images/FMO04/templates/template_source_DJI_0049.1280.jpg"

@pytest.fixture
def source_image_path():
    return Path(__file__).parent / "data/images/FMO04/single_images/DJI_0049.JPG"

@pytest.fixture
def images_path():
    return Path(__file__ ).parent / "data/images/FMO04"

@pytest.fixture
def isaid_images_path():
    return Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/iSAID/DOTA/train/images")

@pytest.fixture
def output_path():
    return Path(__file__ ).parent / "output/cutouts/"

def test_iSAID2COCO(iSAID_annotations_path: Path,
                                         isaid_images_path: Path
                                         ):
    """
    convert coco to hasty annotations

    :param hA:
    :param images_path:
    :return:
    """

    with open(iSAID_annotations_path, "r") as f:
        coco_data = json.load(f)
    n = 2
    sampled_data = sample_coco(coco_data, n=n, method=SampleStrategy.RANDOM)

    hA = coco2hasty(coco_data=sampled_data, images_path=isaid_images_path)

    hA.save(iSAID_annotations_path.parent / "iSAID_hasty_train.json")
    assert len(hA.images) == n

def test_sample_coco(iSAID_annotations_path):
    with open(iSAID_annotations_path, "r") as f:
        coco_data = json.load(f)
    assert len(coco_data["images"]) == 1411

    n = 20
    sampled_data = sample_coco(coco_data, n=n, method="random")

    assert sampled_data["categories"] == coco_data["categories"]
    assert len(sampled_data["images"]) == n
    assert len(sampled_data["annotations"]) >= n
    assert len(sampled_data["annotations"]) != len(coco_data["annotations"]), "the sample should contain less annotations than the original dataset"

    image_ids_in_annotations = {image["id"] for image in sampled_data["images"]}
    assert len(image_ids_in_annotations) == n, "all image ids should be unique"

    sample_dataset_path = iSAID_annotations_path.parent / "sampled_iSAID_train.json"
    # Save the sampled dataset
    with open(sample_dataset_path, "w") as f:
        json.dump(sampled_data, f)



def test_hasty_to_yolo(hA_segment: HastyAnnotationV2):
    class_mapping = HastyConverter.get_label_class_mapping(hA_segment)
    # an_image = hA_segment.sample(n=1, how=SampleStrategy.FIRST).
    an_image = hA_segment.images[0]
    df_annotations = HastyConverter.to_yolo_segment(class_mapping=class_mapping,
                                                    df_all_boxes_list=[],
                                   image=an_image)

    assert len(df_annotations) == 1
    assert len(df_annotations.columns) == 91

    an_image = hA_segment.images[1]
    df_annotations = HastyConverter.to_yolo_segment(class_mapping=class_mapping,
                                                    df_all_boxes_list=[],
                                   image=an_image)

    assert len(df_annotations) == 2, "Two rows should be returned"
    assert len(df_annotations.columns) == 87, "The number of columns should be 87"

def test_hasty_to_deepforest(hA_segment: HastyAnnotationV2):

    class_mapping = HastyConverter.get_label_class_mapping(hA_segment)
    # an_image = hA_segment.sample(n=1, how=SampleStrategy.FIRST).
    an_image = hA_segment.images[0]
    with tempfile.TemporaryDirectory() as tmpFolder:
        tmp_file = Path(tmpFolder) / "deepforest.csv"
        df_annotations = HastyConverter.convert_deep_forest(hA=hA_segment, output_file=Path(tmp_file))

        df_deepforest = pd.read_csv(tmp_file)

        assert len(df_deepforest.columns) == 6, "The number of columns should be 6"
        assert list(df_deepforest.columns) ==  ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]

        first_image = df_deepforest.to_dict(orient="records")[0]

        assert first_image == {'image_path': 'Fer_FCD01-02-03_20122021_single_images___DJI_0079_FCD01_x4096_y512.jpg', 'label': 'iguana', 'xmax': 168.0, 'xmin': 51.0, 'ymax': 477.0, 'ymin': 426.0}


def test_hasty_to_herdnetbox(hA_segment: HastyAnnotationV2):
    class_mapping = HastyConverter.get_label_class_mapping(hA_segment)
    # an_image = hA_segment.sample(n=1, how=SampleStrategy.FIRST).
    an_image = hA_segment.images[0]


    with tempfile.TemporaryDirectory() as tmpFolder:
        tmp_file = Path(tmpFolder) / "herdnet_box.csv"

        df_annotations = HastyConverter.convert_to_herdnet_box_format(hA=hA_segment, output_file=tmp_file)

        df_herdnet = pd.read_csv(tmp_file)

        assert len(df_herdnet.columns) == 7, "The number of columns should be 7"
        assert list(df_herdnet.columns) ==  ["images", "x_min", "y_min", "x_max", "y_max", "species", "labels"]

        first_image = df_herdnet.to_dict(orient="records")[0]

        assert first_image == {'images': 'Fer_FCD01-02-03_20122021_single_images___DJI_0079_FCD01_x4096_y512.jpg', 'labels': 1, 'species': 'iguana', 'x_max': 168.0, 'x_min': 51.0, 'y_max': 477.0, 'y_min': 426.0}


def test_sort():
    data = {
        'bird': 8,
        'crab': 1,
        'iguana': 0,
        'iguana_point': 7,
        'not_iguana_but_similar_look': 6,
        'seal': 3,
        'trash': 5,
        'turtle': 2,
        'ugly_stone': 4
    }

    # Sort by the dictionary value, but store only the keys in a list
    sorted_data = [key for key, value in sorted(data.items(), key=lambda item: item[1])]
    print(sorted_data)

    # Make sure to compare to a list of keys (strings), not a list of tuples
    assert sorted_data == [
        'iguana',               # 0
        'crab',                 # 1
        'turtle',               # 2
        'seal',                 # 3
        'ugly_stone',           # 4
        'trash',                # 5
        'not_iguana_but_similar_look',  # 6
        'iguana_point',         # 7
        'bird'                  # 8
    ]