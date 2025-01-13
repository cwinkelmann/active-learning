"""
testing various data preparation steps
"""
import json

import tempfile

import pytest
import numpy as np
from pathlib import Path


from active_learning.pipelines.data_prep import AnnotationsIntermediary, UnpackAnnotations, DataprepPipeline, \
    AnnotationFormat
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


@pytest.fixture
def iSAID_annotations_path_train():
    return Path("/Users/christian/data/training_data/2025_01_08_isaid/train/Annotations/iSAID_train.json")

@pytest.fixture
def iSAID_annotations_path_val():
    return Path("/Users/christian/data/training_data/2025_01_08_isaid/val/Annotations/iSAID_val.json")

@pytest.fixture
def iSAID_images_path_train():
    return Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/train/images")

@pytest.fixture
def iSAID_images_path_val():
    return Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/val/images")

def test_UnpackAnnotations_hasty():
    # TODO implement me
    uA = UnpackAnnotations()
    uA.unzip_hasty()


    aI = uA.get_intermediary_annotations()

    assert isinstance(aI, AnnotationsIntermediary)

    assert aI.image_list() is not None


def test_UnpackAnnotations_iSAID():
    # TODO implement me
    uA = UnpackAnnotations()
    uA.unzip_iSAID()

    aI = uA.get_intermediary_annotations()

    assert isinstance(aI, AnnotationsIntermediary)



def test_AnnotationsIntermediary(iSAID_annotations_path_train: Path,
                                         iSAID_images_path_train: Path
                                         ):
    """
    convert any format to any other format

    :param hA:
    :param images_path:
    :return:
    """

    # TODO implement me

    aI = AnnotationsIntermediary()
    with open(iSAID_annotations_path_train, "r") as f:
        coco_data = json.load(f)

    aI.set_coco_annotations(coco_data=coco_data, images_path=iSAID_images_path_train)


    hA = aI.get_hasty_annotations()
    assert isinstance(hA, HastyAnnotationV2)

    aI.to_YOLO_annotations(output_path=Path("data/annotations/yolo"))
    aI.get_deepforest_annotations(output_path=Path("data/annotations/yolo"))




def test_DataprepPipeline():
    """
    test the data preparation pipeline
    :return:
    """
    # TODO implement me
    dP = DataprepPipeline()

    # TODO get the annotations in various formats


def test_full_run_isaid():
    """ test everything at once"""

    # unzip the annotations

    # convert the annotations

    # process
    pass

