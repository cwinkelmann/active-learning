from pathlib import Path

import pytest
from exif import Orientation, ResolutionUnit, Saturation, SceneCaptureType, Sharpness, WhiteBalance, MeteringMode, \
    LightSource, GpsAltitudeRef, ExposureProgram, ExposureMode, ColorSpace

from active_learning.types.image_metadata import ExifData, XMPMetaData
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage, ImageLabelCollection, \
    ImageLabel, Keypoint
from com.biospheredata.types.HastyAnnotationV2 import delete_dataset, replace_image

@pytest.fixture()
def hA():
    return HastyAnnotationV2.from_file(Path("../data/FMO02_03_05_labels.json"))

@pytest.fixture()
def aI(hA):
    #hA = HastyAnnotationV2.from_file(Path("../data/FMO02_03_05_labels.json"))

    aI = hA.get_image_by_id("02a22d60-d2e4-44c8-b0ec-231ec37cad30")
    aI.labels = aI.labels[:3]
    return aI


def test_load_hasty_annotations(hA: HastyAnnotationV2):

    assert len(hA.get_flat_df()) == 1526
    assert len(hA.images) == 31


def test_delete_dataset(hA: HastyAnnotationV2):

    dss = hA.dataset_statistics()

    assert dss == {
        'FMO02': {'num_images': 5, 'num_labels': 98},
        'FMO03': {'num_images': 14, 'num_labels': 1086},
        'FMO05': {'num_images': 12, 'num_labels': 342}
   }

    hA = delete_dataset(dataset_name="FMO03", hA=hA)

    dss = hA.dataset_statistics()
    assert dss == {
        'FMO02': {'num_images': 5, 'num_labels': 98},
        'FMO05': {'num_images': 12, 'num_labels': 342}
    }

    assert len(hA.get_flat_df()) == 0
    assert len(hA.images) == 0



def test_replace_image(hA: HastyAnnotationV2, aI: AnnotatedImage):


    hA = replace_image(updateimage=aI, hA=hA)
    rI = hA.get_image_by_id("02a22d60-d2e4-44c8-b0ec-231ec37cad30")

    assert len(rI.labels) == 3


def test_add_annotation(hA: HastyAnnotationV2):
    image_id = "02a22d60-d2e4-44c8-b0ec-231ec37cad30"
    rI = hA.get_image_by_id(image_id)

    hA.images = hA.images[:3]
    assert len(hA.images) == 3
    assert len(rI.labels) == 92

    hkp = Keypoint(
        x=int(5),
        y=int(5),
        norder=0,
        keypoint_class_id="12345",
    )

    il: ImageLabel = ImageLabel(
        class_name="iguana",
        keypoints=[hkp],
    )

    hA.add_labels_to_image(image_id=image_id,
                           dataset_name="FMO03", label=il)
    rI = hA.get_image_by_id(image_id)
    assert len(rI.labels) == 93

    hA.add_labels_to_image_by_image_name(image_name="DJI_0551.JPG",
                           dataset_name="FMO03", label=il)

    rI = hA.get_image_by_name("DJI_0551.JPG")
    assert len(rI.labels) == 94
