from pathlib import Path

import pytest
from exif import Orientation, ResolutionUnit, Saturation, SceneCaptureType, Sharpness, WhiteBalance, MeteringMode, \
    LightSource, GpsAltitudeRef, ExposureProgram, ExposureMode, ColorSpace

from active_learning.types.image_metadata import ExifData, XMPMetaData
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage
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