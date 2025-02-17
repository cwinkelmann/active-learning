from pathlib import Path

import pytest

from active_learning.filter import ImageFilterConstantNum
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, hA_from_file


@pytest.fixture()
def hA():
    return hA_from_file(Path("data/FMO02_03_05_labels.json"))

def test_ImageFilterConstantNum(hA):
    ifcn = ImageFilterConstantNum(num=1)

    s1 = ifcn(hA=hA)
    s2 = ifcn(hA=hA)

    assert s1[0].image_id == '7f0e7111-eebe-462d-abb1-345ecfc84e7d'
    assert s2[0].image_id == '10570906-8af8-45b8-a212-4af1a50d81f8'

    assert s1 == s2, "ImageFilterConstantNum should return the same sample if called twice because of the seed"

    raise NotImplementedError("Test not implemented")