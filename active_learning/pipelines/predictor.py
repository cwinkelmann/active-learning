# TODO implement or delete this code
from typing import List

from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel


class Predictor(object):

    def predict(self, image_path) -> List[PredictedImageLabel]:
        raise NotImplementedError

