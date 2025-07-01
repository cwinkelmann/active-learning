# TODO Implement or delete this code

from loguru import logger
from shapely.geometry.point import Point
from typing import List

from active_learning.pipelines.predictor import Predictor
from active_learning.util.projection import local_coordinates_to_wgs84
from com.biospheredata.types.HastyAnnotationV2 import PredictedImageLabel
import geopandas as gpd

class HerdnetPredictor(Predictor):
    """
    Herdnet prediction on geospatial data
    """
    def __init__(self, model_path, confidence_threshold=0.3):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def _predict(self, image_path) -> List[PredictedImageLabel]:
        """
        model prediction
        :param image_path:
        :return:
        """
        ## TODO implement some logic so we can deal with Orthomosaics:

        raise NotImplementedError("Not implemented yet")


    def predict(self, image_path, local=False) -> List[PredictedImageLabel]:
        """

        :param image_path:
        :param local:
        :return:
        """

        prediction = self._predict(image_path)
        if local:
            return prediction
        else:
            ## TODO project the prediction to the geospatial coordinates
            projected_prediction = local_coordinates_to_wgs84(georeferenced_tiff_path=image_path, annotations=prediction)
            data = []


            ## TOOD implement some logic so we can deal with Orthomosaics:
            return projected_prediction
