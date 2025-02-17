import sahi.prediction
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

from active_learning.pipelines.predictor import Predictor


class YoloPredictor(Predictor):
    """
    Sliced Yolo prediction
    """
    def __init__(self, yolo11n_model_path, confidence_threshold=0.3):

        self.detection_model = detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolo11', # or 'yolov8'
        model_path=yolo11n_model_path,
        confidence_threshold=confidence_threshold,
        device="mps", # or 'cuda:0'
    )

    def predict(self, image_path) -> sahi.prediction.PredictionResult:
        image = read_image(image_path)

        return get_prediction(image=image, detection_model=self.detection_model)

    def sliced_predict(self, image_path, crop_size=640, overlap=0.2) -> sahi.prediction.PredictionResult:
        image = read_image(image_path)

        sliced_prediction = get_sliced_prediction(
            image = image,
            detection_model=self.detection_model,
            slice_height=crop_size,
            slice_width=crop_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap
        )

        return sliced_prediction

