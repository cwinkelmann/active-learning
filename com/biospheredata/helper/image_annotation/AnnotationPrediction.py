import sahi.prediction
from loguru import logger
import PIL


class metadataResolver(object):
    """
    every image has a lot of metadata, which is necessary for a good annotation formatting later
    """


class jsonMetadataResolver(metadataResolver):
    """
    use a static json for testing purposes
    """

    def __init__(self, metadata: dict):
        self.metadata = metadata


class HastyMetadataStub(object):
    def __init__(self, image):
        self.image_id = image


class sahiMetadataResolver(metadataResolver):
    def __init__(self):
        pass





class AnnotationPrediction(object):
    """
    predict objection
    """

    def __init__(self, datasetName):
        self.datasetName = datasetName

    def set_class_names(self, metaDataResolver: metadataResolver):
        self.metaDataResolver = metaDataResolver


    def coco_annotation_hasty(self, x):
        """
            df_boxes["h"] = abs(df_boxes["y1"] - df_boxes["y2"])
            df_boxes["w"] = abs(df_boxes["x1"] - df_boxes["x2"])

            ## TODO visualize the boxes before transformation

            df_boxes["class"] = class_names
            df_boxes["x_yolo"] = abs(df_boxes["x1"] + df_boxes["x2"]) * 0.5 / image_width
            df_boxes["y_yolo"] = abs(df_boxes["y1"] + df_boxes["y2"]) * 0.5 / image_height

            df_boxes["w_yolo"] = df_boxes["w"] / image_width
            df_boxes["h_yolo"] = df_boxes["h"] / image_height

            HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point
            YOLO Format is x,y of the center point, width, height
            https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

        @param x:
        @return:
        """

        ## hasty
        # df_boxes = pd.DataFrame(boxes)
        # df_boxes.columns = ["x1", "y1", "x2", "y2"]
        #
        bbox = [
            102, #
            45,
            420,
            404
        ],
        bbox = x.x1y1x2y2.to_coco_bbox()

        label = {
                            "class_name": x.category.name,
                            "bbox": bbox,
                            "polygon": None,
                            "mask": None,
                            "z_index": 1,
                            "attributes": {
                                "attribute_name": "xyz"
                            }
                        }

        print(x)
        return label


    def sahi_predictions_to_hasty(self, image_name, dataset_name, prediction: sahi.prediction.PredictionResult):
        """
        convert the predictions back to hasty format


        :param path_to_predictions:
        :return:
        """

        images = [
                {
                    "image_name": image_name,
                    "dataset_name": dataset_name,
                    "width": prediction.image_height,
                    "height": prediction.image_width,
                    "image_status": "TO REVIEW",
                    "labels": [self.coco_annotation_hasty(x) for x in prediction.object_prediction_list],
                    "tags": [
                        "TODO",
                        "FIXME"
                    ]
                }
            ]
        return images
