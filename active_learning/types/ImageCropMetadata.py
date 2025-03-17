import json

from pathlib import Path

import typing
from pydantic import BaseModel, Field

from pydantic import BaseModel
from shapely.geometry import Polygon, Point
from typing import Union, Optional, List
from uuid import UUID

from com.biospheredata.types.HastyAnnotationV2 import Keypoint


class ImageCropMetadata(BaseModel):
    parent_image: str
    parent_image_id: Union[str, UUID]  # Assuming the image ID is a string or UUID
    parent_label_id: Union[str, UUID]  # Assuming the label ID is a string or UUID
    cropped_image: str
    cropped_image_id: Union[str, UUID]
    bbox: Optional[List[typing.Union[int, float]]] = Field(None, alias='bbox')  # Shapely polygon for bounding box
    local_coordinate: Optional[List[Keypoint]]  # Point within the cropped image
    global_coordinate: Optional[List[Keypoint]]  # Original coordinate before cropping

    class Config:
        arbitrary_types_allowed = True  # Required to support shapely types


    def save(self, file_path: Path):
        with open(file_path, 'w') as json_file:
            # Serialize the list of Pydantic objects to a list of dictionaries
            json_file.write(self.model_dump_json())

    @staticmethod
    def load(file_path: Path) -> "ImageCropMetadata":
        """
        Load a HastyAnnotationV2 object from a file
        :param file_path:
        :return:
        """
        with open(file_path, mode="r") as f:
            data = json.load(f)
            iCM = ImageCropMetadata(**data)
        return iCM