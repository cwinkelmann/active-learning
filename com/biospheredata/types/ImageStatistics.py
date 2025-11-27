"""
Collect information and statisctics about georeferenced images

@author: Christian Winkelmann

"""
import typing

import uuid

from pathlib import Path
from pydantic import BaseModel, Field

"""
@author: Christian Winkelmann
"""


class ImageCounts(BaseModel):
    image_id: str | int = Field(default=uuid.uuid4(), alias='image_id')
    image_name: str = Field(alias='image_name', description="Name of the image file")
    dataset_name: typing.Optional[str] = Field(alias='dataset_name')
    labels: typing.Dict[str, int]
