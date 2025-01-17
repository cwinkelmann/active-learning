
import json
from pathlib import Path
from pydantic import BaseModel
from pathlib import Path
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any

from active_learning.types.image_metadata import xmp_metadata, get_exif_metadata, ExifData
from util.georeferenced_image import XMPMetaData
from util.util import get_image_id



class GeoreferencedImage(BaseModel):
    """
    retrieve metadata from a georeferenced mavic 2 pro image
    """
    image_path: str
    image_id: Optional[str] = None
    image_name: Optional[str] = None
    exif_metadata: Optional[ExifData] = None
    xmp_metadata: XMPMetaData = None


    @classmethod
    def from_path(cls, image_path: Path):
        return cls(
            image_path=str(image_path),
            image_name=image_path.name,
            exif_metadata=get_exif_metadata(image_path),
            image_id=get_image_id(image_path),
            xmp_metadata=xmp_metadata(image_path) # TODO implement this right
        )







