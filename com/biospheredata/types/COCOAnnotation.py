"""
COCO Annotation Model
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union


class Category(BaseModel):
    id: int
    name: str
    supercategory: Optional[str] = None


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: Optional[Union[List[float], List[List[float]]]] = None  # Polygon or RLE
    area: Optional[float] = None
    bbox: Optional[List[float]] = None  # [x, y, width, height]
    iscrowd: Optional[int] = None


class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None


class COCOAnnotations(BaseModel):
    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]
    licenses: Optional[List[dict]] = None
    info: Optional[dict] = None
