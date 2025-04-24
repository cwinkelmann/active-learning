from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field

from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, Attribute


class DatasetFilterConfig(BaseModel):
    """
    Configuration model for filtering datasets and images.
    """
    dset: str = Field(
        ...,
        description="Name of the dataset, e.g., 'train', 'validation', 'test'."
    )
    dataset_name: str = Field(
        default="",
        description="Unique naming for identifying the dataset."
    )
    images_filter: List[str] = Field(
        default=None,
        description="List of image filenames to include in the filter."
    )
    images_exclude: List[str] = Field(
        default=None,
        description="List of image filenames to exclude in the filter."
    )
    dataset_filter: List[str] = Field(
        default=None,
        description="List of dataset identifiers or categories to include."
    )
    class_filter: List[str] = Field(
        default=None,
        description="List of class labels to include in the filter."
    )
    attribute_filter: List[Attribute] =  Field(
        default=None,
        description="List of Attributes to include in the filter."
    )
    num: int = Field(
        default=None,
        description="An integer parameter, purpose depends on context."
    )

    empty_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction representing the allowable empty space, between 0.0 and 1.0."
    )
    image_tags: List[str] = Field(
        default=None,
        description="List of image tags to include in the filter."
    )
    annotation_types: List[AnnotationType] = Field(
        default=None,
        description="List of annotation types to include in the filter."
    )
    crop_size: int = Field(
        default=512,
        description="Size of the crop."
    )


class DataPrepReport(DatasetFilterConfig):

    images_path: Path = Field(
        ...,
        description="Path to the images directory."
    )
    annotation_data: HastyAnnotationV2 = Field(
        ...,
        description="HastyAnnotationV2 object containing the annotations."
    )

    yolo_boxes_path: Path = Field(
        None,
        description="Path to the labels directory."
    )
    yolo_segments_path: Path = Field(
        None,
        description="Path to the labels directory."
    )

