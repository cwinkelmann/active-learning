import typing

from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field

from com.biospheredata.converter.HastyConverter import AnnotationType, LabelingStatus
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
    images_filter: typing.Optional[List[str]] = Field(
        default=None,
        description="List of image filenames to include in the filter."
    )
    images_exclude: typing.Optional[List[str]] = Field(
        default=None,
        description="List of image filenames to exclude in the filter."
    )
    dataset_filter: typing.Optional[List[str]] = Field(
        default=None,
        description="List of dataset identifiers or categories to include."
    )
    class_filter: List[str] = Field(
        default=None,
        description="List of class labels to include in the filter."
    )
    attribute_filter: typing.Optional[List[Attribute]] =  Field(
        default=None,
        description="List of Attributes to include in the filter."
    )
    num: typing.Optional[int] = Field(
        default=None,
        description="An integer parameter, purpose depends on context."
    )
    overlap: int = Field(
        default=0,
        ge=0,
        description="Overlap size for cropping images, must be a non-negative integer.",

    )
    empty_fraction: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Fraction representing the allowable empty space, between 0.0 and 10.0. Whereas 10 means 10 times more empty images than non-empty images.",

    )
    image_tags: typing.Optional[List[str]] = Field(
        default=None,
        description="List of image tags to include in the filter."
    )
    annotation_types: typing.Optional[List[AnnotationType]] = Field(
        default=None,
        description="List of annotation types to include in the filter."
    )
    crop_size: int = Field(
        default=512,
        description="Size of the crop."
    )
    status_filter: typing.Optional[List[LabelingStatus]] = Field(
        default=None,
        description="Filter for labeling status of images."
    )


class DataPrepReport(DatasetFilterConfig):

    labels_path: typing.Optional[Path] = Field(
        None,
        description="Path to the images directory."
    )

    destination_path: typing.Optional[Path] = Field(
        None,
        description="Path to the destination directory where filtered data will be saved."
    )

