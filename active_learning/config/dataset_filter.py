from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


class DatasetFilterConfig(BaseModel):
    """
    Configuration model for filtering datasets and images.
    """
    dset: str = Field(
        ...,
        description="Name of the dataset, e.g., 'train', 'validation', 'test'."
    )
    images_filter: List[str] = Field(
        default=None,
        description="List of image filenames to include in the filter."
    )
    dataset_filter: List[str] = Field(
        default=None,
        description="List of dataset identifiers or categories to include."
    )
    num: int = Field(
        default=None,
        description="An integer parameter, purpose depends on context."
    )
    output_path: Union[Path, str] = Field(
        ...,
        description="Path to the output directory or file for labels."
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

