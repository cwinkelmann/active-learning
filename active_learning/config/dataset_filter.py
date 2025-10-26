import json
import typing
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

from active_learning.types.filter import SampleStrategy
from com.biospheredata.types.status import LabelingStatus, AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import Attribute


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
        description="Amount of images to include"
    )
    num_labels: typing.Optional[int] = Field(
        default=None,
        description="Amount of labels to include in total."
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
    crop: bool = Field(
        default=True,
        description="Whether to crop at all"
    )
    status_filter: typing.Optional[List[LabelingStatus]] = Field(
        default=None,
        description="Filter for labeling status of images."
    )
    edge_black_out: bool = Field(
        default=False,
        description="Whether to apply edge blackout to images."
    )
    remove_default_folder: bool = Field(
        default=False,
        description="Whether to remove the 'Default' images folder should be removed, which contains all images from the subsets."
    )
    remove_padding_folder: bool = Field(
        default=True,
        description="Whether to remove the 'padding_folder' should be removed, which contains all images from the subsets."
    )
    sample_strategy: SampleStrategy = Field(
        default=SampleStrategy.RANDOM,
        description="Strategy for sampling images from the dataset."
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
    num_images_filtered: int = Field(
        default=0,
        description="Number of images after filtering were filtered out based on the provided filters."
    )
    num_images_crops: int = Field(
        default=0,
        description="Number of images after cropping."
    )
    num_labels_filtered: int = Field(
        default=0,
        description="Number of labels that were filtered out based on the provided filters."
    )
    label_statistics: typing.Optional[dict] = Field(
        default=None,
        description="Statistics about labels in the dataset, such as class distribution, number of labels per"
    )
    cropped_label_statistics: typing.Optional[dict] = Field(
        default=None,
        description="Statistics about labels in the cropped dataset, such as class distribution, number of labels per"
    )
    label_mapping: typing.Optional[dict] = Field(
        default=None,
        description="Label Mapping of class id to class name"
    )
    num_point_labels_filtered: int = Field(
        default=0,
        description="Number of labels that were filtered out based on the provided filters."
    )
    num_boxcenter_labels_filtered: int = Field(
        default=0,
        description="Number of labels that were filtered out based on the provided filters."
    )
    num_labels_crops: int = Field(
        default=0,
        description="Number of labels that were cropped based on the provided crop size."
    )
    bbox_statistics: typing.Optional[dict] = Field(
        default=0,
        description="statistics about bounding boxes in the dataset, such as size distribution, number of boxes per image, etc."
    )


class BasicConfig(BaseModel):
    def save(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path serialization
        config_dict = self.dict()

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


    @classmethod
    def load(cls, file_path: Path) -> 'BasicConfig':
        """Load configuration from JSON file."""
        file_path = Path(file_path)

        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        # Convert string paths back to Path objects
        path_fields = ['base_path', 'reference_base_path', 'images_path', 'output_path']
        for field in path_fields:
            if field in config_dict and config_dict[field] is not None:
                config_dict[field] = Path(config_dict[field])

        return cls(**config_dict)


class DatasetCorrectionConfig(BasicConfig):

    """Configuration model for dataset analysis and processing."""

    # Analysis metadata
    analysis_date: str = Field(..., description="Analysis date in YYYY_MM_DD format")
    type: str = Field(default="points", description="Detection type")

    # Base paths
    subset_base_path: Path = Field(..., description="Base path for training data")
    reference_base_path: Path = Field(..., description="Base path for reference annotations")

    # Annotation files
    hasty_ground_truth_annotation_name: typing.Optional[str] = Field(
        default=...,
        description="Name of the hasty ground truth annotation file for training subset"
    )
    hasty_reference_annotation_name: str = Field(
        ...,
        description="Name of the hasty reference annotation file"
    )
    herdnet_annotation_name: str = Field(
        default=...,
        description="Name of the herdnet annotation file"
    )
    detections_path: Path = Field(
        default=...,
        description="Path to detections of the model"
    )

    # Image processing parameters
    images_path: typing.Optional[Path] = Field(
        None,
        description="Path to images directory (derived from base_path if not provided)"
    )
    suffix: str = Field(default="JPG", description="Image file suffix/extension")

    # Processing parameters
    correct_fp_gt: bool = Field(default=True, description="Whether to correct false positive ground truth")
    box_size: int = Field(default=350, description="Size of bounding boxes")
    radius: int = Field(default=150, description="Radius for point detection")

    # Output configuration
    dataset_name: typing.Optional[str] = Field(
        None,
        description="Name of the dataset (auto-generated if not provided)"
    )
    output_path: typing.Optional[Path] = Field(
        None,
        description="Output path for object crops (derived from base_path if not provided)"
    )
    corrected_path: typing.Optional[Path] = Field(
        None,
        description="Path for all labels"
    )

class GeospatialDatasetCorrectionConfig(BasicConfig):

    """Configuration model for dataset analysis and processing."""
    dataset_name: typing.Optional[str] = Field(
        None,
        description="Name of the dataset"
    )

    type: str = Field(default="points", description="Detection type")

    geojson_prediction_path: Path = Field(
        ...,
        description="Path to the GeoJSON file containing predictions of eiter a model or a human annotator"
    )
    geojson_reference_annotation_path: typing.Optional[Path] = Field(
        None,
        description="Path to the GeoJSON file containing reference annotations from at least one Person"
    )

    image_path: Path = Field(
        default=...,
        description="Path to the image file (orthomosaic or single image) to be processed"
    )
    image_tiles_path: typing.Optional[Path] = Field(
        default=None,
        description="Path to the image tiles directory (if applicable)"
    )
    hasty_reference_annotation_path: Path = Field(
        ...,
        description="path of the hasty reference annotation file"
    )
    hasty_intermediate_annotation_path: typing.Optional[Path] = Field(
        None,
        description="path of the hasty reference annotation file"
    )
    box_size_x: int = Field(default=640, description="Size of a tile in x direction")
    box_size_y: int = Field(default=640, description="Size of a tile in y direction")

    # Processing parameters
    correct_fp_gt: bool = Field(default=True, description="Whether to correct false positive ground truth")
    box_size: int = Field(default=350, description="Size of bounding boxes")
    radius: int = Field(default=150, description="Radius for point detection")

    # Output configuration

    output_path: typing.Optional[Path] = Field(
        None,
        description="Output path for object crops (derived from base_path if not provided)"
    )
    corrected_path: typing.Optional[Path] = Field(
        None,
        description="Path for all labels"
    )

class GeospatialDatasetCorrectionConfigCollection(BasicConfig):

    """Configuration model for dataset analysis and processing."""
    dataset_name: typing.Optional[str] = Field(
        None,
        description="Name of the dataset"
    )
    organization: str = Field(
        ...,
        description="Organization name in CVAT"
    )
    project_name: str = Field(
        ...,
        description="Project name in CVAT"
    )

    configs: typing.List[GeospatialDatasetCorrectionConfig] = Field(..., description="List of geospatial dataset correction configurations")


    output_path: typing.Optional[Path] = Field(
        None,
        description="Output path for object crops (derived from base_path if not provided)"
    )
    corrected_path: typing.Optional[Path] = Field(
        None,
        description="Path for all labels"
    )

class DatasetCorrectionReportConfig(DatasetCorrectionConfig):
    report_path: typing.Optional[Path] = Field(
        None,
        description="Report Path"
    )
    hA_prediction_path: typing.Optional[Path] = Field(
    None,
    description="Hasty Annotation Path full size predictions"
    )
    hA_prediction_tiled_path: typing.Optional[Path] = Field(
    None,
    description="Hasty Annotation Path for tiled predictions"
    )



