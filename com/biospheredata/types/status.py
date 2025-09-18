from enum import Enum


class LabelingStatus(Enum):
    """
    Enumeration for labeling status of images.
    """
    NEW = "NEW"
    IN_PROGRESS = "IN PROGRESS"
    DONE = "DONE"
    TO_REVIEW = "TO REVIEW"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"


class AnnotationType(Enum):
    """
    Enumeration for different types of annotations.
    """
    BOUNDING_BOX = "box"
    KEYPOINT = "keypoint"
    POLYGON = "polygon"


class ImageFormat(Enum):
    JPG = "jpg"
    PNG = "png"
    JP2 = "jp2"
    GeoTIFF = "tif"


class ClassName(Enum):
    iguana = "iguana"
    iguana_point = "iguana_point"
    crab = "crab"
    turtle = "turtle"
    seal = "seal"
    hard_negative = "hard_negative"
