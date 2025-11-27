from pathlib import Path

from dataclasses import dataclass
@dataclass(frozen=True)
class YOLOAnnotationLabel(object):
    """
    refers to a single label file. Those can contain multiple boxes.

    """
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    image_path: Path






