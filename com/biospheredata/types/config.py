from dataclasses import dataclass
from pathlib import Path


@dataclass()
class PresentationConfig:
    image_folder: Path
    output_folder: Path
    hasty_annotation_file_name: str
