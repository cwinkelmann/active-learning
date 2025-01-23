import hashlib
import typing

import numpy as np
from pathlib import Path

from PIL.Image import Image


def get_image_id(filename: Path = None, image: np.ndarray = None):
    """
    @param filename:
    @return:
    """
    assert (filename is not None) != (image is not None), "Either filename or image must be provided"

    if filename is not None:
        with open(filename, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()

        return readable_hash
    if image is not None:
        image_bytes = image.tobytes()
        readable_hash = hashlib.sha256(image_bytes).hexdigest()

        return readable_hash


def get_image_dimensions(image_path: Path) -> typing.Tuple[int, int]:
    """
    Get the dimensions of an image
    :param imag_path:
    :return:
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height
