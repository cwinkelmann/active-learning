import PIL.Image
import hashlib
import typing

import numpy as np
from pathlib import Path

from PIL.Image import Image
import io


def get_image_id(filename: Path = None, image: typing.Union[np.ndarray, PIL.Image.Image] = None):
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

    if image is not None and isinstance(image, np.ndarray):
        image_bytes = image.tobytes()
        readable_hash = hashlib.sha256(image_bytes).hexdigest()
        return readable_hash

    elif image is not None and isinstance(image, PIL.Image.Image):
        with io.BytesIO() as byte_arr:
            image.save(byte_arr, format='PNG')  # Consistent format
            image_bytes = byte_arr.getvalue()
            readable_hash = hashlib.sha256(image_bytes).hexdigest()

        return readable_hash


def get_image_dimensions(image_path: Path) -> typing.Tuple[int, int]:
    """
    Get the dimensions of an image
    :param imag_path:
    :return:
    """
    with PIL.Image.open(image_path) as img:
        width, height = img.size
    return width, height
