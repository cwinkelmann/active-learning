import typing

import copy

import random

import abc
from enum import Enum
import random
import json
from loguru import logger
from typing import Dict
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

class SampleStrategy(Enum):
    FIRST = 1
    RANDOM = 2
    PERCENTAGE = 3


class ImageFilter():
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, hA: HastyAnnotationV2):
        return hA


T = typing.TypeVar('T')
def sample_hasty(l: typing.List[T],
                 n: int = None,
                percentage: float = None,
                method: SampleStrategy = SampleStrategy.FIRST) -> typing.List[T]:
    """
    Sample the Hasty format to get a smaller dataset.
    :param hA:
    :param n:
    :param percentage:
    :param method:
    :return:
    """
    assert isinstance(method, SampleStrategy), "Method must be a SampleStrategy"
    assert percentage is None or (0 < percentage <= 1), "Percentage must be between 0 and 1"

    if method == SampleStrategy.PERCENTAGE:
        # Calculate the number of elements to sample
        n = int(len(l) * percentage)
        sampled_list = random.sample(l, n)

    elif method == SampleStrategy.RANDOM:
        if n is None:
            raise ValueError("n must be specified for the 'random' method")
        n = min(n, len(l))  # Ensure n doesn't exceed the number of l
        sampled_list = random.sample(l, n)

    elif method == SampleStrategy.FIRST:
        if n is None:
            raise ValueError("n must be specified for the 'first' method")
        n = min(n, len(l))  # Ensure n doesn't exceed the number of l
        sampled_list = l[:n]

    else:
        raise ValueError(f"Invalid method: {method}")

    assert type(l[0]) is type(sampled_list[0])
    return sampled_list


def sample_coco(coco_annotations_format: Dict, n: int = None,
                percentage: float = None,
                method: SampleStrategy = SampleStrategy.FIRST) -> Dict:
    """
    Sample the COCO format to get a smaller dataset.

    :param coco_annotations_format: The COCO annotations dataset as a dictionary.
    :param n: Number of elements to sample. Ignored if percentage is specified.
    :param percentage: The percentage of elements to randomly sample (0 < percentage <= 1).
    :param method: Sampling method - "first", "random", or "percentage".
    :return: A smaller dataset in COCO format.
    """
    assert isinstance(method, SampleStrategy), "Method must be a SampleStrategy"
    assert percentage is None or (0 < percentage <= 1), "Percentage must be between 0 and 1"

    # Extract images and annotations
    images = coco_annotations_format.get("images", [])
    annotations = coco_annotations_format.get("annotations", [])

    if method == SampleStrategy.PERCENTAGE:
        # Calculate the number of elements to sample
        n = int(len(images) * percentage)
        sampled_images = random.sample(images, n)

    elif method == SampleStrategy.RANDOM:
        if n is None:
            raise ValueError("n must be specified for the 'random' method")
        n = min(n, len(images))  # Ensure n doesn't exceed the number of images
        sampled_images = random.sample(images, n)

    elif method == SampleStrategy.FIRST:
        if n is None:
            raise ValueError("n must be specified for the 'first' method")
        n = min(n, len(images))  # Ensure n doesn't exceed the number of images
        sampled_images = images[:n]

    else:
        raise ValueError(f"Invalid method: {method}")

    # Get image IDs from sampled images
    sampled_image_ids = {image["id"] for image in sampled_images}

    # Filter annotations for sampled images
    sampled_annotations = [ann for ann in annotations if ann["image_id"] in sampled_image_ids]

    # Create a new COCO dataset with the sampled data
    sampled_coco = {
        "info": coco_annotations_format.get("info", {}),
        "licenses": coco_annotations_format.get("licenses", []),
        "categories": coco_annotations_format.get("categories", []),
        "images": sampled_images,
        "annotations": sampled_annotations
    }

    return sampled_coco


class ImageFilterConstantNum(ImageFilter):
    """ Get N images Randomly or First N images

    """


    def __init__(self, num: int, sample_strategy: SampleStrategy = SampleStrategy.RANDOM, seed: int = 42):
        super().__init__()
        self.num = num
        self.sample_strategy = sample_strategy

        random.seed(seed)

    def __call__(self, hA: HastyAnnotationV2):
        """ get N images randomly
        :return:
        """
        hA = super().__call__(hA)
        if self.num is None:
            return hA


        if self.num > len(hA.images):
            raise ValueError(f"Number of images is {len(hA.images)} but you requested {self.num}")

        sample = sample_hasty(hA.images, n=self.num, method=self.sample_strategy)

        hA_filtered = copy.deepcopy(hA)

        hA_filtered.images = sample
        logger.info(f"Number of images after filtering: {len(hA_filtered.images)}")
        return hA_filtered