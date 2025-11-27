"""
Dataset for Image Classifiction Based on the number of objects in the image, when zero objects exist the image is classified as empty
if there 1 or more it is classified as not empty
"""
import copy

import numpy as np
import random

import pandas as pd
import torch
from pathlib import Path
from matplotlib import pyplot as plt

import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from PIL import Image
from loguru import logger

from com.biospheredata.image.augmentations import crop_augment_crop, get_image_by_id, \
    get_by_id, LabelMissingException, crop_augment_empty
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, \
    convert_HastyAnnotationV2_to_HastyAnnotationV2flat, HastyAnnotationV2_flat, hA_from_file
from com.biospheredata.ml.datasets.helper import imshow



"""
TODO write a Loader which helps to load the data of images / iguana count pairs
"""