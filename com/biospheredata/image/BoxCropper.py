"""
Function to modify with bounding boxes and images
# moved to active learning repo
"""
import typing

import cv2
import random

import pandas as pd
import shapely
import shapely.affinity
from matplotlib import pyplot as plt, patches
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger
from shapely.geometry import Polygon
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import array as ar
import os

from PIL import Image
import PIL

from com.biospheredata.converter.Annotation import reframe_bounding_box, reframe_polygon
from com.biospheredata.types.annotationbox import BboxXYXY
from com.biospheredata.visualization.visualize_result import visualise_image, visualise_polygons


def crop_image_from_box_normalized(output_path: Path,
                                   input_image_path,
                                   normalized_x_center,
                                   normalized_y_center,
                                   normalized_height,
                                   normalized_width,
                                   absolute_size):
    """
    crop boxes out of an image using normalized coordindates

    :param output_path:
    :param input:
    :param normalized_x_center:
    :param normalized_y_center:
    :param normalized_height:
    :param normalized_width:
    :param absolute_size:
    :return:
    """
    PIL.Image.MAX_IMAGE_PIXELS = 509083780

    im = Image.open(input_image_path)
    imgwidth, imgheight = im.size
    absolute_x_center = normalized_x_center * imgwidth
    abs_width = normalized_width * imgwidth
    # abs_width = absolute_size

    absolute_y_center = normalized_y_center * imgheight
    abs_height = normalized_height * imgheight
    # abs_height = absolute_size

    box = (absolute_x_center, absolute_y_center, absolute_x_center + abs_width, absolute_y_center + abs_height)

    left = absolute_x_center - int(abs_width / 2)
    top = absolute_y_center - int(abs_height / 2)
    right = absolute_x_center + int(abs_width / 2)
    bottom = absolute_y_center + int(abs_height / 2)
    # TODO all boxes would have different sizes now
    box = (left, top, right, bottom)

    # im1 = im.crop((left, top, right, bottom))
    im1 = im.crop(box)
    # im1.show()
    im1 = im1.convert("RGB")
    image_name = Path(input_image_path).name
    box_name = f"{image_name}_{absolute_x_center}_{absolute_y_center}.jpg"
    im1.save(output_path.joinpath(box_name))

    return output_path.joinpath(box_name)


def crop_image_from_box_absolute(output_path: Path,
                                 input_image_path: Path,
                                 box: list[int, int, int, int]):
    """
    crop boxes out of an image using absolute coordindate
    :param output_path:
    :param input:
    :param absolute_size:
    :return:
    """
    # in case we are cutting an orthomosaic
    PIL.Image.MAX_IMAGE_PIXELS = 509083780

    im = Image.open(input_image_path)
    im1 = im.crop(box)
    # im1.show()
    im1 = im1.convert("RGB")
    image_name = Path(input_image_path).stem
    box_name = f"{image_name}_crop.jpg"
    im1.save(output_path.joinpath(box_name))

    return im1, output_path.joinpath(box_name)


def crop_image_polygon(
                       bbox: typing.Union[Polygon, list[int, int, int, int], BboxXYXY],
                        input_image_path: Path = None,
                       image: Image = None,
                       output_path: Path = None,
                       angle: float = 0.0,
                       ) -> PIL.Image:
    """
    TODO fix the rotation, it looks fishy
    crop boxes out of an image using a polygon
    :param output_path:
    :param input:
    :param polygon:
    :return:
    """
    PIL.Image.MAX_IMAGE_PIXELS = 509083780
    if input_image_path is not None:
        im = Image.open(input_image_path)
    else:
        im = image
    imr = np.array(im, dtype=np.uint8)
    cx, cy, ch = imr.shape
    center = (cx // 2, cy // 2)
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(imr, M, (imr.shape[1], imr.shape[0]))

    # rotate image and polygon
    bbox_obb = shapely.affinity.rotate(bbox, -angle, origin=(cx, cy), use_radians=False)

    ## TODO rotate the annotations too
    rotated = Image.fromarray(rotated)
    im1 = rotated.crop(bbox_obb.bounds)
    # im1.show()
    im1 = im1.convert("RGB")

    if output_path:
        image_name = Path(input_image_path).stem
        box_name = f"{image_name}_crop.jpg"
        im1.save(output_path.joinpath(box_name))
        return im1, output_path.joinpath(box_name)

    return im1



def crop_image_polygon_v2(
                       bbox: typing.Union[Polygon, list[int, int, int, int], BboxXYXY],
                       image: PIL.Image = None,
                       angle: float = 0.0,
                       ) -> PIL.Image:
    """
    crop and rotate a polygon out of an image and rotate it.
    Imagine an oriented bounding box which we want to rotate to have the long edge on X axis

    :param bbox:
    :param output_path:
    :param input:
    :param polygon:
    :return:
    """

    if isinstance(bbox, BboxXYXY):
        bbox = bbox.to_list()
        bbox_polygon = shapely.box(*bbox)
    if isinstance(bbox, Polygon):
        bbox_polygon = bbox
    if isinstance(bbox, list):
        bbox_polygon = shapely.box(*bbox)

    ## keep in mind the polygon is already rotated
    xmin, ymin, xmax, ymax = bbox_polygon.bounds
    width = xmax - xmin
    height = ymax - ymin

    # ax = visualise_image(image=image, show=False, title="original image")
    # ax = visualise_polygons([bbox_polygon],
    #                         filename=None,
    #                         show=True, title="original image", color="r", ax=ax)

    bounds_polygon = shapely.box(*bbox.bounds)
    # crop the smallest possible image which covers the polygon completely
    cropped = image.crop(bbox.bounds)
    i_width, i_height = cropped.size

    rotated_cropped_image = cropped.rotate(angle,
                                expand=False, #fillcolor=(255, 255, 255),
                                center=(i_width // 2, i_height // 2))



    assert bbox_polygon.within(bounds_polygon), "The bounding box must be within the bounds of the polygon"
    bbox_polygon_local = reframe_polygon(cutout_box=bounds_polygon, label=bbox) # looks weird but is intentional. The OBB bounds are the reference for the new image

    # ax = visualise_image(image=cropped, show=False, title="original image")
    # ax = visualise_polygons([bbox_polygon_local],
    #                         filename=None,
    #                         show=True, title="All Annotations", color="r", ax=ax)

    # TODO center = (cx // 2, cy // 2) == (bbox_polygon_local.centroid.x, bbox_polygon_local.centroid.y) ????


    bbox_polygon_local = shapely.affinity.rotate(bbox_polygon_local, -angle, origin=(i_width // 2, i_height // 2), use_radians=False)

    # ax = visualise_image(image=rotated_cropped_image, show=False, title="original image")
    # ax = visualise_polygons([bbox_polygon_local],
    #                         filename=None,
    #                         show=True, title="All Annotations", color="r", ax=ax)

    im1 = rotated_cropped_image.crop(bbox_polygon_local.bounds)

    # ax = visualise_image(image=im1, show=True, title="Cropped image")
    im1 = im1.convert("RGB")


    return im1