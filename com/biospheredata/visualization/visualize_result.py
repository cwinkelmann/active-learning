import typing
from loguru import logger

from time import sleep

import shapely
from typing import List

import PIL
import copy

import uuid
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import matplotlib.axis as axis
import matplotlib.patches as patches
import matplotlib.axes as axes
from shapely import Polygon
from PIL import Image as PILImage

from active_learning.util.visualisation.annotation_vis import visualise_points_only

PILImage.Image.MAX_IMAGE_PIXELS = 5223651122
from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, HastyAnnotationV2, ImageLabelCollection
from com.biospheredata.types.annotationbox import Annotation
from contextlib import contextmanager


def visualise_polygons(polygons: List[shapely.Polygon] = (),
                       points: List[shapely.Point] = (),
                       filename=None, show=False, title=None,
                       max_x=None, max_y=None, color="blue",
                       ax: axes.Axes = None, linewidth=0.5, markersize=0.5, fontsize=22,
                       labels: List[str] = None, label_position="next_to") -> axes.Axes:
    """
    Visualize a list of polygons
    :param labels:
    :param fontsize:
    :param markersize:
    :param linewidth:
    :param ax:
    :param color:
    :param max_y:
    :param max_x:
    :param title:
    :param show:
    :param filename:
    :param points:
    :param polygons:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    assert isinstance(ax, axes.Axes), f"Expected matplotlib.axes.Axes, got {type(ax)}"

    if max_x:
        plt.xlim(0, max_x)
    if max_y:
        plt.ylim(0, max_y)
    if title:
        plt.title(title)
    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)


        # Add label for each polygon if labels are provided
        if labels and i < len(labels):
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            centroid = polygon.centroid
            if label_position == "center":
                label_x, label_y = centroid.x, centroid.y
                ha, va = 'center', 'center'
            elif label_position == "next_to":
                # Place label to the right of the polygon
                label_x = bounds[2] + (bounds[2] - bounds[0]) * 0.05  # 5% of width to the right
                label_y = centroid.y
                ha, va = 'left', 'center'
            else:
                raise ValueError(f"Unknown label position: {label_position}. Use 'center' or 'next_to'.")

            # Get the centroid of the polygon for labeling
            centroid = polygon.centroid
            # ax.text(centroid.x, centroid.y, labels[i], fontsize=fontsize, ha='center', color='red')
            ax.text(label_x, label_y, labels[i], fontsize=fontsize,
                    ha=ha, va=va, color='red')

    for point in points:
        x, y = point.xy
        ax.plot(x, y, marker='o', color=color, linewidth=linewidth, markersize=markersize)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
        plt.close()

    return ax


def visualize_bounding_boxes(imname,
                             label_name,
                             basepath,
                             output_path=None,
                             suffix_images=None,
                             suffix_labels=None,
                             imarray=None):
    """
    loads annotation and displays them as bounding boxes

    :param output:
    :param imname:
    :param basepath:
    :return:
    """
    if imarray is not None:
        im = imarray
    else:
        # im = Image.open(basepath.joinpath(f'{imname}.JPG'))
        if suffix_images:
            im = Image.open(basepath.joinpath(suffix_images).joinpath(imname))
        else:
            im = Image.open(basepath.joinpath(imname))


    # df = pd.read_csv(basepath.joinpath(f'{imname}.txt'), sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    if suffix_labels:

        df = pd.read_csv(basepath.joinpath(suffix_labels).joinpath(label_name), sep=' ',
                         names=['class', 'x1', 'y1', 'w', 'h'])
    else:
        df = pd.read_csv(basepath.joinpath(label_name), sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    imr = np.array(im, dtype=np.uint8)

    df_scaled = copy.deepcopy(df.iloc[:, 1:]) # TODO this is not good style
    df_scaled[['x1', 'w']] = df_scaled[['x1', 'w']] * imr.shape[1]
    df_scaled[['y1', 'h']] = df_scaled[['y1', 'h']] * imr.shape[0]

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    # Display the image
    ax.imshow(imr)
    for box in df_scaled.values:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0] - (box[2] / 2), box[1] - (box[3] / 2)), box[2], box[3],
                                 linewidth=2, edgecolor='r',
                                 facecolor='none')
        # Add the patch to the axes
        ax.add_patch(rect)

    #    if output:

    #        plt.savefig(basepath.joinpath(f'{imname}_debug_{random_num}.JPG'))
    plt.axis('off')
    if output_path:
        output_path = Path(output_path)
        Path.mkdir(output_path, exist_ok=True, parents=True)
        plt.savefig(output_path.joinpath(f'{imname}'), bbox_inches='tight')
        # plt.show()
    plt.close()
    return str(output_path.joinpath(f'{imname}'))


def visualize_bounding_box_v2(imname,
                              base_path: Path,
                              labels: list[Annotation],
                              output_file_name=None,
                              show=False,
                              output_path=None):
    """
    Visualize Boxes in
    labels is a list of Annotation. Each list depics the bounding box and the ID
    """
    if output_file_name is None:
        output_file_name = imname

    im = Image.open(base_path.joinpath(imname))
    imr = np.array(im, dtype=np.uint8)
    fig, ax = plt.subplots(1, figsize=(25, 25))
    ax.imshow(imr)
    for label in labels:
        # Create a Rectangle patch
        box = label.bbox
        box = Polygon([(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)])
        rect = patches.Polygon(box.exterior.coords,
                                 linewidth=2, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)
        # obviously use a different formula for different shapes
        plt.text(list(box.exterior.coords)[4][0],
                 list(box.exterior.coords)[4][1] - 5,
                 label.class_id, color="red")

    if output_path is not None:
        Path.mkdir(output_path, exist_ok=True, parents=True)
        plt.savefig(output_path.joinpath(f'{output_file_name}'))
    if show:
        plt.show()
    plt.close()

    return output_file_name

@contextmanager
def large_image_context():
    """Context manager to temporarily allow large images"""
    original_limit = PILImage.MAX_IMAGE_PIXELS
    # logger.info(f"Reset PILImage.MAX_IMAGE_PIXELS from {original_limit} to None to allow large images")
    try:

        PILImage.MAX_IMAGE_PIXELS = None  # Remove limit
        yield
    finally:
        PILImage.MAX_IMAGE_PIXELS = original_limit  # Restore original limit

def visualise_image(image_path: Path = None,
                    image: typing.Union[PILImage, np.ndarray] = None,
                    output_file_name: Path = None,
                    show: bool = False,
                    title: str | None = None,
                    ax: axes.Axes = None, figsize=(20, 15), dpi=150) -> axis:
    """
    :param image:
    :param output_file_name:
    :param show:
    :param figsize:
    :param ax:
    :param title:
    :param image_path:
    :return:
    """

    if image is not None and isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)  # TODO use the shape of imr to get the right ration
    if image_path is not None:

        with large_image_context():
            image = PILImage.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

    imr = np.array(image, dtype=np.uint8)
    ax.imshow(imr)

    if title:
        ax.set_title(title)

    if output_file_name is not None:
        plt.savefig(output_file_name)

    if show:
        plt.show()
        return ax
    else:
        return ax



def blur_bounding_boxes(image_path: Path, labels: [ImageLabel]):
    """
    Blurs/black the bounding boxes in the edge of the image to prevent partial objects ruining the training
    :param imname:
    :param basepath:
    :return:
    """
    im = Image.open(image_path)
    # Extract the region to be blurred
    for bbox in labels:
        assert isinstance(bbox, ImageLabel)
        blackout_bbox(im, bbox.x1y1x2y2)

        im.save(image_path)

def blackout_bbox(im: PIL.Image, bbox_x1y1x2y2: List[int]) -> PIL.Image:
    """

    :param im:
    :param bbox_x1y1x2y2:
    :return:
    """
    region = im.crop(bbox_x1y1x2y2)

    # Apply Gaussian blur to the region
    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=5))  # You can adjust the radius as needed
    black_region = Image.new('RGB', (bbox_x1y1x2y2[2] - bbox_x1y1x2y2[0], bbox_x1y1x2y2[3] - bbox_x1y1x2y2[1]), 'black')

    # Paste the blurred region back onto the original image
    im.paste(black_region, bbox_x1y1x2y2)

    return im

def visualise_hasty_annotations(hA: HastyAnnotationV2, images_path: Path, output_path: Path):
    """

    :param hA_train:
    :param output_path_train:
    :return:
    """

    for image in hA.images:
        labels = image.labels
        if labels is None or len(labels) == 0:
            logger.warning(f"Image {image.image_name} has no labels")
        else:
            image_name = image.image_name
            i_width, i_height = image.width, image.height

            ax_ig = visualise_image(image_path=images_path / image_name, show=False, title="original image", dpi=300)

            # segmentation masks
            ax = visualise_polygons([il.polygon_s for il in labels if il.polygon is not None],
                                    filename=None,
                                    show=False, max_x=i_width, max_y=i_height, color="r", ax=ax_ig)

            # bounding boxes
            ax = visualise_polygons([il.bbox_polygon for il in labels if il.bbox_polygon is not None],
                                    filename=None,
                                    show=False, title="All Annotations", max_x=i_width, max_y=i_height, color="white",
                                    ax=ax)

            # dots
            ax = visualise_polygons(points=[il.incenter_centroid for il in labels if il.incenter_centroid is not None], max_x=i_width, max_y=i_height, color="green",
                                    ax=ax, show=False, filename=output_path / f"{image_name}_all_annotations.jpg")


def visualise_hasty_annotation(image: ImageLabelCollection, images_path: Path,
                               output_path: Path | None = None, show: bool = False, title: str | None = None,
                               figsize=(5,5), dpi=150, show_axis=True) -> axes.Axes:
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    labels = image.labels
    if labels is None or len(labels) == 0:
        logger.warning(f"Image {image.image_name} has no labels")
    else:
        image_name = image.image_name
        i_width, i_height = image.width, image.height

        ax_ig = visualise_image(image_path=images_path / image_name, show=False, title=title,
                                figsize=figsize, dpi=dpi)
        if not show_axis:
            ax_ig.set_xticks([])  # Remove x ticks
            ax_ig.set_yticks([])  # Remove y ticks
            ax_ig.set_xlabel('')  # Remove x label
            ax_ig.set_ylabel('')
        # segmentation masks
        ax = visualise_polygons([il.polygon_s for il in labels if il.polygon is not None],
                                filename=None,
                                show=False, max_x=i_width, max_y=i_height, color="r", ax=ax_ig)

        # bounding boxes
        ax = visualise_polygons([il.bbox_polygon for il in labels if il.bbox_polygon is not None],
                                filename=None,
                                show=False,
                                title=title,
                                max_x=i_width,
                                max_y=i_height,
                                color="white",
                                ax=ax)

        # dots
        # When output path is already a full path,
        filename = output_path
        ax = visualise_points_only(points=[il.incenter_centroid for il in labels if il.incenter_centroid is not None],
                                   ax=ax, show=show,
                                   labels=[
                                       il.class_name if il.class_name is not None else "undefined"
                                       for il in labels
                                       if il.incenter_centroid is not None  # Match the points filter!
                                   ],
                                   markersize=4,
                                   font_size=7,
                                   filename=output_path / f"{image_name}_all_annotations.jpg",
                                   title=title)
        if not show_axis:
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks
            ax.set_xlabel('')  # Remove x label
            ax.set_ylabel('')

        if title is None:
            ax.set_title(None)

        plt.tight_layout()

        return ax
