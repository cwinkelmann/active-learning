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

from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, HastyAnnotationV2
from com.biospheredata.types.annotationbox import Annotation

def visualise_polygons(polygons: List[shapely.Polygon] = (), points: List[shapely.Point] = (),
                       filename=None, show=False, title = None,
                       max_x=None, max_y=None, color="blue", ax:axes.Axes =None) -> axes.Axes:
    """
    Visualize a list of polygons
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
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=0.5)
    for point in points:
        x, y = point.xy
        ax.plot(x, y, marker='o', color=color, linewidth=0.5, markersize=0.5)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

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


def visualise_image(image_path: Path = None,
                    image: Image = None,
                    output_file_name: Path = None,
                    show: bool = False,
                    title: str = "original image",
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


    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)  # TODO use the shape of imr to get the right ration
    if image_path is not None:
        image = Image.open(image_path)
    imr = np.array(image, dtype=np.uint8)
    ax.imshow(imr)
    ax.set_title(title)

    if output_file_name is not None:
        plt.savefig(output_file_name)

    if show:
        plt.show()
        # sleep(0.1)
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

    FIXME: This is not working yet because the visualise points of course needs the "iguana_point" class to be present but the polygon and the box are in the "iguana" class
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

            pass

if __name__ == '__main__':
    yolo_base_path = Path("/tmp/train/images")

    visualize_bounding_boxes(imname="ESCG02-2_44",
                             basepath=yolo_base_path,
                             output=True)  ## TODO implement this properly
