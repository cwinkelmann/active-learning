"""
Abstraction of image augmentations

TODO there is this file too: object-detection-pytorch/com/biospheredata/helper/image_annotation/augmentation.py


"""
import copy

from time import sleep
import PIL
import cv2
from pathlib import Path

import shapely
import typing
import albumentations as A
import uuid
import albumentations.augmentations.functional as F
from PIL.Image import Image as PILImage
from PIL import Image
from shapely import affinity
from typing import Any, Tuple
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
import random

import numpy as np
from albumentations import DualTransform
from albumentations.core.types import Targets, BoxInternalType

from loguru import logger

from com.biospheredata.converter.Annotation import add_offset_centroid, reframe_bounding_box, reframe_polygon
from com.biospheredata.image.BoxCropper import crop_image_polygon, crop_image_polygon_v2
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, ImageLabel, Image as HastyImage
from com.biospheredata.visualization.visualize_result import visualise_image, visualise_polygons, blackout_bbox, \
    blur_bounding_boxes


def custom_transforms(image: typing.Union[np.ndarray, PILImage],
                      labels: typing.List[ImageLabel],
                      albumentations_transforms=None,
                      blur_edge_labels=False) -> typing.Tuple[PIL.Image.Image, typing.List[ImageLabel]]:
    """
    Custom transformation for the labels
    :param labels:
    :return:
    """
    if isinstance(image, PILImage):
        image = np.array(image)
    else:
        assert isinstance(image, np.ndarray), f"Expected PILImage or np.ndarray, got {type(image)}"


    bboxes = [l.bbox for l in labels]
    # get with and height
    label_ids = [l.id for l in labels]
    # bboxes = [x + [y.id] for x, y in zip(boxes, labels)]

    if albumentations_transforms:
        ## Augment and cut the images
        transform = albumentations_transforms

        transformed = transform(image=image, bboxes=bboxes, category_ids=label_ids)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        transformed_image = transformed['image']
        transformed_bbox = transformed['bboxes']
        transformed_labels = transformed['category_ids']

        ## TODO implment the empty / prob mechanism later
        # if len(transformed['bboxes']) > 0 or True:
        transformed_image_pil = PIL.Image.fromarray(transformed_image)

        # Convert to string if you need a string representation
        new_labels = []
        for i, t in enumerate(transformed_bbox):

            id = transformed_labels[i]
            reframed_selected_image_label = get_by_id(id, labels)
            if id != reframed_selected_image_label.id:
                raise ValueError("The id of the label is not the same as the id of the transformed label")
            il = ImageLabel(id=id,
                            class_name=reframed_selected_image_label.class_name,
                            bbox=[int(x) for x in t[0:4]]
                            )

            new_labels.append(il)

        return transformed_image_pil, new_labels
    else:
        return image, labels

class LabelMissingException(Exception):
    """
    when there is no label for an image anymore
    """

class TranslationWithinBoundsNotPossible(Exception):
    """
    when there is no label for an image anymore
    """
    pass


def random_translation_within_bounds(c_box: shapely.Polygon,
                                     image_cutout_box: shapely.Polygon,
                                     a_bbox: shapely.Polygon):
    """

    :param c_box:
    :param a_bbox:
    :return:
    """
    # move the box left or right / up or down but the annotation stays within the image
    # left maximum by the half of the NEW image width AND the current centroid - so not outside the image crop

    minx, miny, maxx, maxy = c_box.bounds
    abox_minx, abox_miny, abox_maxx, abox_maxy = a_bbox.bounds
    ### X Axis

    left_translation_max = (-1 *
                            min(minx,  # not beyond the containing cutout box left boarder
                                maxx - abox_maxx))  # don't move the object outside

    right_translation_max = (1 *
                             min(abox_minx - minx,  # distance of annotation to the border
                                 image_cutout_box.bounds[2] - maxx))

    if left_translation_max < right_translation_max and right_translation_max > 0:
        translation_x = random.randint(
            int(left_translation_max),  # to left,
            int(right_translation_max)
        )
    else:
        translation_x = 0
        logger.warning(f"No translation in x possible, left_translation_max: {left_translation_max}, right_translation_max: {right_translation_max}")
        raise TranslationWithinBoundsNotPossible("No translation in x possible")

    ### Y Axis
    # y = 0 is on the top of the image, y = max is on the bottom of the image
    up_translation_max = -1 * min(
        miny, # Distance of final cutout box to containing cutout box
        maxy - abox_maxy # Distance of annotation to the top border of the cutout box
    )
    down_translation_max = 1 * min(
        image_cutout_box.bounds[3] - maxy,  # Distance of final cutout box to containing cutout box
        abox_miny - miny, # Distance of box to the bottom border of the cutout box,
    )

    if up_translation_max < down_translation_max and down_translation_max > 0:
        translation_y = random.randint(
            int(up_translation_max),
            int(down_translation_max)
        )
    else:
        translation_y = 0
        logger.warning(f"No translation in y possible, up_translation_max: {up_translation_max}, down_translation_max: {down_translation_max}")
        raise TranslationWithinBoundsNotPossible("No translation in y possible")

    final_cutout_box_t = affinity.translate(c_box, translation_x, translation_y)
    assert a_bbox.within(final_cutout_box_t), f"{a_bbox} not within {final_cutout_box_t.bounds}"
    a_bbox = reframe_bounding_box(cutout_box=final_cutout_box_t, label=a_bbox)

    return final_cutout_box_t, a_bbox


def centered_fitted_crop_around_bbox(
        #image: PIL.Image,
                                     image_polygon: shapely.Polygon,
                                     bbox_global: shapely.Polygon,
                                     offset: int) -> Tuple[shapely.Polygon, shapely.Polygon, shapely.Polygon]:
    """
    return a crop which contains the bounding box somewhere within the crop

    :param image: PIL image
    :param bbox_local: bounding box in format [xmin, ymin, xmax, ymax]
    :param width: width of new image
    :param height: height of new image

    :return: augmented image and labels
    """
    xmin, ymin, xmax, ymax = image_polygon.bounds
    image_width = xmax - xmin
    image_height = ymax - ymin
    # image_height, image_width = image.size
    # construct the cutout box around the bounding box


    cutout_box_global = add_offset_centroid(bbox_global.centroid,
                                            image_height=image_height,
                                            image_width=image_width,
                                            offset=offset)

    # reproject the coodinates of the bounding box
    bbox_local = reframe_bounding_box(cutout_box=cutout_box_global, label=bbox_global)

    cutout_box_local = affinity.translate(cutout_box_global, -cutout_box_global.bounds[0], -cutout_box_global.bounds[1])

    # assert bbox_local.centroid.within(cutout_box_global), f"Centroid {bbox_local.centroid} not within {cutout_box_global}"
    return cutout_box_global, cutout_box_local, bbox_local


class RandomCropAugmentation:
    def __init__(self, crop_height, crop_width):
        """
        Args:
            crop_height (int): Height of the crop.
            crop_width (int): Width of the crop.
        """
        self.transform = A.RandomCrop(height=crop_height, width=crop_width, always_apply=True)

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Apply the transform
        augmented = self.transform(image=img_np)
        img_np = augmented['image']

        # Convert numpy array back to PIL image
        return Image.fromarray(img_np)


def random_crop_around_bbox_v2(
        image_cutout_box: shapely.Polygon,
        annotation_bbox: typing.Union[shapely.Polygon, ImageLabel],
        new_x_width: int,
        new_y_height: int) -> Tuple[shapely.Polygon, shapely.Polygon, shapely.Polygon]:

    """ shift the image randomly as long as the annotation_bbox-centroid stays within the image
    :param image:
    :param annotation_bbox:
    :param offset:
    :return:
    """

    if isinstance(annotation_bbox, ImageLabel):
        annotation_bbox = annotation_bbox.bbox_polygon

    # translate the cutout box to the new image
    minx, miny, maxx, maxy = image_cutout_box.bounds

    # Calculate width and height
    image_width = maxx - minx
    image_height = maxy - miny
    # cut the image to a smaller box around the annotation
    cutout_bbox_local = add_offset_centroid(annotation_bbox.centroid,
                                     image_height=image_height,
                                     image_width=image_width,
                                     offset=new_y_height // 2)

    annotation_bbox_local = reframe_bounding_box(cutout_box=cutout_bbox_local, label=annotation_bbox)

    # move annotation_bbox to the new image
    cutout_box_global, annotation_bbox_local = random_translation_within_bounds(c_box=cutout_bbox_local,
                                                                                 image_cutout_box=image_cutout_box,
                                                                                 a_bbox=annotation_bbox)

    cutout_box_local = affinity.translate(cutout_box_global, -cutout_box_global.bounds[0], -cutout_box_global.bounds[1])

    return cutout_box_global, cutout_box_local, annotation_bbox_local


def random_crop_around_bbox_full(
        image_cutout_box: shapely.Polygon,
        annotation_bbox: typing.Union[shapely.Polygon, ImageLabel],
        new_x_width: int,
        new_y_height: int) -> Tuple[shapely.Polygon, shapely.Polygon, shapely.Polygon]:

    """ shift the image randomly as long as the annotation_bbox stays COMPLETELY within the image
    :param image:
    :param annotation_bbox:
    :param offset:
    :return:
    """

    if isinstance(annotation_bbox, ImageLabel):
        annotation_bbox = annotation_bbox.bbox_polygon

    # translate the cutout box to the new image
    minx, miny, maxx, maxy = image_cutout_box.bounds

    # Calculate width and height
    image_width = maxx - minx
    image_height = maxy - miny
    # cut the image to a smaller box around the annotation
    cutout_bbox_local = add_offset_centroid(annotation_bbox.centroid,
                                     image_height=image_height,
                                     image_width=image_width,
                                     offset=new_y_height // 2)

    annotation_bbox_local = reframe_bounding_box(cutout_box=cutout_bbox_local, label=annotation_bbox)

    # move annotation_bbox to the new image
    cutout_box_global, annotation_bbox_local = random_translation_within_bounds(c_box=cutout_bbox_local,
                                                                                 image_cutout_box=image_cutout_box,
                                                                                 a_bbox=annotation_bbox)

    cutout_box_local = affinity.translate(cutout_box_global, -cutout_box_global.bounds[0], -cutout_box_global.bounds[1])

    return cutout_box_global, cutout_box_local, annotation_bbox_local


def custom_rotate(im: PILImage, labels, angle=None):
    """
    Rotate the image and the labels since the albumentations version looses the identity of the labels
    :param im:
    :param labels:
    :param angle:
    :return:
    """

    if angle is None:
        angle = np.random.randint(0, 360)

    imr = np.array(im, dtype=np.uint8)
    cx, cy, ch = imr.shape
    center = (cx // 2, cy // 2)
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(imr, M, (imr.shape[1], imr.shape[0]))

    # rotate image and polygon
    labels_rotated = []
    for l in labels:
        bbox_obb = shapely.affinity.rotate(l.bbox_polygon, angle, origin=center, use_radians=False)
        l.bbox = bbox_obb.bounds
        labels_rotated.append(l)

    rotated = PIL.Image.fromarray(rotated)

    rotated = rotated.convert("RGB")

    return rotated, labels_rotated, angle


def crop_augment_empty(image: PIL.Image,
                       labels: typing.List[ImageLabel],
                       intermediate_box_size: int = 600,
                       final_box_size: int = 224,
                       transforms=None,
                       visualise=False) -> PIL.Image:

    """ Crop a perfectly rotated image with no annotations
    """
    i_width, i_height = image.size
    image_polygon = shapely.box(*[0, 0, i_width, i_height])

    # generate random boxes and hope they are empty
    empty_boxes = []
    n = 0

    # try getting a random box
    while len(empty_boxes) < 1 and n < 100:
        cutout_box_global = shapely.box(*[0, 0, intermediate_box_size, intermediate_box_size])
        angle = random.randint(0, 360)
        # cutout_box_global = affinity.rotate(cutout_box_global, angle=angle, origin="centroid")
        cutout_box_global = affinity.translate(cutout_box_global,
                                               xoff=random.randint(0, i_width - intermediate_box_size),
                                               yoff=random.randint(0, i_height - intermediate_box_size))

        labels_in_area = [il for il in labels if il.bbox_polygon.within(cutout_box_global)]
        # the box is within the image and no other label is within the box
        if (cutout_box_global.within(image_polygon) and
                len(labels_in_area) == 0):

            image = crop_image_polygon_v2(image=image, bbox=cutout_box_global)
            if visualise:
                ax_ig = visualise_image(image=image, show=True, title="cropped image")
            if transforms:
                # Apply the transforms

                image, _ = custom_transforms(image, [], transforms)
                if visualise:
                    ax_ig = visualise_image(image=image, show=True, title=f"augmented image")

            custom_rotate(im=image, labels=labels, angle=angle)
            # Create an instance of the RandomCropAugmentation class
            crop_augmenter = RandomCropAugmentation(crop_height=final_box_size, crop_width=final_box_size)

            # Apply random crop augmentation
            cropped_image = crop_augmenter(image)

            if visualise:
                ax_ig = visualise_image(image=cropped_image, show=True, title=f"empty cropped image_{final_box_size}")

            empty_boxes.append(cutout_box_global)

            return cropped_image

        n += 1

    # incase we didn't find a suitable image this will return a noisy image
    noise = np.random.randint(0, 256, (final_box_size, final_box_size, 3), dtype=np.uint8)

    # Create a Pillow Image from the noise array
    noisy_image = PIL.Image.fromarray(noise)
    return noisy_image  # in case we where not able to generate an empty image


def crop_augment_crop(selected_image_label: ImageLabel,
                      image: PIL.Image,
                      labels: typing.List[ImageLabel],
                      offset=600,
                      final_box_size=224,
                      padding=300,
                      albumentations_transforms=None,
                      visualise=False) -> typing.Tuple[PIL.Image, typing.List[ImageLabel]]:
    """
    Crop the image around the selected image label, rotate/augment it and crop it again to the final_size
    :param selected_image_label:
    :param image:
    :param labels:
    :param offset:
    :param final_box_size:
    :return:
    """
    i_width, i_height = image.size
    image_polygon = shapely.box(*[0, 0, i_width, i_height])
    # bbox = selected_image_label.bbox_polygon

    # First Crop in the area of the object
    cutout_box_global, cutout_box_local, annotation_box_local = centered_fitted_crop_around_bbox(
        image_polygon=image_polygon,
        bbox_global=selected_image_label.bbox_polygon,
        offset=offset
    )

    if visualise:
        ax_ig = visualise_image(image=image, show=False, title="original image")
        ax = visualise_polygons([cutout_box_global],
                                filename=None,
                                show=False, title="All Annotations", max_x=i_width, max_y=i_height, color="r", ax=ax_ig)
        ax = visualise_polygons([il.bbox_polygon for il in labels],
                                filename=None,
                                show=False, title="All Annotations", max_x=i_width, max_y=i_height, color="white",
                                ax=ax)
        ax = visualise_polygons([selected_image_label.bbox_polygon],
                                filename=None,
                                show=True, title="All Annotations", max_x=i_width, max_y=i_height, color="blue", ax=ax)

    # FIXME this becomes dirty
    # Create an instance of the PadToFixedSizeAllSides class with default padding of 300

    padder = PadToFixedSizeAllSides(padding=padding)
    cutout_box_local = shapely.box(
        *[0, 0, cutout_box_local.bounds[3] + 2 * padding, cutout_box_local.bounds[3] + 2 * padding])

    # find all image labels that are within the cutout box # TODO do this properly
    image_labels = [il for il in labels if il.bbox_polygon.centroid.within(cutout_box_global)]

    # reproject the image labels to the new image
    image_labels = [reframe_bounding_box(cutout_box=cutout_box_global, label=il) for il in image_labels]

    # crop the full image to the cutout box
    im1 = crop_image_polygon(image=image, bbox=cutout_box_global)

    # Apply padding
    padder(im1, image_labels)

    im1 = padder.padded_image
    image_labels = padder.padded_labels

    # random rotation
    if albumentations_transforms:
        im2, image_labels_2, angle = custom_rotate(im1, labels=image_labels)
    else:
        im2 = im1
        image_labels_2 = image_labels

    image_labels_2 = [reframe_bounding_box(cutout_box=cutout_box_local, label=il) for il in image_labels_2]
    image_labels_2 = list(filter(lambda x: x is not None, image_labels_2))

    if visualise:
        ax_ig = visualise_image(image=im2, show=False, title="cropped image")
        ax = visualise_polygons([cutout_box_local], show=False, title="cutout box", ax=ax_ig)
        ax = visualise_polygons([i.bbox_polygon for i in image_labels_2], show=True, title="cutout box 2", ax=ax,
                                color="b")

    # Quality augmentation like
    im2, image_labels_2 = custom_transforms(im2,
                                            labels=image_labels_2,
                                            albumentations_transforms=albumentations_transforms)

    # identify the Box we started with
    reframed_selected_image_label = get_by_id(selected_image_label.id, image_labels_2)
    if reframed_selected_image_label is None:
        raise LabelMissingException(
            "The selected image label is not in the list of labels. Probably because of the rotation or augmentation. That happens if it is on the corner of the original imge.")

    ## create a crop around the annotation which is somewhere around the box
    cutout_box_global_2, cutout_box_local_2, annotation_box_local = random_crop_around_bbox_v2(
        image_cutout_box=cutout_box_local,
        annotation_bbox=reframed_selected_image_label.bbox_polygon,
        new_x_width=final_box_size,
        new_y_height=final_box_size
    )

    assert cutout_box_global_2.bounds[2] - cutout_box_global_2.bounds[0] == final_box_size, "Width is not correct"
    assert cutout_box_global_2.bounds[3] - cutout_box_global_2.bounds[1] == final_box_size, "Height is not correct"

    assert reframed_selected_image_label.bbox_polygon.centroid.intersects(
        cutout_box_global_2), "Centroid is not within cutout_box_global_2"

    # remove labels which don't fit anymore
    # FIXME class_name is not iguana
    image_labels_3 = [il for il in image_labels_2 if il.bbox_polygon.intersects(cutout_box_global_2)]

    # crop the image to the new size
    im3 = crop_image_polygon(image=im2, bbox=cutout_box_global_2)

    # project them into the new local image
    # TODO ensure none of the labels is outside the cutout box
    image_labels_3 = [reframe_polygon(cutout_box=cutout_box_global_2, label=il, fit_to_box=False) for il in
                      image_labels_3]

    bordering_labels = [
        il for il in image_labels_3 if
        il.bbox_polygon.intersects(cutout_box_local_2)
        and not il.bbox_polygon.within(cutout_box_local_2)
    ]
    if visualise:
        ax = visualise_image(image=im3, show=False, title="final image")
        visualise_polygons([i.bbox_polygon for i in image_labels_3], show=True, title="image polygon", color="g", ax=ax)

    # TODO the bordering labels are kind of a problem
    if len(bordering_labels) > 0:
        logger.info(f"Bordering labels exist. Blacking them out.: {bordering_labels}")
        for il in bordering_labels:
            im3 = blackout_bbox(im3, bbox_x1y1x2y2=[int(x) for x in il.bbox])

    image_labels_3 = [il for il in image_labels_3 if il not in bordering_labels]

    assert reframed_selected_image_label in image_labels_3, "The selected image label is not in the list of labels"

    if visualise:
        ax = visualise_image(image=im3, show=False, title="final image")
        ax = visualise_polygons([i.bbox_polygon for i in image_labels_3], show=True, title="image polygon", ax=ax,
                                color="g")
        sleep(0.1)
    if len(image_labels_3) == 0:
        raise LabelMissingException("There are no labels anymore")

    return im3, image_labels_3

def get_by_id(box_id: str, labels: typing.List[ImageLabel]) -> ImageLabel:
    return next((l for l in labels if l.id == box_id), None) # None is the default if not found

def get_image_by_id(image_id: str, images: typing.List[HastyImage]) -> HastyImage:
    return next((i for i in images if i.image_id == image_id), None) # None is the default if not found

def augs_gen(hA_images: typing.List[HastyImage],
                      offset=600,
                      final_box_size=224):
    """
    Generator to create the augmentations
    :param hA_images:
    :return:
    """
    # FIXME, don't iterate through the images, but the labels of all of them.
    for i in hA_images:
        # TODO plot the image itself
        image_path = full_images_path / i.dataset_name / i.image_name

        try:
            rand_index = random.randint(0, len(i.labels) - 1)
            selected_image_label = i.labels[rand_index]
            # selected_image_label = i.labels[4]
            im = PIL.Image.open(image_path)

            cropped_image, annotations  = crop_augment_crop(
                selected_image_label=selected_image_label,
                image=im,
                labels=i.labels,
                offset=offset,
                final_box_size=final_box_size)

            yield cropped_image, annotations

        except Exception as e:
            logger.warning(e)



class PadToFixedSizeAllSides:
    def __init__(self, padding=300, fill=0):
        """
        Args:
            padding (int): Desired padding to add to all sides.
            fill (int or tuple): Pixel fill value for constant fill. Default is 0.
        """
        self.padding = padding
        self.fill = fill

        self.padded_image = None
        self.padded_labels = None

    def __call__(self, img: PILImage, labels: typing.List[ImageLabel]):
        width, height = img.size
        labels = copy.deepcopy(labels)
        # Calculate the new size with padding
        new_width = width + 2 * self.padding
        new_height = height + 2 * self.padding

        # Create a new image with the specified background color
        new_image = Image.new("RGB", (new_width, new_height), self.fill)

        # Paste the original image onto the new image, centered
        new_image.paste(img, (self.padding, self.padding))

        # Update the labels
        for label in labels:
            label.bbox = [x + self.padding for x in label.bbox]

        self.padded_image = new_image
        self.padded_labels = labels

        return new_image

    def pad(self, img: PILImage, labels: typing.List[ImageLabel]):
        return self.__call__(img, labels)



class StretchToFixedSizeAllSides:
    """
    Stretches the image to a fixed size by adding padding to the sides.

    """
    def __init__(self, grid_size=224):
        """
        Args:
            grid_size (int): Desired padding to add to all sides.
            fill (int or tuple): Pixel fill value for constant fill. Default is 0.
        """
        self.grid_size = grid_size

        self.image = None
        self.labels = None

    def calculate_new_size(self, width, height):
        new_width = ((width + self.grid_size - 1) // self.grid_size) * self.grid_size
        new_height = ((height + self.grid_size - 1) // self.grid_size) * self.grid_size
        return new_width, new_height

    def __call__(self, img: PILImage, labels: typing.List[ImageLabel]):
        # width, height = img.size
        labels = copy.deepcopy(labels)
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Get original dimensions
        height, width = img_np.shape[:2]

        # Calculate new dimensions
        new_width, new_height = self.calculate_new_size(width, height)

        # Define the transformation
        transform = A.Compose([
            A.Resize(new_height, new_width, interpolation=1, always_apply=True)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


        # Prepare the data for transformation
        if labels is not None and len(labels) > 0:
            bboxes = [label.bbox for label in labels]
            category_ids = [label.id for label in labels]  # Replace with actual category ids if available
            # Example of categories, should match the number of bboxes
            ## TODO use the label_id then, reconstruct the label from the label_id later
            # category_ids = [0] * len(bboxes)  # Replace with actual category ids if available

            augmented = transform(image=img_np, bboxes=bboxes, category_ids=category_ids)
            img_np = augmented['image']
            bboxes = augmented['bboxes']
        else:
            augmented = transform(image=img_np)
            img_np = augmented['image']


        # TODO create the ImageLabel now from the bboxes and label_id

        self.image = Image.fromarray(img_np)
        for i, bbox in enumerate(bboxes):
            labels[i].bbox = [int(x) for x in bbox] # reassign the coordinates and round/convert to int
        self.labels =  labels if labels is not None else None

        # Convert numpy array back to PIL image
        return Image.fromarray(img_np), bboxes if bboxes is not None else None

    def stretch(self, img: PILImage, labels: typing.List[ImageLabel]):
        return self.__call__(img, labels)


if __name__ == "__main__":
    # TODO create a test from this
    base_path = Path("/Users/christian/data/training_data/iguanas_2024_05_21")
    crop_size = 224

    dset = "train"
    # dset = "val"
    # dset = "test"

    output_base_path = Path("./")

    labels_path = base_path / dset
    full_images_path = labels_path

    final_training_path = base_path / "final_training"
    final_training_path.mkdir(exist_ok=True)

    empties_all = []
    with_opject_all = []

    hA = hA_from_file(labels_path / "hasty_format.json")
    hA.images = [i for i in hA.images if i.image_status == "DONE"]

    width, height = 600, 600
    offset = 600
    final_box_size = 224

    output_path = output_base_path / f"crops_individual"
    n = 0
    n_augmentations = 10
    augs = []

    for i in hA.images:
        # TODO plot the image itself
        image_path = full_images_path / i.dataset_name / i.image_name
        im = PIL.Image.open(image_path)
        crop_augment_empty(image=im, labels=i.labels, final_box_size=final_box_size)

        break

    gen = augs_gen(hA_images=hA.images)

    for i in gen:
        logger.info(f"n: {n}")
        if n >= n_augmentations:
            break
        n = n + 1

    print("done")




