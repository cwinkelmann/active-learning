""" from flight simulation sim"""
import copy
import typing
import uuid

import shutil
from pathlib import Path
import hashlib
from io import BytesIO
import numpy as np
import pandas as pd
import PIL.Image
import shapely
from loguru import logger
from typing import List, Union, Tuple

from shapely import Polygon, affinity, box

from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.Annotation import project_point_to_crop, project_label_to_crop
from com.biospheredata.converter.Annotation import add_offset_to_box
from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, ImageLabelCollection, PredictedImageLabel, Keypoint
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage
from com.biospheredata.types.annotationbox import BboxXYXY, BboxXYWH, xywh2xyxy
from com.biospheredata.visualization.visualize_result import blackout_bbox
from PIL import Image

from image_rasterization import generate_positions

DATA_SET_NAME = "train_augmented"


import PIL.Image
import shapely
from shapely.geometry.polygon import Polygon
from typing import Tuple, List, Dict

from com.biospheredata.types.HastyAnnotationV2 import ImageLabel


def create_box_point_ImageLabel(il: ImageLabel, width: int, height: int) -> Polygon:
    # TODO implement this right
    return create_box_from_point(il.x_yolo, il.y_yolo, width, height)

def create_box_point_shapely(p: shapely.Point, width: int, height: int) -> Polygon:
    return create_box_from_point(p.x, p.y, width, height)


def create_box_from_point(x: float, y: float, width: int, height: int) -> Polygon:
    """
    Create a rectangular box from a point and its dimensions.

    Parameters:
        x (int): x-coordinate of the point.
        y (int): y-coordinate of the point.
        width (int): Width of the box.
        height (int): Height of the box.

    Returns:
        Polygon: A shapely Polygon representing the box.
    """
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2

    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def create_regular_raster_grid_from_center(max_x: int, max_y: int,
                                           slice_height: int,
                                           slice_width: int) -> [List[Polygon]]:
    """
    Create a regular raster grid of bounding boxes, starting from the center of the image,
    with the center point located at the intersection of four tiles.
    """
    tiles = []
    tile_coordinates = []

    center_x = max_x / 2
    center_y = max_y / 2

    # Generate x and y positions
    x_positions = generate_positions(center_x, max_x, slice_width)
    y_positions = generate_positions(center_y, max_y, slice_height)

    # Generate tiles and calculate distances from the center
    tiles_info = []

    for height_i, y1 in enumerate(y_positions):
        y2 = y1 + slice_height
        for width_j, x1 in enumerate(x_positions):
            x2 = x1 + slice_width

            # Ensure tiles are within image boundaries
            x1_clipped = max(x1, 0)
            y1_clipped = max(y1, 0)
            x2_clipped = min(x2, max_x)
            y2_clipped = min(y2, max_y)

            # Adjust tile size if it goes beyond boundaries
            if x2_clipped - x1_clipped <= 0 or y2_clipped - y1_clipped <= 0:
                continue

            # Calculate the center of the tile
            tile_center_x = (x1_clipped + x2_clipped) / 2
            tile_center_y = (y1_clipped + y2_clipped) / 2

            # Calculate the Euclidean distance from the tile center to the image center
            distance = ((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2) ** 0.5

            # Append tile information
            tiles_info.append((distance, height_i, width_j, x1_clipped, y1_clipped, x2_clipped, y2_clipped))

    # Sort the tiles based on their distance from the image center (closest first)
    tiles_info_sorted = sorted(tiles_info, key=lambda x: x[0])

    # Generate tiles starting from the center
    for info in tiles_info_sorted:
        distance, height_i, width_j, x1, y1, x2, y2 = info

        # Create the polygon for the tile
        pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        # Append the polygon and its coordinates to the lists
        tiles.append(pol)

        tile_coordinates.append({
            "height_i": height_i,
            "width_j": width_j,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return tiles, tile_coordinates

def create_regular_raster_grid(max_x: int, max_y: int,
                               slice_height: int,
                               slice_width: int,
                               overlap: int = 0) -> Tuple[List[Polygon], List[Dict]]:
    """
    Create a regular raster grid of bounding boxes starting from the top-left corner of the image.

    Parameters:
        max_x (int): Width of the image.
        max_y (int): Height of the image.
        slice_height (int): Height of each tile.
        slice_width (int): Width of each tile.
        overlap (int): Overlap between tiles in pixels.

    Returns:
        Tuple[List[Polygon], List[Dict]]:
            - A list of shapely Polygons representing the grid tiles.
            - A list of dictionaries with tile metadata (indices and coordinates).
    """
    tiles = []
    tile_coordinates = []

    # Calculate step size (tile size minus overlap)
    step_x = slice_width - overlap
    step_y = slice_height - overlap

    # Iterate over grid positions
    for y1 in range(0, max_y, step_y):
        for x1 in range(0, max_x, step_x):
            x2 = x1 + slice_width
            y2 = y1 + slice_height

            # Create a polygon for the tile
            pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            if x2 <= max_x and y2 <= max_y:
                # Append to results
                tiles.append(pol)
                tile_coordinates.append({
                    "height_i": y1 // step_y,
                    "width_j": x1 // step_x,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # Break when the tile reaches the max dimensions
            if x2 == max_x:
                break
        if y2 == max_y:
            break

    return tiles, tile_coordinates



# TODO this is more or less the same as the function above create_regular_raster_grid
def crop_by_regular_grid(
        crop_size,
        full_images_path_padded,
        i,
        images_path,
        train_images_output_path,
        empty_fraction,
        overlap):
    """

    :param crop_size:
    :param full_images_path_padded:
    :param i:
    :param images_path:
    :param train_images_output_path:
    :param empty_fraction:
    :param overlap:
    :return:
    """
    # original_image_path = self.images_path / i.dataset_name / i.image_name if i.dataset_name else self.images_path / i.image_name
    # padded_image_path = full_images_path_padded / i.dataset_name / i.image_name if i.dataset_name else full_images_path_padded / i.image_name
    original_image_path = images_path / i.image_name
    padded_image_path = full_images_path_padded / i.image_name
    padded_image_path.parent.mkdir(exist_ok=True, parents=True)
    new_width, new_height = pad_to_multiple(original_image_path,
                                            padded_image_path,
                                            crop_size,
                                            crop_size,
                                            overlap)
    logger.info(f"Padded {i.image_name} to {new_width}x{new_height}")

    grid, _ = create_regular_raster_grid(max_x=new_width,
                                         max_y=new_height,
                                         slice_height=crop_size,
                                         slice_width=crop_size,
                                         overlap=overlap)

    logger.info(f"Created grid for {i.image_name} with {len(grid)} tiles")
    # TODO visualise the grid
    # axi = visualise_image(image_path=padded_image_path, show=False, title=f"grid_{i.image_name}", )
    # visualise_polygons(grid, show=True, title=f"grid_{i.image_name}", max_x=new_width, max_y=new_height, ax=axi)

    images, cropped_images_path = crop_out_images_v2(i, rasters=grid,
                                                     full_image_path=padded_image_path,
                                                     output_path=train_images_output_path,
                                                     include_empty=empty_fraction,
                                                     dataset_name=i.dataset_name)
    # image = Image.open(full_images_path / i.dataset_name / i.image_name)
    logger.info(f"Cropped {len(images)} images from {i.image_name}")

    return images, cropped_images_path

def crop_polygons(image: PIL.Image.Image,
                  rasters: List[shapely.Polygon],
                  ) -> List[PIL.Image.Image]:
    """
    iterate through rasters and crop out the tiles from the image return the new images

    :param image:
    :param rasters:
    :param output_path:
    :param full_images_path:

    """
    assert isinstance(image, PIL.Image.Image)
    images = []

    # sliding window of the image
    for pol in rasters:
        assert isinstance(pol, shapely.Polygon)
        sliced = image.crop(pol.bounds)
        images.append(sliced)

    return images


# Assume `sliced` is your cropped image
def get_image_hash(image: Image.Image, hash_function="sha256"):
    """Generate a unique hash for an image crop."""
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")  # Use PNG to ensure consistency
    img_bytes = img_bytes.getvalue()

    if hash_function == "sha256":
        return hashlib.sha256(img_bytes).hexdigest()
    elif hash_function == "md5":
        return hashlib.md5(img_bytes).hexdigest()
    else:
        raise ValueError("Unsupported hash function")


def crop_out_individual_object(i: ImageLabelCollection,
                               im: PIL.Image.Image,
                               output_path: Path,
                               offset: int = None,
                               width: int = None,
                               height: int = None) -> Tuple[List[ImageCropMetadata], List[AnnotatedImage], List[Path]]:
    """
    Create crops from individual objects in each image.

    Parameters:
    - i: ImageLabelCollection object containing labels and bounding boxes.
    - im: PIL Image object.
    - output_path: Path to save cropped images.
    - offset: Optional, offset to add around bounding boxes.
    - width: Optional, width of cropped images.
    - height: Optional, height of cropped images.

    Returns:
    - Tuple containing:
        - List of ImageCropMetadata mapping to see which images is cropped.
        - List of AnnotatedImage objects with cropped image data.
        - List of cropped image paths.
    """

    assert isinstance(i, ImageLabelCollection)
    cropped_annotated_images: List[AnnotatedImage] = []
    boxes: List[Polygon] = []
    image_mappings: List[ImageCropMetadata] = []
    images_set: List[Path] = []

    for label in i.labels:
        label_id = label.id

        # add a bit of an offset
        if offset and label.bbox_polygon is not None and isinstance(label.bbox_polygon, shapely.Polygon):
            box = add_offset_to_box(label.bbox, i.height, i.width, offset)
            boxes.append(shapely.box(*box))

        # constant boundary around the box
        elif width and height:
            if label.bbox_polygon is not None and isinstance(label.bbox_polygon, shapely.Polygon):
                box = resize_box(label.bbox_polygon, width, height)
            else:
                box = create_box_from_point(x=label.centroid.x, y=label.centroid.y, width=width, height=height)

            # TODO write a function for this, Ensure the bounding box stays within image bounds
            if box.bounds[0] < 0:
                box = affinity.translate(box, -box.bounds[0], 0)
            if box.bounds[1] < 0:
                box = affinity.translate(box, 0, -box.bounds[1])

            if box.bounds[2] > i.width:
                box = affinity.translate(box, i.width - box.bounds[2], 0)
            if box.bounds[3] > i.height:
                box = affinity.translate(box, 0, i.height - box.bounds[3])

            # TODO project every keypoint, box or segmentation mask to the new crop
            projected_keypoint = project_point_to_crop(label.incenter_centroid, box)
            # project_label_to_crop(label, box) # TODO the code above and below can be outsourced into this.

            # TODO get the ids right and keep the projections
            projected_keypoint = [Keypoint(
                id = label.keypoints[0].id, # TODO get this right for cases when there are other label types
                x=int(projected_keypoint.x),
                y=int(projected_keypoint.y),
                keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff", # TODO programmatically get the right class id
            )]  # you can store extra information if needed

            if isinstance(label, ImageLabel):
                pI = ImageLabel(id=label_id, class_name=label.class_name,
                                         keypoints=projected_keypoint)
            elif isinstance(label, PredictedImageLabel):
                pI = PredictedImageLabel(id=label_id, class_name=label.class_name,
                                         keypoints=projected_keypoint, score=label.score,
                                         kind=None)
            else:
                raise NotImplementedError("Only ImageLabel and PredictedImageLabel are supported")

            boxes.append(box)
            sliced = im.crop(box.bounds)
            image_id = get_image_hash(sliced)
            slice_path_jpg = output_path / Path(f"{i.image_name}_{label_id}.jpg")
            sliced.save(slice_path_jpg)
            images_set.append(slice_path_jpg)

            aI = AnnotatedImage(
                # image_id=str(uuid.uuid4()),
                image_id=image_id,
                dataset_name="cropped_predictions",
                image_name=slice_path_jpg.name,
                labels=[pI],
                width=int(width),
                height=int(width))

            # TOTO this is deprecated
            image_mapping = ImageCropMetadata(
                            parent_image = i.image_name,
                            parent_image_id = i.image_id,
                            parent_label_id = label_id,
                            cropped_image = slice_path_jpg.name,
                            cropped_image_id = aI.image_id,
                            box = [int(box.bounds[0]), int(box.bounds[1]), int(box.bounds[2]), int(box.bounds[3])],  # Creates a Shapely bounding box
                            local_coordinate = pI.keypoints,  # Example point inside crop
                            global_coordinate = label.keypoints  # Original image point
                        )
            image_mappings.append(image_mapping)

            cropped_annotated_images.append(aI)

        else:
            raise ValueError("offset or width and height must be provided")

    return image_mappings, cropped_annotated_images, images_set

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


def resize_box(rectangle: Polygon, width, height) -> Polygon:
    minx, miny, maxx, maxy = rectangle.bounds

    resize_x = width - (maxx - minx)
    resize_y = height - (maxy - miny)
    new_rectangle = Polygon([
        (int(round(minx - resize_x / 2)), int(round(miny - resize_y / 2))),
        (int(round(maxx + resize_x / 2)), int(round(miny - resize_y / 2))),
        (int(round(maxx + resize_x / 2)), int(round(maxy + resize_y / 2))),
        (int(round(minx - resize_x / 2)), int(round(maxy + resize_y / 2)))
    ])

    return new_rectangle


## TODO this seems to be redundand too, since it create a regular raster grid
def crop_out_images(hi: AnnotatedImage,
                    slice_height: int,
                    slice_width: int,
                    full_images_path: Path,
                    output_path: Path,
                    ) -> (List[Path], List[Path], List[AnnotatedImage]):
    """ iterate through the image and crop out regular grid tiles

    :param output_path:
    :param full_images_path:
    :param slice_width:
    :param slice_height:
    :param hi:
    """
    counter = 0  ## TODO is this the right position?
    images_with_objects = []
    images_without_objects = []
    labels_paths = []

    output_path.mkdir(parents=True, exist_ok=True)
    output_path.joinpath("object").mkdir(parents=True, exist_ok=True)
    output_path.joinpath("empty").mkdir(parents=True, exist_ok=True)

    image = PIL.Image.open(full_images_path / hi.dataset_name / hi.image_name)
    # imr = np.array(image)

    # Convert to string if you need a string representation
    annotations = []

    # slice the image in tiles
    for height_i in range((hi.height // slice_height)):
        for width_j in range((hi.width // slice_width)):
            x1 = width_j * slice_width
            y1 = hi.height - (height_i * slice_height)
            x2 = ((width_j + 1) * slice_width)
            y2 = (hi.height - (height_i + 1) * slice_height)

            # cut the tile slice from the image

            sliced = image.crop((x1, y2, x2, y1))  # TODO use a type for these?

            # TODO save the coordinate of the slice to reconstruct it later

            # sliding window of the image
            pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            # TODO create boxes too
            imsaved = False
            slice_labels = []

            ## TODO extract this to another function
            empty = True
            reject = False
            max_bbox_area = 0
            one_good_box = False

            for box in hi.labels:
                # iterate through labels and check if any of the boxes intersects with the sliding window

                if pol.intersects(box.bbox_polygon):
                    # any of the annotations is in the sliding window
                    intersection_polygon = shapely.intersection(pol, box.bbox_polygon)
                    max_bbox_area = max(max_bbox_area, intersection_polygon.area)
                    empty = False

                    if box.bbox_polygon.within(pol):
                        one_good_box = True

                        minx, miny, maxx, maxy = pol.bounds

                        # Translate the coordinates of the inner polygon
                        translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
                        # Create a new polygon with the translated coordinates
                        translated_inner_polygon = Polygon(translated_coords)
                        il = ImageLabel(id=str(uuid.uuid4()),
                                        image_id=str(uuid.uuid4()),
                                        # category_id="888",  # TODO get this right
                                        class_name="iguana",
                                        bbox=[int(x) for x in translated_inner_polygon.bounds],
                                        iscrowd=0, segmentation=[])
                        slice_labels.append(il)

                    else:
                        # a part of the box is in the sliding window
                        minx, miny, maxx, maxy = pol.bounds
                        # Translate the coordinates of the inner polygon
                        translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
                        translated_inner_polygon = Polygon(translated_coords)

                        sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])

            if not empty:
                # print("not empty")
                pass

            if empty == False and max_bbox_area < 5000:
                reject = True

            # sliced_im = PIL.Image.fromarray(sliced)

            filename = str(Path(hi.image_name).stem)

            if not reject or one_good_box:
                if empty:
                    slice_path_jpg = output_path / "empty" / f"{filename}_{height_i}_{width_j}.jpg"
                    slice_path_tiff = output_path / "empty" / f"{filename}_{height_i}_{width_j}.tiff"
                    slice_path_png = output_path / "empty" / f"{filename}_{height_i}_{width_j}.png"
                    images_without_objects.append(slice_path_jpg)

                else:
                    slice_path_jpg = output_path / "object" / f"{filename}_{height_i}_{width_j}.jpg"
                    slice_path_tiff = output_path / "object" / f"{filename}_{height_i}_{width_j}.tiff"
                    slice_path_png = output_path / "object" / f"{filename}_{height_i}_{width_j}.png"

                    images_with_objects.append(slice_path_jpg)
                    im = AnnotatedImage(
                        image_id=str(uuid.uuid4()),
                        dataset_name="train_augmented",
                        image_name=f"{filename}_{height_i}_{width_j}.jpg",
                        labels=slice_labels,
                        width=slice_width,
                        height=slice_height)
                    annotations.append(im)

                # logger.info(f"slice_path: {slice_path_jpg}")
                sliced_im = sliced.convert("RGB")  # TODO I need this?
                sliced_im.save(slice_path_jpg, 'JPEG', quality=90, optimize=True)
                #sliced_im.save(slice_path_tiff, 'TIFF', compression=None)
                #sliced_im.save(slice_path_png )

    return images_without_objects, images_with_objects, annotations

def include_empty_frac(frac) -> bool:
    if frac == False:
        return frac
    return np.random.rand() < frac

def crop_out_images_v2(hi: AnnotatedImage,
                       rasters: List[shapely.Polygon],
                       full_image_path: Path,
                       output_path: Path,
                       dataset_name: str = DATA_SET_NAME,
                       include_empty = False,
                       edge_blackout = True) -> typing.Tuple[List[AnnotatedImage], List[Path]]:
    """ iterate through rasters and crop out the tiles from the image return the new images and an annotations file

    :param include_empty:
    :param dataset_name:
    :param full_image_path:
    :param rasters:
    :param output_path:
    :param full_images_path:

    :param hi:
    """

    images_with_objects = []

    output_path.mkdir(parents=True, exist_ok=True)
    output_path_images = output_path
    output_path_images.mkdir(parents=True, exist_ok=True)

    image = PIL.Image.open(full_image_path)
    # imr = np.array(image)

    # Convert to string if you need a string representation
    images = []
    image_paths: typing.List[Path] = []

    # slice the image in tiles

    # sliding window of the image
    for pol in rasters:
        assert isinstance(pol, shapely.Polygon)
        sliced = image.crop(pol.bounds)
        minx, miny, maxx, maxy = pol.bounds
        slice_width = maxx - minx
        slice_height = maxy - miny
        slice_labels = []


        ## TODO extract this to another function
        empty = True
        reject = False
        max_bbox_area = 0
        one_good_box = False
        raster_id = 0 # TODO encapsulate the Rasters into a class which contains a trackable id

        masks = [] # for bordering bounding boxes, those are blacked out, the masks are saved here to remove keypoints later


        for annotation in hi.labels:
            is_polygon = isinstance(annotation.polygon_s, shapely.Polygon)
            is_keypoint = isinstance(annotation.keypoints, typing.List) and len(annotation.keypoints) > 0
            is_bbox = isinstance(annotation.bbox_polygon, shapely.Polygon) and not is_polygon

            # iterate through labels and check if any of the boxes intersects with the sliding window
            if ((is_polygon and pol.intersects(annotation.polygon_s))
                    or (is_bbox and pol.intersects(annotation.bbox_polygon))
                    or (is_keypoint and pol.contains(annotation.keypoints[0].coordinate))):

                # any of the annotations is in the sliding window
                if is_bbox:
                    intersection_polygon = shapely.intersection(pol, annotation.bbox_polygon)
                    max_bbox_area = max(max_bbox_area, intersection_polygon.area)
                elif is_keypoint:
                    is_point_inside = shapely.Point(annotation.keypoints[0].coordinate).within(pol)
                else:
                    try:
                        intersection_polygon = shapely.intersection(pol, annotation.polygon_s)
                        max_bbox_area = max(max_bbox_area, intersection_polygon.area)
                    except shapely.errors.GEOSException as e:
                        logger.error(f"Error in intersection: {e}")
                        logger.error(f"Annotation is not valid: {annotation.id} ")
                        continue

                if is_bbox and annotation.bbox_polygon.within(pol):
                    # the box is completely within the sliding window
                    one_good_box = True
                    empty = False
                    # Translate the coordinates of the inner polygon
                    translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
                    # Create a new polygon with the translated coordinates
                    translated_inner_polygon = Polygon(translated_coords)
                    il = ImageLabel(
                        id=str(uuid.uuid4()),
                        class_name=annotation.class_name,
                        bbox=[int(x) for x in translated_inner_polygon.bounds],
                    )
                    slice_labels.append(il)


                elif is_polygon and annotation.polygon_s.within(pol):
                    # the Polygon is completely within the sliding window
                    one_good_box = True
                    empty = False
                    # Translate the coordinates of the inner polygon
                    translated_coords = [(x - minx, y - miny) for x, y in annotation.polygon]
                    # Create a new polygon with the translated coordinates

                    il = ImageLabel(
                        id=annotation.id,
                        class_name=annotation.class_name,
                        polygon=translated_coords,
                    )
                    slice_labels.append(il)
                    # a part of the box is outside of the sliding window, we want to black it out


                # Process the keypoints
                elif is_keypoint and annotation.keypoints[0].coordinate.within(pol):
                    # translated_coords = [(k.x - minx, k.y - miny) for k in box.keypoints]
                    # Create a new polygon with the translated coordinates
                    empty = False
                    box_keypoints = []
                    for k in annotation.keypoints:
                        kc = copy.deepcopy(k)
                        kc.x = int(k.x - minx)
                        kc.y = int(k.y - miny)
                        box_keypoints.append(kc)

                    #translated_keypoints = [Keypoint(x=int(k.x - minx), y=int(k.y - miny)) for k in box.keypoints]

                    il = ImageLabel(
                        id=annotation.id,
                        class_name=annotation.class_name,
                        keypoints=box_keypoints,
                    )
                    slice_labels.append(il)

                elif is_keypoint and not annotation.keypoints[0].coordinate.within(pol):
                    # The keypoint is outside of the sliding window
                    # So far we do not do anything here
                    pass


                # elif box.attributes.get("partial", False) or box.attributes.get("visibility", -1) < 2:
                #     logger.info(f"Bor or Polygon is a partial={box.attributes.get('partial')} or is badly visible=box.attributes.get('visibility', -1)")
                #
                #     translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
                #     translated_inner_polygon = Polygon(translated_coords)
                #     sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])

                else:
                    # logger.info(f"Box or polygon is not completly within the sliding window {annotation.id}")
                    # Translate the coordinates of the inner polygon
                    if edge_blackout:
                        translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
                        translated_inner_polygon = Polygon(translated_coords)
                        masks.append(translated_inner_polygon)

                        sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])

                    else:
                        raise NotImplementedError("Redraw the box or polygon to fit the sliding window")
            else:
                # logger.info(f"Box or polygon is not within the sliding window {annotation.id}")
                pass
        for m in masks:
            # remove points which are in the mask
            slice_labels = [sl for sl in slice_labels if isinstance(sl.incenter_centroid, shapely.Point) and sl.incenter_centroid.within(m) == False]


        if empty == False and max_bbox_area < 5000:
            # reject = True
            # logger.warning(f"Should Rejecting image, label very small {hi.image_name} because of max_bbox_area {max_bbox_area}")
            # slice_labels = []
            pass
        filename = str(Path(hi.image_name).stem)

        # if (not reject and not empty) or one_good_box:

        xx, yy = pol.exterior.coords.xy
        slice_path_jpg = output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"

        if (include_empty_frac(include_empty) or not empty) and not reject:

            im = AnnotatedImage(
                image_id=str(uuid.uuid4()),
                dataset_name=dataset_name if dataset_name else DATA_SET_NAME,
                image_name=slice_path_jpg.name,
                labels=slice_labels,
                width=int(slice_width),
                height=int(slice_height))

            images.append(im)

            sliced_im = sliced.convert("RGB")
            sliced_im.save(slice_path_jpg)
            image_paths.append(slice_path_jpg)

        raster_id += 1

    return images, image_paths


def crop_out_images_v3(image: PIL.Image,
                       rasters: List[shapely.Polygon],
                       ) -> List[PIL.Image]:
    """
    iterate through rasters and crop out the tiles from the image return the new images

    :param rasters:
    :param output_path:
    :param full_images_path:

    :param hi:
    """

    # imr = np.array(image)

    # Convert to string if you need a string representation
    images = []

    # slice the image in tiles

    # sliding window of the image
    for pol in rasters:
        assert isinstance(pol, shapely.Polygon)
        sliced = image.crop(pol.bounds)
        images.append(sliced)

    return images

def create_box_around(point: shapely.geometry.Point, box_width: float, box_height: float) -> shapely.geometry.Polygon:
    """
    Create a rectangular box (polygon) centered at a given point.

    Parameters:
        point: Shapely Point around which to create the box.
        box_width: Total width of the box.
        box_height: Total height of the box.

    Returns:
        A Shapely Polygon representing the box.
    """
    x, y = point.x, point.y
    half_w = box_width / 2.0
    half_h = box_height / 2.0
    # shapely.geometry.box(minx, miny, maxx, maxy)
    return shapely.geometry.box(x - half_w, y - half_h, x + half_w, y + half_h)




def pad_to_multiple(original_image_path: Path, padded_image_path: Path,
                    slice_width: int, slice_height: int, overlap: int):
    """
    Pads an image so that its width and height are rounded up to the nearest step defined by:
    step_x = slice_width - overlap
    step_y = slice_height - overlap

    Padding is applied only to the bottom (y_max) and right (x_max) edges.

    Parameters:
        original_image_path (Path): Path to the original image file.
        padded_image_path (Path): Path to save the padded image.
        slice_width (int): Width of each tile.
        slice_height (int): Height of each tile.
        overlap (int): Overlap between tiles in pixels.

    Returns:
        (int, int): The new padded width and height of the image.
    """
    # Open the original image
    image = Image.open(original_image_path)
    width, height = image.size

    # Compute the step sizes
    step_x = slice_width - overlap
    step_y = slice_height - overlap

    # Calculate padding so that width and height become multiples of step_x and step_y respectively
    padding_x = (step_x - (width % step_x)) % step_x
    padding_y = (step_y - (height % step_y)) % step_y

    new_width = width + padding_x
    new_height = height + padding_y

    # Create a new image with the padded dimensions
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))  # black background
    padded_image.paste(image, (0, 0))
    padded_image.save(padded_image_path)

    logger.info(f"Padded image saved to {padded_image_path} with size: {new_width}x{new_height}")
    return new_width, new_height


def sliced_predict_geotiff(geotiff_path: Path):
    """

    :param geotiff_path:
    :return:
    """
    slicer = GeoSlicer(base_path=geotiff_path.parent, image_name=geotiff_path.name, x_size=5120, y_size=5120)
    tiles = slicer.slice_very_big_raster()
    print(tiles)