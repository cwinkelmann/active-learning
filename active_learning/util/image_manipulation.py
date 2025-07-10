"""
Collection of function to manipulate images
"""
import PIL.Image
import copy
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import typing
import uuid
from PIL import Image
from io import BytesIO
from loguru import logger
from pathlib import Path
from shapely import affinity

from active_learning.filter import SpatialSampleStrategy
from active_learning.types.Exceptions import WrongSpatialSamplingStrategy, LabelInconsistenyError
from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.Annotation import project_point_to_crop
from active_learning.util.geospatial_slice import GeoSlicer
from active_learning.util.image import get_image_id
from active_learning.util.image_rasterization import generate_positions
#
from com.biospheredata.converter.Annotation import add_offset_to_box
from com.biospheredata.converter.HastyConverter import ImageFormat
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage
from com.biospheredata.types.HastyAnnotationV2 import ImageLabelCollection, PredictedImageLabel, Keypoint
from com.biospheredata.visualization.visualize_result import blackout_bbox
from com.biospheredata.types.HastyAnnotationV2 import ImageLabel

from image_template_search.util.util import visualise_image, visualise_polygons

DATA_SET_NAME = "train_augmented"

import PIL.Image
import shapely
from shapely.geometry.polygon import Polygon
from typing import Tuple, List, Dict
import geopandas as gpd



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


def crop_by_regular_grid(
        crop_size: int,
        full_images_path_padded: Path,
        i: AnnotatedImage,
        images_path: Path,
        train_images_output_path: Path,
        empty_fraction: float,
        overlap: float = 0.0,
        edge_black_out: bool = True,
        visualisation_path: Path = None,
        grid_manager: typing.Callable = None):
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

    # TODO use the grid manager to create the grid
    grid, _ = create_regular_raster_grid(max_x=new_width,
                                         max_y=new_height,
                                         slice_height=crop_size,
                                         slice_width=crop_size,
                                         overlap=overlap)

    logger.info(f"Created grid for {i.image_name} with {len(grid)} tiles")

    box_labels = [i for i in i.labels if i.bbox is not None]
    ib = copy.deepcopy(i)
    ib.labels = box_labels
    cropper = RasterCropperBoxes(hi=ib,
                                 rasters=grid,
                                 full_image_path=padded_image_path,
                                 output_path=train_images_output_path,
                                 include_empty=empty_fraction,
                                 dataset_name=i.dataset_name,
                                 edge_blackout=edge_black_out)

    point_labels = [i for i in i.labels if i.bbox is None]
    ip = copy.deepcopy(i)
    ip.labels = point_labels
    cropper_points = RasterCropperPoints(hi=ip,
                                 rasters=grid,
                                 full_image_path=padded_image_path,
                                 output_path=train_images_output_path,
                                 include_empty=empty_fraction,
                                 dataset_name=i.dataset_name,
                                 edge_blackout=edge_black_out)

    images_boxes, cropped_images_boxes_path = cropper.crop_out_images()
    images_points, cropped_images_points_path = cropper_points.crop_out_images(masks=cropper.masks)
    boxes_count = len(images_boxes)
    points_count = len(images_points)

    # Simplest version:
    if boxes_count > 0 and points_count > 0:
        logger.error(f"Either boxes {boxes_count} or points {points_count} should be cropped, not both.")

    if (boxes_count > 0 and points_count > 0):
        error_msg = f"Both have items - only one should be non-empty" if boxes_count > 0 else "Both are empty - at least one should have items. image: {i.image_name}"
        logger.error(f"Invalid state: boxes={boxes_count}, points={points_count}. {error_msg}, image: {i.image_name}")

        raise LabelInconsistenyError(error_msg)

    images = images_boxes + images_points
    cropped_images_path = cropped_images_boxes_path + cropped_images_points_path

    axi = visualise_image(image_path=padded_image_path, show=False, title=f"grid_{i.image_name}", )

    axi = visualise_polygons(cropper.empty_rasters, show=False, title=f"grid_{i.image_name}", max_x=new_width,
                             max_y=new_height, ax=axi, color="red")
    axi = visualise_polygons(cropper.occupied_rasters, show=False, title=f"grid_{i.image_name}", max_x=new_width,
                             max_y=new_height, ax=axi, color="green", linewidth=3)
    axi = visualise_polygons([c["closest_empty"] for c in cropper.closest_pairs], show=False, title=f"grid_{i.image_name}", max_x=new_width,
                             max_y=new_height, ax=axi, color="blue", linewidth=3)
    axi = visualise_polygons(cropper.partial_empty_rasters, show=False, title=f"grid_{i.image_name}", max_x=new_width,
                       max_y=new_height, ax=axi, color="orange", linewidth=0.8,
                       filename=visualisation_path / f"selected_grid_{padded_image_path.name}")
    plt.close()

    logger.info(f"Visualised grid for {i.image_name} with {len(cropper.empty_rasters)} empty tiles and {visualisation_path}")
    # visualise_polygons(grid, show=True, title=f"grid_{i.image_name}", max_x=new_width, max_y=new_height, ax=axi)

    logger.info(f"Cropped {len(images)} images from {i.image_name} to {train_images_output_path}")

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
        if offset is not None and label.bbox_polygon is not None and isinstance(label.bbox_polygon, shapely.Polygon):
            cutout_box = add_offset_to_box(label.bbox, i.height, i.width, offset)
            # boxes.append(shapely.box(*cutout_box))

            width = cutout_box[2] - cutout_box[0]
            height = cutout_box[3] - cutout_box[1]

        # constant boundary around the box
        if width and height:

            cutout_box = create_box_from_point(x=label.incenter_centroid.x, y=label.incenter_centroid.y, width=width,
                                               height=height)

            # TODO write a function for this, Ensure the bounding box stays within image bounds
            if cutout_box.bounds[0] < 0:
                cutout_box = affinity.translate(cutout_box, -cutout_box.bounds[0], 0)
            if cutout_box.bounds[1] < 0:
                cutout_box = affinity.translate(cutout_box, 0, -cutout_box.bounds[1])

            if cutout_box.bounds[2] > i.width:
                cutout_box = affinity.translate(cutout_box, i.width - cutout_box.bounds[2], 0)
            if cutout_box.bounds[3] > i.height:
                cutout_box = affinity.translate(cutout_box, 0, i.height - cutout_box.bounds[3])

            # TODO ensure every label type is handled correctly: Box, polygon and point
            # TODO project every keypoint, box or segmentation mask to the new crop
            projected_keypoint = project_point_to_crop(label.incenter_centroid, cutout_box)

            logger.warning(f"implement the other projections")
            # projected_label = project_label_to_crop(label, cutout_box) # TODO the code above and below can be outsourced into this.

            # TODO use these functions here:
            # reframe_bounding_box()

            # reframe_polygon()

            # TODO get the ids right and keep the projections
            if label.keypoints is not None and len(label.keypoints) > 0:
                label_id = label.keypoints[0].id
            else:
                label_id = label.id

            projected_keypoint = [Keypoint(
                id=label_id,  # TODO get this right for cases when there are other label types other than points
                x=int(projected_keypoint.x),
                y=int(projected_keypoint.y),
                keypoint_class_id="ed18e0f9-095f-46ff-bc95-febf4a53f0ff",
                # TODO programmatically get the right class id
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

            # TODO maybe do the cropping outside
            boxes.append(cutout_box)
            sliced = im.crop(cutout_box.bounds)
            image_id = get_image_id(image=sliced)
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

            # TODO this is deprecated
            image_mapping = ImageCropMetadata(
                parent_image=i.image_name,
                parent_image_id=i.image_id,
                parent_label_id=label_id,
                cropped_image=slice_path_jpg.name,
                cropped_image_id=aI.image_id,
                bbox=[int(cutout_box.bounds[0]), int(cutout_box.bounds[1]), int(cutout_box.bounds[2]),
                      int(cutout_box.bounds[3])],  # Creates a Shapely bounding box
                local_coordinate=pI.keypoints,  # Example point inside crop
                global_coordinate=label.keypoints  # Original image point
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
    crop boxes out of an image using absolute coordindates

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

            if empty == False and max_bbox_area < 5000:
                reject = True

            # sliced_im = PIL.Image.fromarray(sliced)

            # TODO put this somewhere else
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
                # sliced_im.save(slice_path_tiff, 'TIFF', compression=None)
                # sliced_im.save(slice_path_png )

    return images_without_objects, images_with_objects, annotations


def include_empty_frac(frac) -> bool:
    if frac == False:
        return frac
    return np.random.rand() < frac

class RasterCropper():

    def __init__(self,
                 hi: AnnotatedImage,
                 rasters: List[shapely.Polygon],
                 full_image_path: Path,
                 output_path: Path,
                 dataset_name: str = DATA_SET_NAME,
                 include_empty: float = 0.0,
                 edge_blackout=True,
                 sample_strategy=SpatialSampleStrategy.RANDOM):

        self.empty_selection_strategy = sample_strategy
        self.annotated_images: typing.List[AnnotatedImage] = []
        self.image_paths: typing.List[Path] = []
        self.closest_pairs: typing.List[dict] = []
        self.gdf_empty_rasters: gpd.GeoDataFrame = gpd.GeoDataFrame(columns=['geometry'])
        self.empty_rasters: typing.Optional[typing.List[shapely.Polygon]] = None
        self.occupied_rasters: typing.Optional[typing.List[shapely.Polygon]] = None
        self.partial_empty_rasters: typing.Optional[typing.List[shapely.Polygon]] = None

        self.hi = hi
        self.rasters = rasters
        self.full_image_path = full_image_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.include_empty = include_empty
        self.edge_blackout = edge_blackout


class RasterCropperBoxes(RasterCropper):
    """
    Crop Annotations from a rasterized image into smaller images based on the provided rasters (shapely polygons).
    """

    def __init__(self, hi: AnnotatedImage, rasters: List[shapely.Polygon], full_image_path: Path, output_path: Path,
                 dataset_name: str = DATA_SET_NAME, include_empty: float = 0.0, edge_blackout=True,
                 sample_strategy=SpatialSampleStrategy.RANDOM):
        self.masks = {}

        super().__init__(hi, rasters, full_image_path, output_path, dataset_name, include_empty, edge_blackout,
                         sample_strategy)


    def crop_out_images(self):
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

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_path_images = self.output_path
        output_path_images.mkdir(parents=True, exist_ok=True)

        image = PIL.Image.open(self.full_image_path)
        # imr = np.array(image)

        # Convert to string if you need a string representation
        annotated_images: List[AnnotatedImage] = []
        image_paths: typing.List[Path] = []
        empty_rasters: List[shapely.Polygon] = []
        partial_empty_rasters: List[shapely.Polygon] = []
        occupied_rasters: List[shapely.Polygon] = []

        # sliding window of the image
        for raster_id, pol in enumerate(self.rasters):
            assert isinstance(pol, shapely.Polygon)
            minx, miny, maxx, maxy = pol.bounds
            slice_width = maxx - minx
            slice_height = maxy - miny
            full_slice_labels = []
            partial_slice_labels = []

            ## TODO extract this to another function
            empty = True
            partial = False
            reject = False
            max_bbox_area = 0
            one_good_box = False

            masks = []  # for bordering bounding boxes, those are blacked out, the masks are saved here to remove keypoints later
            sliced_image = image.crop(pol.bounds)

            filename = str(Path(self.hi.image_name).stem)

            # TODO this is a bit messy, I want to get tiles which are completely empty, partially empty and occupied
            # Right now if the first label
            for annotation in self.hi.labels:

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

                    if is_bbox and annotation.bbox_polygon.within(pol):
                        logger.info(f"Box is completely within the sliding window, annotation id: {annotation.id}")
                        # the box is completely within the sliding window
                        one_good_box = True
                        empty = False
                        partial = False
                        # Translate the coordinates of the inner polygon
                        translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
                        # Create a new polygon with the translated coordinates
                        translated_inner_polygon = Polygon(translated_coords)
                        il = ImageLabel(
                            id=str(uuid.uuid4()),
                            class_name=annotation.class_name,
                            bbox=[int(x) for x in translated_inner_polygon.bounds],
                            attributes=annotation.attributes,
                        )
                        full_slice_labels.append(il)

                        xx, yy = pol.exterior.coords.xy
                        slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                        if slice_path_jpg.name == "FMO03___DJI_0514_x3200_y2560.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                            pass  # TODO do not commit

                    elif is_bbox and annotation.bbox_polygon.intersects(pol):
                        logger.info(f"Box is partially in the sliding window, annotation id: {annotation.id}")

                        # the box is partially within the sliding window
                        empty = False
                        partial = True
                        # First, get the intersection of the bbox with the sliding window
                        intersection = annotation.bbox_polygon.intersection(pol)

                        # If intersection is valid, use it; otherwise skip this annotation
                        if intersection.is_empty or not hasattr(intersection, 'bounds'):
                            logger.warning(f"Invalid intersection for annotation {annotation.id}, skipping")
                            continue

                        # Translate the intersection coordinates to local window coordinates
                        int_minx, int_miny, int_maxx, int_maxy = [int(x) for x in intersection.bounds]
                        translated_coords = [
                            (int_minx - minx, int_miny - miny),
                            (int_maxx - minx, int_miny - miny),
                            (int_maxx - minx, int_maxy - miny),
                            (int_minx - minx, int_maxy - miny)
                        ]

                        if int_minx == 3200 and int_miny == 2560:
                            logger.warning(f"Intersection coordinates: {translated_coords} for annotation {annotation.id}")

                        translated_inner_polygon = Polygon(translated_coords)
                        il = ImageLabel(
                            id=str(uuid.uuid4()),
                            class_name=annotation.class_name,
                            bbox=[int(x) for x in translated_inner_polygon.bounds],
                            attributes=annotation.attributes,
                        )
                        il.attributes["edge_partial"] = True
                        partial_slice_labels.append(il)


                    else:
                        raise ValueError("At least on condition should be true, either bbox_polygon or polygon_s")

                    # If an iguana is in the middle of the image but under a rock it can be a partial iguana
                    if il.attributes.get("edge_partial", False):
                        # logger.info(f"Box or polygon is not completly within the sliding window {annotation.id}")
                        # Translate the coordinates of the inner polygon
                        if self.edge_blackout:
                            masks.append(il.bbox_polygon)

                            sliced_image = blackout_bbox(sliced_image,
                                                   bbox_x1y1x2y2=il.x1y1x2y2)


                else:
                    # logger.info(f"Box or polygon is not within the sliding window {annotation.id}")
                    pass

                self.masks[annotation.id] = masks

            if empty == False and max_bbox_area < 5000: # TODO this should depend on the portion which is outside the sliding window
                # reject = True
                # logger.warning(f"Should Rejecting image, label very small {hi.image_name} because of max_bbox_area {max_bbox_area}")
                pass

            if empty: # box is completely empty
                empty_rasters.append(pol)
            elif partial and not one_good_box: # box contains partial iguanas and no single full one
                partial_empty_rasters.append(pol)
            elif one_good_box:
                occupied_rasters.append(pol)
            else:
                ValueError(f"This should not happen, the box is neither empty nor occupied {self.hi.image_name} with {len(full_slice_labels)} labels")

            # Save the image if it is not empty or if we want to include empty images
            if one_good_box:
                xx, yy = pol.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg.name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=full_slice_labels,
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

                logger.info(f"Saving occupied raster {slice_path_jpg} with {len(full_slice_labels)} labels")
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)


        ### Loooking for empties now!
        closest_pairs = []
        # for each occupied Raster find the closest empty raster and add it to the list

        gdf_empty_rasters = gpd.GeoDataFrame(geometry=empty_rasters)
        closest_empty = None

        # sample empty raster which are nearby the found iguanas
        for oR in occupied_rasters:
            # query for the closest empty raster

            if self.empty_selection_strategy == SpatialSampleStrategy.RANDOM:
                """ 
                # Select a random empty raster from the list 
                """
                if not gdf_empty_rasters.empty:
                    min_idx = np.random.randint(0, len(gdf_empty_rasters))

                    # Get the row and its index
                    closest_empty = gdf_empty_rasters.iloc[min_idx]
                    sampled_idx = gdf_empty_rasters.index[min_idx]

                    # Remove the row
                    gdf_empty_rasters = gdf_empty_rasters.drop(sampled_idx)  # This is the index of the sampled row
                    distance = oR.distance(closest_empty.geometry)
                    closest_pairs.append({
                        'occupied_raster': oR,
                        'closest_empty': closest_empty.geometry,
                        'distance': distance
                    })

            elif self.empty_selection_strategy == SpatialSampleStrategy.NEAREST:
                # Calculate distances from this occupied raster to all empty rasters
                distances = gdf_empty_rasters.distance(oR)

                # Find the index of the minimum distance
                if not distances.empty:
                    min_idx = distances.idxmin()
                    closest_empty = gdf_empty_rasters.loc[min_idx]
                    distance = distances.loc[min_idx]

                    # Add to our results
                    closest_pairs.append({
                        'occupied_raster': oR,
                        'closest_empty': closest_empty.geometry,
                        'distance': distance
                    })

                gdf_empty_rasters.drop(min_idx, inplace=True)

            else:
                raise WrongSpatialSamplingStrategy(f"Use any of {SpatialSampleStrategy}")


            if closest_empty is not None:
                sliced_image = image.crop(closest_empty.geometry.bounds)
                xx, yy = closest_empty.geometry.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg.name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)

                minx, miny, maxx, maxy = closest_empty.geometry.bounds
                slice_width = maxx - minx
                slice_height = maxy - miny

                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=[],
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

        # Now save these results to a file
        self.occupied_rasters = occupied_rasters
        self.empty_rasters = empty_rasters
        self.partial_empty_rasters = partial_empty_rasters
        self.closest_pairs = closest_pairs
        self.annotated_images = annotated_images
        self.image_paths = image_paths

        return annotated_images, image_paths

class RasterCropperPoints(RasterCropper):
    """
    Crop Point Annotations from a rasterized image into smaller images based
    on the provided rasters (shapely polygons).
    """

    def __init__(self, hi: AnnotatedImage, rasters: List[shapely.Polygon], full_image_path: Path, output_path: Path,
                 dataset_name: str = DATA_SET_NAME, include_empty: float = 0.0, edge_blackout=True,
                 sample_strategy=SpatialSampleStrategy.RANDOM):

        super().__init__(hi, rasters, full_image_path, output_path, dataset_name, include_empty, edge_blackout,
                         sample_strategy)


    def crop_out_images(self, masks):
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

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_path_images = self.output_path
        output_path_images.mkdir(parents=True, exist_ok=True)

        image = PIL.Image.open(self.full_image_path)
        # imr = np.array(image)

        # Convert to string if you need a string representation
        annotated_images: List[AnnotatedImage] = []
        image_paths: typing.List[Path] = []
        empty_rasters: List[shapely.Polygon] = []
        partial_empty_rasters: List[shapely.Polygon] = []
        occupied_rasters: List[shapely.Polygon] = []
        # slice the image in tiles

        # sliding window of the image
        for raster_id, pol in enumerate(self.rasters):
            assert isinstance(pol, shapely.Polygon)
            minx, miny, maxx, maxy = pol.bounds
            slice_width = maxx - minx
            slice_height = maxy - miny
            full_slice_labels = []
            partial_slice_labels = []

            ## TODO extract this to another function
            empty = True
            partial = False
            reject = False
            max_bbox_area = 0

            sliced_image = image.crop(pol.bounds)

            filename = str(Path(self.hi.image_name).stem)

            # TODO this is a bit messy, I want to get tiles which are completely empty, partially empty and occupied
            # Right now if the first label
            for annotation in self.hi.labels:

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



                    # Process the keypoints
                    if is_keypoint and annotation.keypoints[0].coordinate.within(pol):
                        # keypoint is within the sliding window, project the keypoints to the sliding window and keep that
                        empty = False
                        box_keypoints = []
                        for k in annotation.keypoints:
                            kc = copy.deepcopy(k)
                            kc.x = int(k.x - minx)
                            kc.y = int(k.y - miny)
                            box_keypoints.append(kc)

                        # translated_keypoints = [Keypoint(x=int(k.x - minx), y=int(k.y - miny)) for k in box.keypoints]

                        il = ImageLabel(
                            id=annotation.id,
                            class_name=annotation.class_name,
                            keypoints=box_keypoints,
                            attributes=annotation.attributes,
                        )

                        full_slice_labels.append(il)

                    # is a keypoint but outside the sliding window
                    elif is_keypoint and not annotation.keypoints[0].coordinate.within(pol):
                        # The keypoint is outside of the sliding window
                        # So far we do not do anything here
                        pass


                else:
                    # logger.info(f"Box or polygon is not within the sliding window {annotation.id}")
                    pass

            for m in masks.values():
                # remove points which are in the mask
                full_slice_labels = [sl for sl in full_slice_labels if
                                     isinstance(sl.incenter_centroid, shapely.Point) and sl.incenter_centroid.within(
                                         m) == False]

            if empty == False and max_bbox_area < 5000: # TODO this should depend on the portion which is outside the sliding window
                # reject = True
                # logger.warning(f"Should Rejecting image, label very small {hi.image_name} because of max_bbox_area {max_bbox_area}")
                pass

            if empty: # box is completely empty
                empty_rasters.append(pol)
            # elif partial: # box contains partial iguanas and no single full one
            #     partial_empty_rasters.append(pol)
            elif not empty:
                occupied_rasters.append(pol)
            else:
                ValueError(f"This should not happen, the box is neither empty nor occupied {self.hi.image_name} with {len(full_slice_labels)} labels")

            # Save the image if it is not empty or if we want to include empty images
            if not empty:
                xx, yy = pol.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg.name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=full_slice_labels,
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

                logger.info(f"Saving occupied raster {slice_path_jpg} with {len(full_slice_labels)} labels")
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)
        ### Loooking for empties now!
        closest_pairs = []
        # for each occupied Raster find the closest empty raster and add it to the list

        gdf_empty_rasters = gpd.GeoDataFrame(geometry=empty_rasters)

        # sample empty raster which are nearby the found iguanas
        for oR in occupied_rasters:
            # query for the closest empty raster

            # Calculate distances from this occupied raster to all empty rasters
            distances = gdf_empty_rasters.distance(oR)

            # Find the index of the minimum distance
            if not distances.empty:
                min_idx = distances.idxmin()
                closest_empty = gdf_empty_rasters.loc[min_idx]
                distance = distances.loc[min_idx]

                # Add to our results
                closest_pairs.append({
                    'occupied_raster': oR,
                    'closest_empty': closest_empty.geometry,
                    'distance': distance
                })

                sliced_image = image.crop(closest_empty.geometry.bounds)
                xx, yy = closest_empty.geometry.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)

                minx, miny, maxx, maxy = closest_empty.geometry.bounds
                slice_width = maxx - minx
                slice_height = maxy - miny

                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=[],
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

                # TODO remove that raster from
                gdf_empty_rasters.drop(min_idx, inplace=True)

        # Now save these results to a file
        self.occupied_rasters = occupied_rasters
        self.empty_rasters = empty_rasters
        self.partial_empty_rasters = partial_empty_rasters
        self.closest_pairs = closest_pairs
        self.annotated_images = annotated_images
        self.image_paths = image_paths

        return annotated_images, image_paths

class RasterCropperPolygon(RasterCropper):
    """
    Crop Point Annotations from a rasterized image into smaller images based
    on the provided rasters (shapely polygons).
    """

    def __init__(self, hi: AnnotatedImage, rasters: List[shapely.Polygon], full_image_path: Path, output_path: Path,
                 dataset_name: str = DATA_SET_NAME, include_empty: float = 0.0, edge_blackout=True,
                 sample_strategy=SpatialSampleStrategy.RANDOM):

        super().__init__(hi, rasters, full_image_path, output_path, dataset_name, include_empty, edge_blackout,
                         sample_strategy)


    def crop_out_images(self):
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

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_path_images = self.output_path
        output_path_images.mkdir(parents=True, exist_ok=True)

        image = PIL.Image.open(self.full_image_path)
        # imr = np.array(image)

        # Convert to string if you need a string representation
        annotated_images: List[AnnotatedImage] = []
        image_paths: typing.List[Path] = []
        empty_rasters: List[shapely.Polygon] = []
        partial_empty_rasters: List[shapely.Polygon] = []
        occupied_rasters: List[shapely.Polygon] = []
        # slice the image in tiles

        # sliding window of the image
        for raster_id, pol in enumerate(self.rasters):
            assert isinstance(pol, shapely.Polygon)
            minx, miny, maxx, maxy = pol.bounds
            slice_width = maxx - minx
            slice_height = maxy - miny
            full_slice_labels = []
            partial_slice_labels = []

            ## TODO extract this to another function
            empty = True
            partial = False
            reject = False
            max_bbox_area = 0
            one_good_box = False

            masks = []  # for bordering bounding boxes, those are blacked out, the masks are saved here to remove keypoints later
            sliced_image = image.crop(pol.bounds)

            filename = str(Path(self.hi.image_name).stem)

            # TODO this is a bit messy, I want to get tiles which are completely empty, partially empty and occupied
            # Right now if the first label
            for annotation in self.hi.labels:

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

                    if annotation.bbox_polygon.within(pol):
                        logger.info(f"Box is completely within the sliding window, annotation id: {annotation.id}")
                        # the box is completely within the sliding window
                        one_good_box = True
                        empty = False
                        partial = False
                        # Translate the coordinates of the inner polygon
                        translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
                        # Create a new polygon with the translated coordinates
                        translated_inner_polygon = Polygon(translated_coords)
                        il = ImageLabel(
                            id=str(uuid.uuid4()),
                            class_name=annotation.class_name,
                            bbox=[int(x) for x in translated_inner_polygon.bounds],
                            attributes=annotation.attributes,
                        )
                        full_slice_labels.append(il)

                        xx, yy = pol.exterior.coords.xy
                        slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                        if slice_path_jpg.name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                            pass  # TODO do not commit

                    elif is_bbox and annotation.bbox_polygon.intersects(pol):
                        logger.info(f"Box is partially in the sliding window, annotation id: {annotation.id}")

                        # the box is partially within the sliding window
                        empty = False
                        partial = True
                        # First, get the intersection of the bbox with the sliding window
                        intersection = annotation.bbox_polygon.intersection(pol)

                        # If intersection is valid, use it; otherwise skip this annotation
                        if intersection.is_empty or not hasattr(intersection, 'bounds'):
                            logger.warning(f"Invalid intersection for annotation {annotation.id}, skipping")
                            continue

                        # Translate the intersection coordinates to local window coordinates
                        int_minx, int_miny, int_maxx, int_maxy = intersection.bounds
                        translated_coords = [
                            (int_minx - minx, int_miny - miny),
                            (int_maxx - minx, int_miny - miny),
                            (int_maxx - minx, int_maxy - miny),
                            (int_minx - minx, int_maxy - miny)
                        ]

                        translated_inner_polygon = Polygon(translated_coords)
                        il = ImageLabel(
                            id=str(uuid.uuid4()),
                            class_name=annotation.class_name,
                            bbox=[int(x) for x in translated_inner_polygon.bounds],
                            attributes=annotation.attributes,
                        )
                        il.attributes["edge_partial"] = True
                        partial_slice_labels.append(il)

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
                            attributes=annotation.attributes,

                        )
                        full_slice_labels.append(il)

                    # Process the keypoints
                    elif is_keypoint and annotation.keypoints[0].coordinate.within(pol):
                        # keypoint is within the sliding window, project the keypoints to the sliding window and keep that
                        empty = False
                        box_keypoints = []
                        for k in annotation.keypoints:
                            kc = copy.deepcopy(k)
                            kc.x = int(k.x - minx)
                            kc.y = int(k.y - miny)
                            box_keypoints.append(kc)

                        # translated_keypoints = [Keypoint(x=int(k.x - minx), y=int(k.y - miny)) for k in box.keypoints]

                        il = ImageLabel(
                            id=annotation.id,
                            class_name=annotation.class_name,
                            keypoints=box_keypoints,
                            attributes=annotation.attributes,
                        )

                        full_slice_labels.append(il)

                    # is a keypoint but outside the sliding window
                    elif is_keypoint and not annotation.keypoints[0].coordinate.within(pol):
                        # The keypoint is outside of the sliding window
                        # So far we do not do anything here
                        pass


                    if il.attributes.get("edge_partial", False):
                        # logger.info(f"Box or polygon is not completly within the sliding window {annotation.id}")
                        # Translate the coordinates of the inner polygon
                        if self.edge_blackout:
                            masks.append(il.bbox_polygon)

                            sliced_image = blackout_bbox(sliced_image,
                                                   bbox_x1y1x2y2=il.x1y1x2y2)

                        else:
                            raise NotImplementedError("Redraw the box or polygon to fit the sliding window")
                else:
                    # logger.info(f"Box or polygon is not within the sliding window {annotation.id}")
                    pass

            for m in masks:
                # remove points which are in the mask
                full_slice_labels = [sl for sl in full_slice_labels if
                                     isinstance(sl.incenter_centroid, shapely.Point) and sl.incenter_centroid.within(
                                         m) == False]

            if empty == False and max_bbox_area < 5000: # TODO this should depend on the portion which is outside the sliding window
                # reject = True
                # logger.warning(f"Should Rejecting image, label very small {hi.image_name} because of max_bbox_area {max_bbox_area}")
                pass

            if empty: # box is completely empty
                empty_rasters.append(pol)
            elif partial and not one_good_box: # box contains partial iguanas and no single full one
                partial_empty_rasters.append(pol)
            elif one_good_box:
                occupied_rasters.append(pol)
            else:
                ValueError(f"This should not happen, the box is neither empty nor occupied {self.hi.image_name} with {len(full_slice_labels)} labels")

            # Save the image if it is not empty or if we want to include empty images
            if one_good_box:
                xx, yy = pol.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg.name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=full_slice_labels,
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

                logger.info(f"Saving occupied raster {slice_path_jpg} with {len(full_slice_labels)} labels")
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)
        ### Loooking for empties now!
        closest_pairs = []
        # for each occupied Raster find the closest empty raster and add it to the list

        gdf_empty_rasters = gpd.GeoDataFrame(geometry=empty_rasters)

        # sample empty raster which are nearby the found iguanas
        for oR in occupied_rasters:
            # query for the closest empty raster

            # Calculate distances from this occupied raster to all empty rasters
            distances = gdf_empty_rasters.distance(oR)

            # Find the index of the minimum distance
            if not distances.empty:
                min_idx = distances.idxmin()
                closest_empty = gdf_empty_rasters.loc[min_idx]
                distance = distances.loc[min_idx]

                # Add to our results
                closest_pairs.append({
                    'occupied_raster': oR,
                    'closest_empty': closest_empty.geometry,
                    'distance': distance
                })

                sliced_image = image.crop(closest_empty.geometry.bounds)
                xx, yy = closest_empty.geometry.exterior.coords.xy
                slice_path_jpg = self.output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
                if slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg" or slice_path_jpg == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    pass # TODO do not commit
                sliced_im = sliced_image.convert("RGB")
                sliced_im.save(slice_path_jpg)
                image_paths.append(slice_path_jpg)

                minx, miny, maxx, maxy = closest_empty.geometry.bounds
                slice_width = maxx - minx
                slice_height = maxy - miny

                im = AnnotatedImage(
                    image_id=str(uuid.uuid4()),
                    dataset_name=self.dataset_name if self.dataset_name else DATA_SET_NAME,
                    image_name=slice_path_jpg.name,
                    labels=[],
                    width=int(slice_width),
                    height=int(slice_height))

                annotated_images.append(im)

                # TODO remove that raster from
                gdf_empty_rasters.drop(min_idx, inplace=True)

        # Now save these results to a file
        self.occupied_rasters = occupied_rasters
        self.empty_rasters = empty_rasters
        self.partial_empty_rasters = partial_empty_rasters
        self.closest_pairs = closest_pairs
        self.annotated_images = annotated_images
        self.image_paths = image_paths

        return annotated_images, image_paths




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
    if slice_width is None:
        pass
    # Compute the step sizes
    step_x = slice_width - overlap
    step_y = slice_height - overlap

    # Calculate padding so that width and height become multiples of step_x and step_y respectively
    padding_x = (step_x - (width % step_x)) % step_x
    padding_y = (step_y - (height % step_y)) % step_y

    new_width = width + padding_x
    new_height = height + padding_y

    if not padded_image_path.exists():
        # Create a new image with the padded dimensions
        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))  # black background
        padded_image.paste(image, (0, 0))
        padded_image.save(padded_image_path)

        logger.info(f"Padded image saved to {padded_image_path} with size: {new_width}x{new_height}")
    else:
        logger.info(f"Padded image already exists at {padded_image_path}, skipping padding.")
    return new_width, new_height


def sliced_predict_geotiff(geotiff_path: Path):
    """
    @deprecated
    :param geotiff_path:
    :return:
    """
    slicer = GeoSlicer(base_path=geotiff_path.parent, image_name=geotiff_path.name, x_size=5120, y_size=5120)
    tiles = slicer.slice_very_big_raster()
    print(tiles)


# TODO modify this so it can be integrated into the latter
def convert_image(image_path) -> Path:
    """
    convert an geospatial image to a jpg
    :param image_path:
    :return:
    """
    jpg_path = image_path.with_suffix(".jpg")
    with Image.open(image_path) as img:
        img.convert("RGB").save(jpg_path, "JPEG", quality=95)

    return jpg_path


def convert_tiles_to(tiles: typing.List[Path], format: ImageFormat, output_dir: Path) -> typing.Generator[
    Path, None, None]:
    """
    Convert a list of image tiles to a specified format. Either from geospatial
    to pixel coordinates or vice versa (geospatial logic not fully implemented here).

    :param tiles: A list of Path objects pointing to the input tiles.
    :param format: The desired output image format (e.g., ImageFormat.PNG).
    :param output_dir: The directory where converted images will be saved.
    """

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tiles: typing.List[Path] = []

    for tile in tiles:
        try:
            if not tile.exists():
                logger.error("Tile does not exist: %s", tile)
                continue

            # Build the output filename (e.g., tile_name.png)
            output_filename = f"{tile.stem}.{format.value.lower()}"
            out_path = output_dir / output_filename

            if out_path.exists():
                logger.warning(f"Output file already exists, skipping: {out_path}")
                continue

            # Open the image tile
            with Image.open(tile) as img:
                # If a world_file is provided, you can apply geospatial transformations here

                if format == ImageFormat.JPG:
                    img = img.convert("RGB")
                # Save the image in the desired format
                img.save(out_path, quality=95)

                output_tiles.append(out_path)

                yield out_path
        except Exception as e:
            logger.error(f"Failed to convert tile {tile}: {e}")


def remove_empty_tiles(tiles: typing.List[Path], threshold=0.7, empty_value=(0, 0, 0)) -> typing.List[Path]:
    """
    check amount of black pixels for each tile and remove the empty ones
    :param threshold: percent empty pixels
    :param tiles:
    :return:
    """
    assert 0 <= threshold <= 1

    non_empty_tiles = []
    for tile in tiles:
        with Image.open(tile) as img:
            img = img.convert("RGB")  # remove the possible alpha channel
            img_array = np.array(img)
            empty_pixels = np.count_nonzero(np.all(img_array == empty_value, axis=2))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            empty_ratio = empty_pixels / total_pixels
            if empty_ratio < threshold:
                non_empty_tiles.append(tile)

    return non_empty_tiles
