import copy
import json
import typing
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import UUID

import numpy as np
import pandas as pd
import shapely
from loguru import logger
from pydantic import BaseModel, Field
from scipy.spatial import Voronoi
from shapely import Polygon

from com.biospheredata.types.status import LabelingStatus
from com.biospheredata.types.decorators import deprecated_warning


class PolygonNotSetError(ValueError):
    pass


def chebyshev_center(coords: list) -> shapely.Point:
    """
    find the Chebyshev center of a polygon, which is not the centroid but the point that is furthest away from the edges
    In a Snake shaped polygon, the centroid would be in the middle of the snake, but the Chebyshev center would be at the head of the snake

    :param coords:
    :return:
    """
    # Compute the Voronoi diagram for the vertices of the triangles
    try:
        points = np.array(coords)
        vor = Voronoi(points)

        polygon = Polygon(coords)

        # Find the Voronoi vertices inside the polygon
        voronoi_vertices = [shapely.Point(vertex) for vertex in vor.vertices if polygon.contains(shapely.Point(vertex))]

        # Calculate the distance to the polygon's edges for each vertex
        max_distance = 0
        center_point = None
        for vertex in voronoi_vertices:
            distance = vertex.distance(polygon.exterior)
            if distance > max_distance:
                max_distance = distance
                center_point = vertex

        center_point = shapely.Point(int(round(center_point.x)), int(round(center_point.y)))
        return center_point
    except Exception as e:
        logger.error(f"Error in chebyshev_center: {e} with coords {coords}")
        return None




class LabelClass(BaseModel):
    class_id: str
    parent_class_id: Optional[str] = None
    class_name: str
    class_type: str
    color: str
    norder: float
    icon_url: Optional[str] = None
    attributes: List[str] = []
    description: Optional[str] = None
    use_description_as_prompt: Optional[bool] = False


class Keypoint(BaseModel):
    x: int
    y: int
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    norder: int = Field(default=0)
    visible: bool = Field(default=True)
    created_by: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    updated_by: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    create_date: Optional[datetime] = Field(default=datetime.now())
    update_date: Optional[datetime] = Field(default=datetime.now())
    keypoint_class_id: str = Field(alias='keypoint_class_id')

    @property
    def coordinate(self):
        return shapely.Point(self.x, self.y)


class ImageLabel(BaseModel):
    """
    A label for a single object in an image
    """
    id: typing.Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()), alias='id')
    class_name: str = Field(alias='class_name')
    bbox: Optional[List[typing.Union[int, float]]] = Field(default=None)
    polygon: Optional[List[typing.Tuple[typing.Union[int, float], typing.Union[int, float]]]] = Field(default=None)  # A list of points=[x,y] that make up the polygon
    mask: Optional[List[typing.Union[int, float]]] = Field(default=None)
    z_index: Optional[int] = Field(default=0)
    attributes: dict = Field(default_factory=dict)
    keypoints: Optional[List[Keypoint]] = Field(default=None)

    # incenter: Optional[List[int]] = None # The point which is either the centroid or the nearest point to the centroid that is withing the shape

    @property
    def x1y1x2y2(self):
        """
        @deprecated, a bbox could be anything
        :return:
        """
        if self.bbox is not None:
            return self.bbox
        elif self.polygon is not None:
            return self.polygon_s.bounds
        else:
            return None

    @property
    def bbox_polygon(self) -> Optional[shapely.Polygon]:
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            poly = shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            return poly
        else:
            return None

    @property
    def width(self) -> int:
        if self.bbox is not None:
            return self.bbox[2] - self.bbox[0]
        else:
            return 0

    @property
    def height(self) -> int:
        if self.bbox is not None:
            return self.bbox[3] - self.bbox[1]
        else:
            return 0

    @property
    def centroid(self) -> shapely.Point:
        """
        deprecated
        :return:
        """
        if self.bbox is not None:
            return self.incenter_centroid
        if self.keypoints is not None:
            return shapely.Point(self.keypoints[0].x, self.keypoints[0].y)

    @property
    def get_mask(self) -> typing.List[float] :
        if self.mask is None or len(self.mask) == 0:
            if self.polygon is None:
                raise PolygonNotSetError("the polygon is None, can't create a mask from it")
            coords_list = [coord for x, y in self.polygon_s.exterior.coords for coord in (x, y)]

            return coords_list
        else:
            return self.mask

    @property
    def incenter_centroid(self) -> shapely.Point:
        """
        Find the point within the polygon that is closest to the centroid
        :return:
        """
        if self.polygon is not None and len(self.polygon) > 0:
            center_point = chebyshev_center(self.polygon)

        elif self.keypoints is not None and len(self.keypoints) > 0:
            if self.keypoints[0].keypoint_class_id == 'ed18e0f9-095f-46ff-bc95-febf4a53f0ff':
                center_point = shapely.Point(self.keypoints[0].x, self.keypoints[0].y)
            else:
                logger.warning("Not properly implemented yet. It is a bit of a hack")
                # TODO implement this with the keypoint schema
                return shapely.Point(self.keypoints[0].x, self.keypoints[0].y)
        elif self.bbox_polygon is not None:
            center_point = self.bbox_polygon.centroid
        else:
            center_point = None

        return center_point

    @property
    def polygon_s(self) -> shapely.Polygon:
        """ shapely represantation of the polygon """

        if self.polygon is not None:
            return Polygon(self.polygon)
        else:
            return None

    @bbox_polygon.setter
    def bbox_polygon(self, value):
        self._bbox_polygon = value
        self.bbox = [int(value.bounds[0]), int(value.bounds[1]), int(value.bounds[2]), int(value.bounds[3])]

    def __hash__(self) -> int:
        # Convert to JSON string (handles nested structures)
        json_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hash(json_str)
        #
        # return self.id


class PredictedImageLabel(ImageLabel):
    score: float = Field(..., alias='score', description="The confidence score for the prediction")
    kind: Optional[str] = Field(..., alias='kind', description="The kind of prediction, e.g. 'fp' or 'fn' or 'ground_truth'")


class ImageLabelCollection(BaseModel):
    image_name: str = Field(alias='image_name', description="Name of the image file")
    image_id: typing.Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()), alias='image_id')

    labels: List[typing.Union[ImageLabel, PredictedImageLabel]] = Field(...,
                                                                        description="A list of labels on the image. Can be either ImageLabel or PredictedImageLabel")
    width: int
    height: int

    def denormalise(self):
        """
        Normalise the labels to the image size
        :return:
        """
        for label in self.labels:
            if label.bbox is not None:
                label.bbox = [int(label.bbox[0] * self.width), int(label.bbox[1] * self.height),
                              int(label.bbox[2] * self.width), int(label.bbox[3] * self.height)]

            if label.polygon is not None:
                label.polygon = [(int(x * self.width), int(y * self.height)) for x, y in label.polygon]

            if label.mask is not None:
                label.mask = [int(coord * self.width) for coord in label.mask]

        return self

    def normalise(self):
        """
        Normalise the labels to the image size
        :return:
        """
        for label in self.labels:
            if label.bbox is not None:
                label.bbox = [label.bbox[0] / self.width, label.bbox[1] / self.height,
                              label.bbox[2] / self.width, label.bbox[3] / self.height]

            if label.polygon is not None:
                label.polygon = [(x / self.width, y / self.height) for x, y in label.polygon]

            if label.mask is not None:
                label.mask = [int(coord / self.width) for coord in label.mask]

        return self

    def save(self, file_path: Path):
        with open(file_path, 'w') as json_file:
            # Serialize the list of Pydantic objects to a list of dictionaries
            json_file.write(self.model_dump_json())


class AnnotatedImage(ImageLabelCollection):
    dataset_name: Optional[str] = Field(default=None, alias='dataset_name')
    ds_image_name: Optional[str] = Field(default=None)

    image_status: Optional[str | LabelingStatus] = "DONE"
    tags: Optional[List[str]] = Field(default=list(), alias='tags')

    image_mode: Optional[str] = None




class KeypointClass(BaseModel):
    keypoint_class_id: str
    keypoint_class_name: str
    norder: int
    editor_x: Optional[float] = None
    editor_y: Optional[float] = None
    max_points: int
    min_points: int


class KeypointSchema(BaseModel):
    keypoint_schema_id: str
    keypoint_schema_name: str
    keypoint_schema_type: str
    associated_label_classes: List[str]
    keypoint_classes: List[KeypointClass]
    keypoint_skeleton: List  # Define this as List[Any] if you expect various data types in the skeleton


class TagGroup(BaseModel):
    group_id: str
    group_name: str
    group_type: str
    tags: List[str]


# TODO check if this format is correct
class Attribute(BaseModel):
    name: str
    type: str
    values: List




class HastyAnnotationV2(BaseModel):
    project_name: str = Field(alias='project_name')
    create_date: datetime = Field(default=datetime.now())
    export_format_version: str = Field(alias='export_format_version', default="1.1")
    export_date: datetime = Field(default=datetime.now())
    label_classes: List[LabelClass]
    keypoint_schemas: List[KeypointSchema] = Field(default=[])
    tag_groups: List[TagGroup] = Field(default=[])
    images: List[typing.Union[ImageLabelCollection, AnnotatedImage]] = Field(default=[])
    attributes: List[Attribute] = Field(default=[])

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as json_file:
            # Serialize the list of Pydantic objects to a list of dictionaries
            json_file.write(self.model_dump_json(indent=2))

    def name2id(self):
        """
        convert the class names to class ids
        """

        label_classes = {label.class_name: idx for idx, label in enumerate(self.label_classes, start=1)}
        return label_classes

    def label_count(self) -> int:
        """
        Count the total number of labels in the project.

        :return: Total number of labels across all images.
        """
        return sum(len(image.labels) for image in self.images)

    def dataset_statistics(self):
        """
        Print statistics about the datasets in the project.
        """
        dataset_stats = {}
        for image in self.images:
            dataset_name = image.dataset_name if isinstance(image, AnnotatedImage) else "unknown"
            if dataset_name not in dataset_stats:
                dataset_stats[dataset_name] = {
                    'num_images': 0,
                    'num_labels': 0,
                    'num_point_labels': 0,
                    'num_box_labels': 0
                }
            dataset_stats[dataset_name]['num_images'] += 1

            if len(image.labels) > 0:

                num_boxes = sum(1 for label in image.labels if label.bbox_polygon is not None)
                num_points = sum(1 for label in image.labels if label.keypoints is not None and len(label.keypoints) > 0)
                dataset_stats[dataset_name]['num_box_labels'] += num_boxes
                dataset_stats[dataset_name]['num_point_labels'] += num_points
                dataset_stats[dataset_name]['num_labels'] += len(image.labels)

        for dataset, stats in dataset_stats.items():
            logger.info(f"Dataset: {dataset}, Images: {stats['num_images']}, Labels: {stats['num_labels']}")

        return dataset_stats

    def delete_dataset(self, dataset_name: str):
        """
        delete a dataset within the project completely
        :param dataset_name:
        :return:
        """
        initial_count = len(self.images)

        # Filter out images from the specified dataset
        self.images = [image for image in self.images if image.dataset_name != dataset_name]

        deleted_count = initial_count - len(self.images)
        logger.info(f"Deleted {deleted_count} images from dataset '{dataset_name}'")

    def rename_dataset(self, dataset_old: str, dataset_new: str):
        """
        delete a dataset within the project completely
        :param dataset_name:
        :return:
        """
        initial_count = len(self.images)

        for i in range(len(self.images)):
            if self.images[i].dataset_name == dataset_old:
                self.images[i].dataset_name = dataset_new


    def rename_label_class(self, k, v):
        """
        Rename a label class in the project.

        :param k: The class name to rename.
        :param v: The new class name.
        """

        for image in self.images:
            for label in image.labels:
                if label.class_name == k:
                    label.class_name = v

    @staticmethod
    def from_file(file_path: Path):
        """
        Load a HastyAnnotationV2 object from a file
        :param file_path:
        :return:
        """
        with open(file_path, mode="r") as f:
            data = json.load(f)
            hA = HastyAnnotationV2(**data)
        return hA

    def get_flat_df(self):
        return get_flat_df(self)

    def add_labels_to_image(self, image_id: str, dataset_name: str, label: ImageLabel):
        """
        add a label to an image in the project
        :param image_id:
        :param dataset_name:
        :param label:
        :return:
        """

        # find the image in queston

        # add the label to the

        for image in self.images:
            if image.image_id == image_id and image.dataset_name == dataset_name:
                if isinstance(image, AnnotatedImage):
                    image.labels.append(label)
                elif isinstance(image, ImageLabelCollection):
                    image.labels.append(label)
                else:
                    raise ValueError("Image type not supported")
                return
        raise ValueError("image id not found in project")

    def add_labels_to_image_by_image_name(self, image_name: str, dataset_name: str, label: ImageLabel):
        """
        add a label to an image in the project
        :param image_id:
        :param dataset_name:
        :param label:
        :return:
        """

        # find the image in queston

        # add the label to the

        for image in self.images:
            if image.image_name == image_name and image.dataset_name == dataset_name:
                if isinstance(image, AnnotatedImage):
                    image.labels.append(label)
                elif isinstance(image, ImageLabelCollection):
                    image.labels.append(label)
                else:
                    raise ValueError("Image type not supported")
                return
        raise ValueError("image id not found in project")

    def get_image_by_name(self, image_name: str) -> Optional[AnnotatedImage]:
        """
        Get an image by its name from the project.

        :param image_name: The name of the image to retrieve.
        :return: The AnnotatedImage object if found, otherwise None.
        """
        for image in self.images:
            if image.image_name == image_name:
                return image
        return None

    def get_image_by_id(self, id: str) -> Optional[AnnotatedImage]:
        """
        Get an image by its name from the project.

        :param image_name: The name of the image to retrieve.
        :return: The AnnotatedImage object if found, otherwise None.
        """
        for image in self.images:
            if image.image_id == id:
                return image
        return None


class HastyAnnotationV2_flat(BaseModel):
    project_name: str
    create_date: datetime
    export_format_version: str
    export_date: datetime
    label_classes: List[LabelClass]

    image_id: str | int = Field(default_factory=lambda: str(uuid.uuid4()), alias='image_id')
    image_name: str
    dataset_name: str
    ds_image_name: Optional[str] = None
    width: int
    height: int
    image_status: Optional[str] = "New"
    tags: List[str]
    # labels: List[ImageLabel]
    image_mode: Optional[str] = None

    # ImageLabel
    label_id: str
    class_name: str
    bbox: Optional[List[int]] = Field(None)
    mask: Optional[List[int]] = Field(default_factory=list)
    z_index: int
    ID: Optional[str] = None

    @property
    def x1y1x2y2(self):
        """
        deprecated, a bbox could be anything
        :return:
        """
        return self.bbox

    @property
    def bbox_polygon(self) -> shapely.Polygon:
        x1, y1, x2, y2 = self.bbox
        poly = shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return poly

    @property
    def centroid(self) -> shapely.Point:
        return self.bbox_polygon.centroid


def filter_by_class(hA: HastyAnnotationV2, class_names: Optional[List[str]] = None) -> HastyAnnotationV2:
    """
    remove any labels that are not in the class_names list
    :param hA:
    :param class_names:
    :return:
    """
    if class_names is None or len(class_names) == 0:
        return hA

    assert type(class_names) is list or type(class_names) is tuple, "class_names must be a list or tuple"

    if len(class_names) > 0:
        # Create a deep copy of the project to avoid modifying the original object
        filtered_project = copy.deepcopy(hA)

        for image in filtered_project.images:
            # Filter labels by class_name
            filtered_labels = [label for label in image.labels if label.class_name in class_names]
            image.labels = filtered_labels
        logger.info(f"Number of images after filtering: {len(filtered_project.images)}")
        return filtered_project
    else:
        return hA


def filter_by_image_tags(hA: HastyAnnotationV2, image_tags: Optional[List[str]] = None,
                         logic="and") -> HastyAnnotationV2:
    """
    Remove any images that do not contain the specified image tags.

    :param hA: HastyAnnotationV2 object containing images and labels.
    :param image_tags: List of tags to filter images by.
    :param logic: Logic for filtering, currently unused but can be extended.
    :return: Filtered HastyAnnotationV2 object.
    """
    if image_tags is None or len(image_tags) == 0:
        return hA

    assert isinstance(image_tags, (list, tuple)), "image_tags must be a list or tuple"

    # Create a deep copy of the project to avoid modifying the original object
    filtered_project = copy.deepcopy(hA)

    # Retain only images that have at least one tag in image_tags
    filtered_project.images = [
        image for image in filtered_project.images if set(image.tags) & set(image_tags)
    ]

    logger.info(f"Number of images after filtering: {len(filtered_project.images)}")
    return filtered_project

def convert_masks_to_bbox(hA: HastyAnnotationV2) -> HastyAnnotationV2:
    """
    convert every mask to a bounding box
    :param hA:
    :return:
    """

    # TODO it seems this already implemented in the HastyAnnotationV2 object

    return hA


def remove_images_with_no_labels(project: HastyAnnotationV2) -> HastyAnnotationV2:
    # Create a deep copy of the project to avoid modifying the original object
    filtered_project = copy.deepcopy(project)

    # Filter images to only include those with at least one label
    filtered_project.images = [image for image in filtered_project.images if len(image.labels) > 0]

    return filtered_project


def convert_HastyAnnotationV2_to_HastyAnnotationV2flat(project: HastyAnnotationV2) -> [HastyAnnotationV2_flat]:
    """

    :param project:
    :return:
    """
    flat_annotations = []

    for image in project.images:
        for label in image.labels:
            ID = label.attributes.get("ID", None)

            flat_annotations.append(
                HastyAnnotationV2_flat(
                    project_name=project.project_name,
                    create_date=project.create_date,
                    export_format_version=project.export_format_version,
                    export_date=project.export_date,
                    label_classes=project.label_classes,

                    image_id=image.image_id,
                    image_name=image.image_name,
                    dataset_name=image.dataset_name,
                    ds_image_name=image.ds_image_name,
                    width=image.width,
                    height=image.height,
                    image_status=image.image_status,
                    tags=image.tags,
                    image_mode=image.image_mode,

                    label_id=label.id,
                    class_name=label.class_name,
                    bbox=label.bbox,
                    mask=label.mask,
                    z_index=label.z_index,
                    ID=ID
                )
            )
    return flat_annotations


def get_flat_df(project: HastyAnnotationV2) -> pd.DataFrame:
    """
    Convert a HastyAnnotationV2 object to a flat DataFrame
    :param project:
    :return:
    """
    label_data = []

    for image in project.images:
        for label in image.labels:
            poly = label.bbox_polygon

            unique_ID = label.attributes.get("ID", None)
            # [["image_name", "class_name", "ID", "centroid", "bbox", "bbox_polygon"]]
            label_data.append({
                "dataset_name": image.dataset_name if isinstance(image, AnnotatedImage) else None,
                "image_name": image.image_name,
                "image_id": image.image_id,

                "class_name": label.class_name,
                "bbox_x1y1x2y2": label.x1y1x2y2,
                "bbox": label.x1y1x2y2,
                "bbox_polygon": poly,
                "centroid": label.incenter_centroid,
                "mask": label.mask,
                "label_id": label.id,
                "unique_ID": unique_ID,
                "attribute": label.attributes,
            })

    labels_df = pd.DataFrame(label_data)

    return labels_df


@deprecated_warning("This function has been deprecated and will be removed in a future version. Use the static method `HastyAnnotationV2.from_file` instead.")
def hA_from_file(file_path: Path) -> HastyAnnotationV2:
    """
    Load a HastyAnnotationV2 object from a file
    :param file_path:
    :return:
    """
    with open(file_path, mode="r") as f:
        data = json.load(f)
        hA = HastyAnnotationV2(**data)
    return hA


def label_dist_edge_threshold(patch_size, source_image):
    """
    remove labels which are too close to the border. Only in literal edge cases those are not covered anywhereelse
    :param patch_size:
    :param source_image:
    :return:
    """
    n_labels = len(source_image.labels)
    # bd_th = int((patch_size ** 2 // 2) ** 0.5)  # TODO THIS would be the right way to calculate the distance
    bd_th = int(patch_size // 2)
    source_image.labels = [l for l in source_image.labels if l.centroid.within(
        Polygon([(0 + bd_th, bd_th), (source_image.width - bd_th, bd_th),
                 (source_image.width - bd_th, source_image.height - bd_th), (bd_th, source_image.height - bd_th)]))]
    logger.info(
        f"After edge thresholding with distance {bd_th} in {len(source_image.labels)}, remove {n_labels - len(source_image.labels)} labels")

    return source_image.labels


def delete_dataset(dataset_name: str, hA: HastyAnnotationV2):
    """
    delete a dataset within the project completely
    :param dataset_name:
    :param hA:
    :return:
    """
    initial_count = len(hA.images)

    # Filter out images from the specified dataset
    hA.images = [image for image in hA.images if image.dataset_name != dataset_name]

    deleted_count = initial_count - len(hA.images)
    logger.info(f"Deleted {deleted_count} images from dataset '{dataset_name}'")
    logger.info(f"Remaining images: {len(hA.images)}")

    return hA


def replace_image(updateimage: AnnotatedImage, hA: HastyAnnotationV2):
    """
    updateimage (AnnotatedImage): The new/updated image to replace or add
    hA (HastyAnnotationV2): The annotation object containing the images list

    Returns:
        HastyAnnotationV2: The updated annotation object
    """

    # Search for existing image
    update_image_id = updateimage.image_id
    update_image_name = updateimage.image_name
    update_dataset_name = updateimage.dataset_name

    for i, existing_image in enumerate(hA.images):
        existing_id = existing_image.image_id
        existing_name = existing_image.image_name
        existing_dataset_name = existing_image.dataset_name

        # Match by image_id (primary) or image_name (secondary) and dataset_name as third
        if (  # (existing_id == update_image_id) and
                (existing_name == update_image_name) and
                (existing_dataset_name == update_dataset_name)):
            # Replace existing image
            hA.images[i] = updateimage
            return hA

    # No match found, append new image
    hA.images.append(updateimage)
    return hA