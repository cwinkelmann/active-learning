import json
import typing

import copy
import shutil
from typing import Optional, List

import PIL
from PIL import Image

from com.biospheredata.types.status import LabelingStatus, AnnotationType

# Increase the decompression limit
Image.MAX_IMAGE_PIXELS = None
import yaml
from pathlib import Path

import pandas as pd
from loguru import logger
import zipfile
from pycocotools.coco import COCO
from collections import Counter

from com.biospheredata.helper.filenames import (
    get_dataset_image_merged_filesname_v2
)
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage, \
    filter_by_class, filter_by_image_tags
from com.biospheredata.types.ImageStatistics import ImageCounts

class HastyConverter(object):
    """
    Convert Hasty Annotation to other Annotations
    """

    yolo_labels_names = []
    image_names_with_dataset = []
    hA: HastyAnnotationV2

    IMAGES_PATH = "unzipped_images"
    ANNOTATION_PATH = "unzipped_hasty_annotation"
    DEFAULT_DATASET_NAME = "Default"

    def __init__(
            self,
            base_path,
            hA: HastyAnnotationV2,
            images_path: Optional[Path] = None
    ):
        """
        :param base_path:
        :param hasty_annotations_labels_zipped:
        :param hasty_annotations_images_zipped:
        """

        self.class_filter = None
        self.all_annotations = None
        self.base_path = base_path
        self.hA: HastyAnnotationV2 = hA

        if images_path:
            self.hasty_image_base_path = base_path.joinpath(images_path)
        else:
            self.hasty_image_base_path = base_path.joinpath(self.IMAGES_PATH)
        self.hasty_image_annotations_path = base_path.joinpath(self.ANNOTATION_PATH)

        self.hasty_image_flat_path = base_path.joinpath("unzipped_images_flat")

        self.yolo_base_path = base_path.joinpath("unzipped_yolo_format")
        self.yolo_image_path = self.yolo_base_path.joinpath("images")
        self.yolo_label_path = self.yolo_base_path.joinpath("labels")

        self.yolo_labels_names = []
        self.image_names_with_dataset = []

        # delete_hidden_files(self.base_path)

    def get_unzipped_image_files(self):
        """
        This includes the dataset too because we need to pick it out of the right folder
        @return:
        """
        return [f"{x.parts[-2]}/{x.parts[-1]}" for x in self.hasty_image_base_path.glob("*/*")]

    @staticmethod
    def dataset_filter(hA: HastyAnnotationV2, dataset_filter: List[str]):
        """

        :param hA:
        :param dataset_filter: list of dataset names
        :return:
        """
        hA = copy.deepcopy(hA)
        hA.images = [i for i in hA.images if i.dataset_name in dataset_filter]
        return hA

    @staticmethod
    def status_filter(hA: HastyAnnotationV2, status_filter: List[LabelingStatus] = None):
        """
        let only images pass, which are in the status_filter to prevent images which are incomplete
        :param hA:
        :param status_filter: List of Status names
        :return:
        """
        if status_filter is None:
            return hA
        assert type(status_filter) is list or type(status_filter) is tuple, "status_filter must be a list or tuple"
        hA = copy.deepcopy(hA)
        hA.images = [i for i in hA.images if LabelingStatus(i.image_status) in status_filter]

        logger.info(f"Filtered {len(hA.images)} images by status: {status_filter}")
        return hA

    @staticmethod
    def get_unzipped_label_files(annotations_path):
        return set(annotations_path.glob("*.json")) - set(annotations_path.glob(".*.json"))

    @staticmethod
    def get_label_class_mapping(hA: HastyAnnotationV2) -> dict:
        index = 0
        mapping = {}

        for value in hA.label_classes:
            mapping[value.class_name] = index
            index += 1

        return mapping

    @staticmethod
    def convert_to_yolo_boxes(hA: HastyAnnotationV2, yolo_base_path: Path) -> pd.DataFrame:
        """
        # TODO implement this right
        transform the image coordinate from hasty.ai json to yolo.

        HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point
        YOLO Format is x,y of the center point, width, height
        https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

        :param annotation_file:
        :return:
        """
        yolo_base_path.joinpath().mkdir(parents=True, exist_ok=True)

        class_mapping = HastyConverter.get_label_class_mapping(hA)
        yolo_labels_names = []
        df_all_boxes_list = []

        for image in hA.images:

            new_label_name = Path(image.image_name).with_name(Path(image.image_name).stem + ".txt")

            if len(image.labels)> 0:
                df_boxes = HastyConverter.to_yolo_box(class_mapping, df_all_boxes_list, image)

                logger.info(f"new_label_name: {new_label_name}, len of df_boxes: {len(df_boxes)}")
                with open(yolo_base_path / new_label_name, "w") as f:
                    df_boxes[["class_id", "x_yolo", "y_yolo", "w_yolo", "h_yolo"]].to_csv(
                        f, index=False, header=False, sep=" "
                    )
                # raise ValueError("Stop here")
            else:
                logger.warning(f"No labels found in {image.image_name}")
                with open(yolo_base_path / new_label_name, "w") as f:
                    pass  # Creates an empty file

        all_annotations = pd.concat(df_all_boxes_list)

        with open(yolo_base_path.joinpath("class_names.txt"), "w") as f:
            for class_name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):  # Sort by index
                f.write(f"{class_name}\n")

        class_mapping = dict(sorted(class_mapping.items(), key=lambda item: item[1]))

        return class_mapping, all_annotations

    @staticmethod
    def convert_to_yolo_segments(hA: HastyAnnotationV2, yolo_base_path: Path) -> dict:
        """
        # TODO implement this right
        transform the image coordinate from hasty.ai json to yolo.

        HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point
        YOLO Format is x,y of the center point, width, height
        https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

        :param annotation_file:
        :return:
        """
        yolo_base_path.joinpath().mkdir(parents=True, exist_ok=True)

        class_mapping = HastyConverter.get_label_class_mapping(hA)
        df_all_segments_list = []

        for image in hA.images:
            new_label_name = yolo_base_path / Path(image.image_name).with_name(Path(image.image_name).stem + ".txt")

            if len(image.labels) > 0:
                df_segments = HastyConverter.to_yolo_segment(class_mapping, image)
                df_all_segments_list.append(df_segments)
                with open(new_label_name, "w") as f:
                    df_segments.to_csv(
                        f, index=False, header=False, sep=" "
                    )
            else:
                logger.warning(f"No labels found in {image.image_name}")
                with open(new_label_name, "w") as f:
                    pass

        # all_annotations = pd.concat(df_all_segments_list)

        with open(yolo_base_path.joinpath("class_names.txt"), "w") as f:
            for class_name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):  # Sort by index
                f.write(f"{class_name}\n")

        class_mapping = dict(sorted(class_mapping.items(), key=lambda item: item[1]))

        return class_mapping

    @staticmethod
    def to_yolo_box(class_mapping, df_all_boxes_list, image):
        image_name = image.image_name
        image_name_split = image_name.split(".")
        image_name_split[-1] = "txt"
        label_name = ".".join(image_name_split)
        logger.info(label_name)
        image_height = image.height
        image_width = image.width
        labels = image.labels
        if len(labels) > 0:

            # Create boxes
            boxes = [x.x1y1x2y2 for x in labels]
            class_names = [x.class_name for x in labels]
            class_ids = [class_mapping[x] for x in class_names]

            df_boxes = pd.DataFrame(boxes)
            df_boxes.columns = ["x1", "y1", "x2", "y2"]

            df_boxes["h"] = abs(df_boxes["y1"] - df_boxes["y2"])
            df_boxes["w"] = abs(df_boxes["x1"] - df_boxes["x2"])

            #                df_boxes["class"] = class_names
            df_boxes["class_names"] = class_names
            df_boxes["class_id"] = class_ids

            df_boxes["x_yolo"] = (
                    abs(df_boxes["x1"] + df_boxes["x2"]) * 0.5 / image_width
            )
            df_boxes["y_yolo"] = (
                    abs(df_boxes["y1"] + df_boxes["y2"]) * 0.5 / image_height
            )

            df_boxes["w_yolo"] = df_boxes["w"] / image_width
            df_boxes["h_yolo"] = df_boxes["h"] / image_height

            df_boxes["image_name"] = image_name
            df_all_boxes_list.append(df_boxes)


            return df_boxes
        return []


    @staticmethod
    def to_yolo_segment(class_mapping, image: AnnotatedImage):
        """
        create the annotation format of YOLO <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>, see https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format
        :param class_mapping:
        :param image:
        :param yolo_labels_names:
        :return:
        """
        assert isinstance(image, AnnotatedImage)

        image_name = image.image_name
        image_name_split = image_name.split(".")
        image_name_split[-1] = "txt"
        label_name = ".".join(image_name_split)
        logger.info(f"label_name: {label_name}")


        normalised_image = copy.deepcopy(image).normalise()
        labels = normalised_image.labels

        if len(labels) > 0:
            # Create segmentations in YOLO
            segments = [x.get_mask for x in labels]

            class_names = [x.class_name for x in labels]
            class_ids = [class_mapping[x] for x in class_names]
            df_segments = pd.DataFrame(pd.concat([pd.DataFrame(class_ids), pd.DataFrame(segments)], axis=1))
            return df_segments



    @staticmethod
    def convert_to_coco(hA: HastyAnnotationV2,
                        output_file=None):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        data = hA.model_dump()

        annotation_id = 1
        category_mapping = {}
        for idx, category in enumerate(hA.label_classes, start=1):
            coco_format['categories'].append({
                "id": idx,
                "name": category.class_name
            })
            category_mapping[category.class_name] = idx

        for image in hA.images:
            coco_format['images'].append({
                "id": image.image_id,
                "file_name": image.image_name,
                "width": image.width,
                "height": image.height
            })

            for label in image.labels:
                bbox = label.x1y1x2y2
                # Convert bbox format from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
                bbox_coco = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                coco_format['annotations'].append({
                    "id": annotation_id,
                    "image_id": image.image_id,
                    "category_id": category_mapping[label.class_name],
                    "bbox": bbox_coco,
                    "area": bbox_coco[2] * bbox_coco[3],  ### TODO check this
                    "segmentation": [], # TODO check this
                    "iscrowd": 0
                })
                annotation_id += 1

        if output_file:
            with open(output_file, "w") as f:
                json.dump(coco_format, f)
            coco = COCO(output_file)

        return coco_format

    @staticmethod
    def convert_to_herdnet_format(hA: HastyAnnotationV2,
                                  label_mapping=None,
                                  output_file=None,
                                  label_filter: typing.Optional[List[AnnotationType]]=None):
        """
        CSV with image_name, x, y, label_id
        images,x,y,labels
        Example.JPG,517,1653,2
        Example.JPG,800,1253,1
        Example.JPG,78,33,3
        Example_2.JPG,896,742,1
        :param hA: HastyAnnotationV2
        :param output_file:
        :return:
        """
        if label_mapping is None:
            name2id = hA.name2id()
        else:
            name2id = label_mapping

        records = []
        for image in hA.images:
            for l in image.labels:
                # we have keypoints
                if l.keypoints is not None and len(l.keypoints) > 0 and (label_filter is None or AnnotationType.KEYPOINT in label_filter):
                    x = l.keypoints[0].x
                    y = l.keypoints[0].y
                    species = l.class_name
                    labels = name2id[species]

                    records.append((image.image_name, x, y, species, labels))

                elif l.polygon is not None and len(l.polygon) > 0 and l.incenter_centroid is not None and (label_filter is None or AnnotationType.KEYPOINT in label_filter): # TODO check why a polygon can have no center
                    logger.warning(f"No keypoints found in {image.image_name} therefore creating it by getting the center of the polygon")
                    x, y = int(l.incenter_centroid.x), int(l.incenter_centroid.y)
                    species = l.class_name
                    labels = name2id[species]

                    records.append((image.image_name, x, y, species, labels))

                elif l.bbox is not None and len(l.bbox) > 0 and (label_filter is None or AnnotationType.BOUNDING_BOX in label_filter):
                    logger.warning(f"No keypoints found in {image.image_name} therefore creating it by getting the center of the bbox, which will be inaccurate")
                    x, y = int(l.incenter_centroid.x), int(l.incenter_centroid.y)
                    species = l.class_name
                    labels = name2id[species]

                    records.append((image.image_name, x, y, species, labels))
                else:
                    logger.error(f"Skipping {l} in {image.image_name}")

        df_herdnet = pd.DataFrame(records, columns=["images", "x", "y", "species", "labels"])


        df_herdnet.to_csv(output_file, index=False)

        return df_herdnet

    @staticmethod
    def convert_to_herdnet_box_format(hA: HastyAnnotationV2,
                                  output_file=None):
        """
        Hernet Box Format https://github.com/Alexandre-Delplanque/HerdNet

        images,x_min,y_min,x_max,y_max,labels
        Example.JPG,530,1458,585,1750,4
        Example.JPG,95,1321,152,1403,2
        Example.JPG,895,478,992,658,1

        :param hA: HastyAnnotationV2
        :param output_file:
        :return:
        """
        name2id = hA.name2id()

        records = []
        for image in hA.images:
            for l in image.labels:
                # we have keypoints
                if l.keypoints is not None and len(l.keypoints) > 0:
                    logger.warning(f"Creating a box from a keypoint doesn work")

                elif l.polygon is not None and len(
                        l.polygon) > 0 and l.incenter_centroid is not None:  # TODO check why a polygon can have no center
                    logger.info(
                        f"Now creating a box from a polygon in {image.image_name}.")
                    x_min, y_min, x_max, y_max = l.x1y1x2y2
                    species = l.class_name
                    labels = name2id[species]

                    records.append((image.image_name, x_min, y_min, x_max, y_max, species, labels))

                elif l.bbox is not None and len(l.bbox) > 0:
                    x_min, y_min, x_max, y_max = l.x1y1x2y2
                    species = l.class_name
                    labels = name2id[species]

                    records.append((image.image_name, x_min, y_min, x_max, y_max, species, labels))
                else:
                    logger.error(f"Skipping {l} in {image.image_name}")

        df_herdnet = pd.DataFrame(records, columns=["images", "x_min", "y_min", 'x_max', 'y_max', "species", "labels"])

        if output_file is not None:
            df_herdnet.to_csv(output_file, index=False)

        return df_herdnet

    @staticmethod
    def convert_deep_forest(hA: HastyAnnotationV2, output_file: Path) -> pd.DataFrame:
        """
        convert the hasty format to the deep forest format (CSV file with 6 columns: image_path, XYXY absolute position, string label )

        image_path, xmin, ymin, xmax, ymax, label
        OSBS_029.jpg, 256, 99, 288, 140, Tree
        OSBS_029.jpg, 166, 253, 225, 304, Tree

        :param output_file:
        :param hA:
        :return:
        """
        df_herdnet = HastyConverter.convert_to_herdnet_box_format(hA=hA)
        df_deep_forest = df_herdnet.rename(columns={
            "images": "image_path",
            "x_min": "xmin",
            "y_min": "ymin",
            "x_max": "xmax",
            "y_max": "ymax",
            "species": "label"
        })[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]]

        df_deep_forest.to_csv(output_file, index=False)
        logger.info(f"Saved deep forest format to {output_file}")
        return df_deep_forest


    @staticmethod
    def prepare_YOLO_output_folder_str(base_path: Path,
                                       images_train_path: Path,
                                       images_val_path: Path,
                                       labels_train_path: Path,
                                       labels_val_path: Path,
                                       class_names: List,
                                       data_yaml_path: Path,
                                       images_test_path: typing.Optional[Path] = None,
                                       labels_test_path: typing.Optional[Path] = None
                                       ):
        """
        :param base_path:
        :return:
        """

        """
        path:  # (Optional) root dataset directory

        train:
          images: /absolute/path/to/your/train/images
          labels: /absolute/path/to/your/train/labels
        
        val:
          images: /absolute/path/to/your/val/images
          labels: /absolute/path/to/your/val/labels
          """

        # TODO wasn't there a class for this? YoloDataYaml


        data_yaml = {
            "train": {
                "images": str(images_train_path),
                "labels": str(labels_train_path),
            },
            "val": {
                "images": str(images_val_path),
                "labels": str(labels_val_path),
            },
            #"nc": len(class_names),
            "names": class_names,
        }

        if labels_test_path and images_test_path:
            data_yaml["test"] = {"images": str(images_test_path), "labels": str(labels_test_path)}

        with open(base_path.joinpath(data_yaml_path), "w+") as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=False)

        return data_yaml

    def set_class_filter(self, keep_classes):
        ## TODO remove
        self.class_filter = keep_classes

    @staticmethod
    def from_zip(output_path: Path,
                 hasty_annotations_labels_zipped: Path,
                 hasty_annotations_images_zipped: Path):
        """
        Unzip downloaded files to base_path
        :param base_path:
        :param hasty_annotations_labels_zipped:
        :param hasty_annotations_images_zipped:
        :return:
        """

        full_hasty_annotations_images_zipped_path = hasty_annotations_images_zipped
        full_hasty_annotations_labels_zipped_path = hasty_annotations_labels_zipped

        if not full_hasty_annotations_images_zipped_path.exists():
            raise FileNotFoundError(f"File not found: {full_hasty_annotations_images_zipped_path}")
        if not full_hasty_annotations_labels_zipped_path.exists():
            raise FileNotFoundError(f"File not found: {full_hasty_annotations_labels_zipped_path}")

        ## unzip the files
        images_path = output_path / HastyConverter.IMAGES_PATH
        annotations_path = output_path / HastyConverter.ANNOTATION_PATH

        if not images_path.exists():
            unzip_files(hasty_annotations_images_zipped, images_path)
        if not annotations_path.exists():
            unzip_files(hasty_annotations_labels_zipped, annotations_path)

        for idx, annotation_file in enumerate(
                HastyConverter.get_unzipped_label_files(annotations_path)):
            hA = HastyConverter.from_folder(hA_path=annotation_file)
            return hA, images_path

            if idx > 0:
                raise ValueError("We only support one annotation file at the moment")

    @staticmethod
    def from_folder(hA_path: Path) -> HastyAnnotationV2:
        """
        @depreacted, use hA_from_file from HastyAnnotationV2
        :param hA_path:
        :return:
        """
        with open(hA_path, 'r') as f:
            hasty_annotation = json.load(f)

            # hasty_annotation["images"] = [x for x in hasty_annotation["images"] if x["image_name"] == "DJI_0395.JPG"]
            hA = HastyAnnotationV2(**hasty_annotation)
            # HastyAnnotation.from_dict(hasty_annotation)

        return hA

    @staticmethod
    def copy_images_to_flat_structure(hA: HastyAnnotationV2,
                                      base_bath: Path,
                                      folder_path: Path):
        """
        Copy images from their Dataset folders to a flat structure

        :param hA_flat:
        :param base_bath:
        :param folder_path:
        :return:
        """
        folder_path.mkdir(exist_ok=True)
        hA_flat = copy.deepcopy(hA)
        for row in hA_flat.images:
            dataset_name = row.dataset_name
            source_image = row.image_name
            ds_image_name = row.ds_image_name

            image_source_name = base_bath / dataset_name / source_image
            if not folder_path.joinpath(ds_image_name).exists():
                shutil.copy(
                    image_source_name,
                    folder_path / ds_image_name
                )
        return hA_flat


    @staticmethod
    def delete_hidden_files(base_path):
        """pass"""
        for file in set(base_path.rglob("._*")):
            file.unlink()
        return True

    @staticmethod
    def convert_ids_to_int(hA: HastyAnnotationV2) -> HastyAnnotationV2:
        """image_id is a uui, convert to int"""

        def create_unique_ids(strings):
            # Remove duplicates by converting to a set, then back to a list to preserve order
            unique_strings = list(set(strings))
            # Use enumerate to assign unique IDs
            id_mapping = {string: id for id, string in enumerate(unique_strings)}
            return id_mapping

        image_id_mapping = create_unique_ids([x.image_id for x in hA.images])

        for image in hA.images:
            image.image_id = image_id_mapping[image.image_id]

            # for l in image.labels:
            #     assert isinstance(l, ImageLabel)
            #     l.ID = image_id_mapping[l.ID]

        return hA

    @staticmethod
    def convert_to_regression(images: List[AnnotatedImage]) -> List[ImageCounts]:
        """Get a Regression Ready Dataset which includes only the counts per class per image"""

        rows = []

        for i in images:
            c = Counter()
            for l in i.labels:
                c[l.class_name] += 1

            ic = ImageCounts(image_id=i.image_id, dataset_name=i.dataset_name,
                             labels=c, image_name=i.image_name)
            rows.append(ic)

        return rows

    @staticmethod
    def convert_to_regression_df(images: List[AnnotatedImage]) -> pd.DataFrame:
        """Get a Regression Ready Dataset which includes only the counts per class per image"""
        rows = []

        for o in HastyConverter.convert_to_regression(images):
            df_labels = pd.DataFrame(pd.Series(o.labels)).T
            df_labels["image_id"] = o.image_id
            df_labels["dataset_name"] = o.dataset_name
            df_labels["image_name"] = o.image_name
            rows.append(df_labels)

        return pd.concat(rows).reset_index(drop=True)



    @classmethod
    def annotation_types_filter(cls, hA: HastyAnnotationV2, annotation_types: List[AnnotationType]):
        """
        remove all annotations which are not in the list of annotation_types, e.g. ["box", "polygon"] keeps boxes and polygons
        :param hA:
        :param annotation_types:
        :return:
        """
        assert type(annotation_types) is list or type(annotation_types) is tuple, "image_names must be a list or tuple"

        if len(annotation_types) > 0:
            # Create a deep copy of the project to avoid modifying the original object
            filtered_project = copy.deepcopy(hA)

            for image in filtered_project.images:
                keep = []
                for il in image.labels:
                    # bounding box
                    if il.bbox is not None and AnnotationType.BOUNDING_BOX in annotation_types and il.polygon is None:
                        keep.append(il)
                    if (il.keypoints is not None and len(il.keypoints) > 0 and il.keypoints[0].keypoint_class_id == 'ed18e0f9-095f-46ff-bc95-febf4a53f0ff'
                            and AnnotationType.KEYPOINT in annotation_types):
                        keep.append(il)
                    if il.polygon is not None and AnnotationType.POLYGON in annotation_types:
                        keep.append(il)
                image.labels = keep

            # filtered_project.images = filtered_images
            logger.info(f"Kept {filtered_project.label_count()} labels by annotation types: {annotation_types}")
            return filtered_project
        else:
            return hA

    @classmethod
    def convert_to_classification(cls, source_path: Path, hA_crops: HastyAnnotationV2, output_file: Path):
        """
        Organize images for image classification by moving them to appropriate folders.
        Empty images go to an 'empty' folder, while images with labels go to folders
        named after the most frequent class in the image.

        Args:
            hA_crops: HastyAnnotationV2 object containing cropped images and their annotations
            output_file: Base directory where the classification folders will be created

        Returns:
            Dictionary with statistics about the classification conversion
        """
        logger.warning(f"So far this doesn't really work")
        # Create output directory if it doesn't exist
        output_file.mkdir(parents=True, exist_ok=True)

        # Create 'empty' subfolder
        empty_dir = output_file / "empty"
        empty_dir.mkdir(exist_ok=True)

        # Track statistics
        stats = {
            "total_images": len(hA_crops.images),
            "empty_images": 0,
            "labeled_images": 0,
            "class_distribution": {}
        }

        # Process each image
        for image in hA_crops.images:
            source_image_path = source_path / Path(image.image_name)

            assert source_image_path.exists()

            # Check if the image has any labels
            if not image.labels or len(image.labels) == 0:
                # This is an empty image, move to 'empty' folder
                target_dir = empty_dir
                stats["empty_images"] += 1
            else:
                # Count labels by class
                class_counts = {}
                for label in image.labels:
                    if label.class_name in class_counts:
                        class_counts[label.class_name] += 1
                    else:
                        class_counts[label.class_name] = 1

                # Find most frequent class
                most_frequent_class = max(class_counts.items(), key=lambda x: x[1])[0]

                # Create class folder if it doesn't exist
                class_dir = output_file / most_frequent_class
                class_dir.mkdir(exist_ok=True)

                # Update statistics
                target_dir = class_dir
                stats["labeled_images"] += 1
                if most_frequent_class in stats["class_distribution"]:
                    stats["class_distribution"][most_frequent_class] += 1
                else:
                    stats["class_distribution"][most_frequent_class] = 1

            # Copy the image to the target directory
            try:
                # Get just the filename without any parent directories
                filename = source_image_path.name

                # If image path is absolute or exists, copy the file

                shutil.copy(source_image_path, target_dir / filename)

            except Exception as e:
                logger.error(f"Error copying image {source_image_path}: {e}")

        # Log summary
        logger.info(f"Classification conversion complete. Total: {stats['total_images']}, "
                    f"Empty: {stats['empty_images']}, Labeled: {stats['labeled_images']}")
        logger.info(f"Class distribution: {stats['class_distribution']}")

        return stats


def unzip_files(input_path, output_path):
    """
    unzip files to get annotations
    """
    logger.info(f"unzip: {input_path} to {output_path}")

    with zipfile.ZipFile(input_path, "r") as zip_ref:
        zip_ref.extractall(output_path)


def filter_by_image_name(hA, image_names: Optional[List[str]], exclude=False):
    if image_names is None or len(image_names) == 0:
        return hA

    assert type(image_names) is list or type(image_names) is tuple, "image_names must be a list or tuple"

    if len(image_names) > 0:
        # Create a deep copy of the project to avoid modifying the original object
        filtered_project = copy.deepcopy(hA)

        # check if image is in the list "image_names"
        if exclude == False:
            filtered_images = [i for i in filtered_project.images if i.image_name in image_names]
        else:
            filtered_images = [i for i in filtered_project.images if i.image_name not in image_names]
        filtered_project.images = filtered_images

        logger.info(f"Kept {hA.label_count()} labels ")
        return filtered_project
    else:
        return hA


def hasty_filter_pipeline(
                            hA: HastyAnnotationV2,

                           dataset_filter: List[str] = None,
                           status_filter: List[LabelingStatus] = None,
                           class_filter: List[str] = None,
                           image_tags: List[str] = None,
                           images_filter: Optional[List[str]] = None,
                           images_exclude: Optional[List[str]] = None,

                            annotation_types: Optional[List[AnnotationType]] = None,
                            num_images: int = None,
                            sample_strategy: str = None,
                           default_dataset_name=HastyConverter.DEFAULT_DATASET_NAME,

) -> HastyAnnotationV2:
    """
    process the hasty annotations format and filter it

    :param labels_path:
    :param hasty_annotations_labels_zipped:
    :param hasty_annotations_images_zipped:
    :param filter_by_dataset_name:
    :return:
    """


    # remove elements not in here
    if dataset_filter is not None and len(dataset_filter) > 0:
        hA = HastyConverter.dataset_filter(hA, dataset_filter)
    # remove labels not in here
    hA = filter_by_class(hA=hA, class_names=class_filter)
    hA = HastyConverter.status_filter(hA=hA, status_filter=status_filter)

    # image names
    hA = filter_by_image_name(hA=hA, image_names=images_filter)
    hA = filter_by_image_name(hA=hA, image_names=images_exclude, exclude=True)

    # image_tags
    hA = filter_by_image_tags(hA=hA, image_tags=image_tags)


    if annotation_types is not None and len(annotation_types) > 0:
        hA = HastyConverter.annotation_types_filter(hA, annotation_types)

    # remove completely empty images
    hA.images = [i for i in hA.images if len(i.labels) > 0]
    logger.info(f"Amount of labels {hA.label_count()} ")
    for i in hA.images:
        i.ds_image_name = get_dataset_image_merged_filesname_v2(dsn=i.dataset_name, imn=i.image_name)

    return hA


def get_image_dimensions(image_path) -> typing.Tuple[int, int]:
    """
    Get the dimensions of an image file.
    :param image_path:
    :return:
    """
    with PIL.Image.open(image_path) as img:
        width, height = img.size
    return width, height



