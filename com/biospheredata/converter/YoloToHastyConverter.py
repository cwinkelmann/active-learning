import glob
import json
import numpy as np
import shutil
import yaml
from pathlib import Path
from PIL import Image

import pandas as pd
from loguru import logger
import zipfile

from com.biospheredata.converter.YoloTiler import YoloTiler
from com.biospheredata.helper.filenames import get_dataset_image_merged_filesname, get_dataset_image_merged_filesname_v2
from com.biospheredata.helper.image.identifier import get_image_id
from com.biospheredata.types.HastyAnnotation import HastyAnnotation


class YoloToHastyConverter(object):
    """
    Convert YOLO Annotations to Hasty Annotations
    often these are made on slices and not the original images so projecting them back would be
    """
    yolo_labels_names = []
    image_names_with_dataset = []
    annotations_labels_path = []
    base_path: Path

    def __init__(self,
                 base_path: Path,
                 annotations_labels_path: Path,
                 annotations_images_path: Path,
                 class_name_mapping_path: Path,
                 ext="JPG"):
        """

        :param base_path:
        :param annotations_labels_path:
        :param annotations_images_path:
        """

        self.base_path = base_path
        self.annotations_labels_path = annotations_labels_path
        self.annotations_images_path = annotations_images_path
        self.class_name_mapping_path = class_name_mapping_path

        self.found_labels = list(self.base_path.joinpath(annotations_labels_path).glob("*.txt"))
        if self.found_labels == 0:
            raise ValueError(f"no labels found in {annotations_labels_path}")

        self.image_names_with_dataset = list(self.base_path.joinpath(annotations_images_path).glob(f"*.{ext}"))
        if self.image_names_with_dataset == 0:
            raise ValueError(f"no images found in {annotations_labels_path}")

    def get_x_for_y(self, label_path: Path, suffix: str = "JPG") -> Path:
        """
        find the right image for the label or the other way around
        :param suffix:
        :param label_path:
        :return:
        """
        return Path(f"{str(label_path).rstrip(label_path.suffix)}.{suffix}")

    def transform_coordinates(self) -> list[pd.DataFrame]:
        """
        :return:
        """
        converted_images_list = {}

        for image_path in self.image_names_with_dataset:

            label_path = self.get_x_for_y(Path(image_path), suffix="txt")
            im = Image.open(image_path)
            imr = np.array(im, dtype=np.uint8)

            image_height = imr.shape[0]
            image_width = imr.shape[1]

            df_yolo = pd.read_csv(label_path, sep=" ", names=["class_id", "x_center", "y_center", "width", "height"])
            # the coordinates are all in percentage

            df_yolo[["y_center", "height"]] = df_yolo[["y_center", "height"]] * image_height
            df_yolo[["x_center", "width"]] = df_yolo[["x_center", "width"]] * image_width

            df_hasty = pd.DataFrame()
            df_hasty["x1"] = df_yolo["x_center"] - df_yolo["width"]/2
            df_hasty["x2"] = df_yolo["x_center"] + df_yolo["width"]/2
            df_hasty["y1"] = df_yolo["y_center"] - df_yolo["height"]/2
            df_hasty["y2"] = df_yolo["y_center"] + df_yolo["height"]/2
            df_hasty["image_height"] = image_height
            df_hasty["image_width"] = image_width
            df_hasty["image_id"] = get_image_id(image_path)
            df_hasty["image_name"] = image_path.name
            df_hasty["label_name"] = label_path.name
            df_hasty["class_id"] = df_yolo["class_id"]

            converted_images_list[image_path.name] = df_hasty

        return converted_images_list

    def transform(self, hA: HastyAnnotation):
        """
        take each image and convert it to HastyAnnotation
        :type hA: HastyAnnotation
        :return:

        """

        converted_images_list = self.transform_coordinates()

        for image, annotations_df in converted_images_list.items():
            assert isinstance(image, str)
            assert isinstance(annotations_df, pd.DataFrame)
            labels_dict = annotations_df.to_dict(orient="records")
            for l in labels_dict:
                hA.add_image(image_id=l["image_id"],
                             width=l["image_width"],
                             height=l["image_height"],
                             image_name=image
                             )

                hA.add_label(l["image_id"],
                             class_name=self.get_class_name(l["class_id"]),
                             bbox=[
                                 int(round(l["x1"])),
                                 int(round(l["y1"])),
                                 int(round(l["x2"])),
                                 int(round(l["y2"]))
                             ],
                             id=l["class_id"]
                             )
        return hA

    def get_class_name(self, class_id):
        with open(self.base_path.joinpath(self.class_name_mapping_path), "r") as f:
            class_mappings = json.load(f)
            return class_mappings[class_id]


