from __future__ import annotations

import copy
import json
import pandas as pd
import uuid
from pathlib import Path

from loguru import logger
from typing_extensions import deprecated

from com.biospheredata.types.annotationbox import bbox_creator


@deprecated("There is enough spam in the world")
class HastyAnnotation(object):
    """@deprecated
    Wrapper for an Annotation
    """

    project_name = None
    create_date = None
    export_date = None

    def __init__(self, project_name: str, create_date: str, export_date: str,
                 detection_model: object = None) -> object:
        self.project_name = project_name
        self.create_date = create_date
        self.export_date = export_date

        self.export_format_version = "1.1"
        self.label_classes = []
        self.images = []
        self.images_dict = {}
        self.image_id_labels = {}
        self.attributes = []

        # self.set_labelclasses_from_model(detection_model)

    def set_labelclasses_from_model(self, detection_model):
        """

        @param detection_model:
        @return:
        """
        for category_name in detection_model.category_names:
            import random
            random_number = random.randint(0, 16777215)
            hex_number = str(hex(random_number))
            hex_number = '#' + hex_number[2:]

            self.add_label_class(
                {
                    "class_id": str(uuid.uuid1()),
                    "parent_class_id": None,
                    "class_name": category_name,
                    "class_type": "object",
                    "color": hex_number,
                    "norder": 10.0,
                    "icon_url": None,
                    "attributes": []
                }
            )

    def set_label_classes(self, label_classes):
        self.label_classes = label_classes

    def add_label_class(self, label_class) -> bool:
        """

        @param label_class:
        @return:
        """
        ## TODO do a proper check based on the class_id
        if label_class["class_name"] not in [x["class_name"] for x in self.label_classes]:
            self.label_classes.append(label_class)

    def add_image(self, image_id, width, height, image_name, dataset_name="Default", image_status="IN PROGRESS",
                  image_mode="YCbCr") -> bool:
        image_object = {
            "image_id": image_id,
            "width": width,
            "height": height,
            "image_name": image_name,
            "image_mode": image_mode,
            "dataset_name": dataset_name,
            "image_status": image_status,
            "labels": [],
            "tags": [],
        }

        if image_id not in self.images_dict:
            self.images_dict[image_id] = image_object
            self.images.append(Path(image_name).parts[-1])  # TODO: this is deprecated
            return True
        else:
            logger.warning(f"prevented to a add a duplicate to the dataset: {image_object}. This might be ok, because of the hacky nature of the script...")
            return False

    def add_label(self, image_id, id, class_name, bbox, polygon=None, mask=None, z_index=1, attributes={}):
        label = {
            "id": id,
            "class_name": class_name,
            "bbox": bbox,
            "polygon": polygon,
            "mask": mask,
            "z_index": z_index,
            "attributes": attributes
        }

        self.images_dict[image_id]["labels"].append(label)

    def set_labels(self, image_id, labels):
        self.images_dict[image_id]["labels"] = labels

    def set_image(self, image_dict):
        self.images_dict[image_dict["image_id"]] = image_dict

    @staticmethod
    def from_dict(ha_dict):
        """
        @param self:
        @return:
        """
        export_format_version = ha_dict.get("export_format_version")
        if export_format_version == "1.1":
            project_name = ha_dict.get("project_name")
            create_date = ha_dict.get("create_date")
            export_date = ha_dict.get("export_date")
            hA = HastyAnnotation(project_name=project_name, create_date=create_date, export_date=export_date)

            hA.label_classes = ha_dict.get("label_classes")
            hA.images_dict = {n["image_id"]: n for n in ha_dict.get("images")}
            hA.attributes = ha_dict.get("attributes")

            return hA
        else:
            raise Exception(f"Export Version is wrong, {export_format_version}")

    @staticmethod
    def from_file(file):
        """
        load the serialized json reprensentation
        @param file:
        @return:
        """
        with open(file, 'r') as f:
            return HastyAnnotation.from_dict(json.load(f))

    def get_statistic(self, global_count = True):
        """
        return counts per label for all annotations
        @return:
        """
        if global_count:

            label_statistic = {}
            for image in self.images_dict.values():
                for label in image.get("labels"):
                        class_name = label["class_name"]
                        label_statistic[class_name] = label_statistic.get(class_name, 0) + 1
            label_statistic = pd.DataFrame.from_dict([label_statistic])

        else:

            label_statistic = []

            for image in self.images_dict.values():

                image_name = image["image_name"]

                for label in image.get("labels"):

                    class_name = label["class_name"]
                    label_statistic.append({"image_name": image_name, "class_name": class_name, "count": 1})

                    # [image_name][class_name] = label_statistic[image_name].get(class_name, 0) + 1

            df_statistic = pd.DataFrame.from_dict(label_statistic)
            label_statistic = df_statistic.groupby(by=["image_name", "class_name"], as_index=False).count()

        return label_statistic


    def to_dict(self):
        """

        @return:
        """
        return {
            "project_name": self.project_name,
            "create_date": self.create_date,
            "export_format_version": self.export_format_version,
            "export_date": self.export_date,
            "label_classes": self.label_classes,
            "images": list(self.images_dict.values()),
            "attributes": []

        }

    def flat_df(self):
        """
        The Annotation Format is quite nested. This flattens it completely and add each attribute as a columns

        """
        lst_df_flat = []

        for key, _image in self.images_dict.items():
            try:
                df_labels = pd.DataFrame(_image["labels"])
                df_labels["image_id"] = _image["image_id"]

                _image.pop("labels")
                _image.pop("tags")
                if "tag_groups" in _image:
                    _image.pop("tag_groups")
                df_image = pd.DataFrame([_image])

                df_flat = df_labels.merge(df_image, on=["image_id"])
                df_flat = df_flat.apply(bbox_creator, args=(), output_path=Path("crops"), axis=1)

                if "ID" in list(df_flat.columns) and 1 > df_flat.groupby(['ID'])['ID'].count().max():
                    df_flat.groupby(['ID'])['ID'].count()
                    raise ValueError("there is duplicate ID in the data.")
                lst_df_flat.append(df_flat)

            except Exception as e:
                # no annotatations
                logger.warning(e)
                logger.warning(f"No Annotations for image: {df_image['image_name']}")
                pass

        df_flat = pd.concat(lst_df_flat, axis=0)
        df_flat["centroid"] = [poly.centroid for poly in df_flat["bbox_polygon"]]
        return df_flat



    def get_images_with_more_than(self, searched_label, threshold):
        """
        @deprecated
        @param searched_label:
        @param threshold:
        @return:
        """
        matching_images = []
        for image in self.images_dict.values():
            current = 0
            for label in image["labels"]:
                if label["class_name"] == searched_label:
                    current += 1
            if current > threshold:
                matching_images.append(image["image_name"])

        return matching_images

    def merge_hasty_annotations_two(hA_1: HastyAnnotation, hA_2: HastyAnnotation) -> HastyAnnotation:
        """
        combine two annotation sets

        @param hA_1:
        @param hA_2:
        @param kargs:
        @return:
        """

        ## work through the images classes

        ## work through the images
        for label_class in hA_2.label_classes:
            hA_1.add_label_class(label_class)

        for k, id in hA_2.images_dict.items():
            added = hA_1.add_image(image_id=id["image_id"],
                                   width=id["width"],
                                   height=id["height"],
                                   image_name=id["image_name"],
                                   dataset_name=id["dataset_name"],
                                   )
            if added:
                hA_1.set_labels(image_id=id["image_id"], labels=id["labels"])

        return hA_1

    @staticmethod
    def merge_hasty_annotations(hA_1: HastyAnnotation, hA_list: [HastyAnnotation]) -> HastyAnnotation:
        """
        combine two annotation sets

        @return:
        @param hA_list:
        @return:
        @param hA_1:
        @param hA_2:
        @param kargs:
        @return:
        """

        ## work through the images classes
        n = 1
        for hA_list_item in hA_list:
            logger.info(f"added {n}th item")
            hA_1 = HastyAnnotation.merge_hasty_annotations_two(hA_1, hA_list_item)
            n += 1
        return hA_1

    @staticmethod
    def static_get_images_with_more_than(hA: HastyAnnotation, searched_label, threshold) -> HastyAnnotation:
        """
        @param hA:
        @param searched_label:
        @param threshold:
        @return:
        """

        hA_new = copy.deepcopy(hA)
        hA_new.images_dict = {}

        matching_images = []
        for image in hA.images_dict.values():
            current = 0
            for label in image["labels"]:
                if label["class_name"] == searched_label:
                    current += 1
            if current > threshold:
                matching_images.append(image["image_name"])
                hA_new.set_image(image)

        return hA_new

    def set_base_path(self, base_path):
        self.base_path = base_path



    def image_summary_to_dataframe(self):
        """

        :return:
        """
        summaries = []
        for id, image in self.images_dict.items():
            image_summary = {}
            image_summary["image_name"] = image["image_name"]
            for l in image["labels"]:
                image_summary[l["class_name"]] = image_summary.get(l["class_name"], 0) + 1

            summaries.append(image_summary)
        summary_df = pd.DataFrame.from_records(summaries)

        return summary_df.fillna(0)