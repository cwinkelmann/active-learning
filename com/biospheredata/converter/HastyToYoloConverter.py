import glob
import json
import shutil
import yaml
from pathlib import Path

import pandas as pd
from loguru import logger
import zipfile
from pycocotools.coco import COCO

from com.biospheredata.converter.YoloTiler import YoloTiler
from com.biospheredata.helper.filenames import (
    get_dataset_image_merged_filesname,
    get_dataset_image_merged_filesname_v2,
)
from com.biospheredata.types.HastyAnnotation import HastyAnnotation


class HastyToYOLOConverter(object):
    """
    Convert Hasty Annotation to YOLO Annotations
    """

    yolo_labels_names = []
    image_names_with_dataset = []

    def __init__(
        self,
        base_path,
        hasty_annotations_labels_zipped,
        hasty_annotations_images_zipped,
    ):
        """

        :param base_path:
        :param hasty_annotations_labels_zipped:
        :param hasty_annotations_images_zipped:
        """

        self.class_filter = None
        self.all_annotations = None
        self.base_path = base_path
        self.hasty_annotations_labels_zipped = hasty_annotations_labels_zipped
        self.hasty_annotations_images_zipped = hasty_annotations_images_zipped

        self.hasty_image_base_path = base_path.joinpath("unzipped_images")
        self.hasty_image_annotations = base_path.joinpath("unzipped_hasty_annotation")
        self.yolo_base_path = base_path.joinpath("unzipped_yolo_format")

        self.yolo_image_path = self.yolo_base_path.joinpath("images")
        self.yolo_label_path = self.yolo_base_path.joinpath("labels")

        self.yolo_labels_names = []
        self.image_names_with_dataset = []

    def prepare_zip_files(self):
        """
        unzip files to get annotations
        """
        logger.info(f"unzip labels: {self.hasty_annotations_labels_zipped}")
        path_to_labels_zip_file = self.base_path.joinpath(
            self.hasty_annotations_labels_zipped
        )
        with zipfile.ZipFile(path_to_labels_zip_file, "r") as zip_ref:
            zip_ref.extractall(self.hasty_image_annotations)

        logger.info(f"unzip images : {self.hasty_annotations_images_zipped}")
        path_to_images_zip_file = self.base_path.joinpath(
            self.hasty_annotations_images_zipped
        )
        with zipfile.ZipFile(path_to_images_zip_file, "r") as zip_ref:
            zip_ref.extractall(self.hasty_image_base_path)

        logger.info(f"unzipped files successfully.")

    def get_unzipped_image_files(self):
        """
        This includes the dataset too because we need to pick it out of the right folder
        @return:
        """
        return [
            f"{x.parts[-2]}/{x.parts[-1]}"
            for x in self.hasty_image_base_path.glob("*/*")
        ]
        # return [f"{x.parts[-1]}" for x in self.hasty_image_base_path.glob("*/*")]

    def merge_image_datasets(self):
        """
        With hasty.ai there are multiple datasets. They are all moved into unzipped_yolo_format
        @return:
        """

        self.yolo_image_path.mkdir(parents=True, exist_ok=True)

        for image_name in self.get_unzipped_image_files():

            image_name_with_dataset = "___".join(
                Path(image_name).parts
            )  # TODO don't do this anymore or maybe do because you don't know how it is used later

            logger.info(f"processing {image_name_with_dataset}")
            self.image_names_with_dataset.append(image_name_with_dataset)

            shutil.copy(
                self.hasty_image_base_path.joinpath(image_name),
                self.yolo_image_path.joinpath(image_name_with_dataset),
            )

    def get_unzipped_label_files(self):
        return set(self.hasty_image_annotations.glob("*.json")) - set(self.hasty_image_annotations.glob(".*.json"))

    def get_label_class_mapping(self, hA: HastyAnnotation) -> dict:
        index = 0
        mapping = { }

        for value in hA.label_classes:
            mapping[value["class_name"]] = index
            index += 1

        return mapping

    def transform_hasty_to_yolo(self, annotation_file):
        """
        transform the image coordinate from hasty.ai json to yolo.

        HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point
        YOLO Format is x,y of the center point, width, height
        https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

        :param annotation_file:
        :return:
        """

        self.yolo_base_path.mkdir(parents=True, exist_ok=True)
        self.yolo_label_path.mkdir(parents=True, exist_ok=True)

        with open(annotation_file, "r") as f:
            data = json.load(f)
            # print(data)
            hA = HastyAnnotation.from_dict(data)

        class_mapping = self.get_label_class_mapping(hA)

        df_all_boxes_list = []


        for image_id, image in hA.images_dict.items():
            image_name = image["image_name"]
            dataset_name = image["dataset_name"]
            image_name_split = image_name.split(".")
            image_name_split[-1] = "txt"
            label_name = ".".join(image_name_split)
            logger.info(label_name)

            image_height = image["height"]
            image_width = image["width"]
            labels = image["labels"]
            if len(labels) > 0:
                boxes = [x["bbox"] for x in labels]
                class_names = [x["class_name"] for x in labels]
                class_ids = [class_mapping[x] for x in class_names]

                df_boxes = pd.DataFrame(boxes)
                df_boxes.columns = ["x1", "y1", "x2", "y2"]

                # df_boxes = pd.DataFrame(image["marks"])
                # for class_label in set([x["class_name"] for x in labels]):
                #     if class_label in consistent_ids:
                #         pass
                #     else:
                #         consistent_ids.append(class_label)

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

                # df_boxes = df_boxes.merge(
                #     pd.DataFrame(consistent_ids, columns=["class"]).reset_index(),
                #     on="class",
                #     how="left",
                # )
                # df_boxes["class"] = df_boxes["index"]
                df_boxes["image_name"] = image_name
                df_boxes["class_names"].filter(["iguana"])
                if self.class_filter:
                    df_boxes = df_boxes[df_boxes['class_names'].isin(self.class_filter)]

                df_all_boxes_list.append(df_boxes)
                # write the annotation to disk

                new_label_name = self.yolo_label_path.joinpath(
                    f"{get_dataset_image_merged_filesname(dsn=dataset_name, imn=label_name, iid=image['image_id'])}"
                )
                new_label_name = self.yolo_label_path.joinpath(
                    f"{get_dataset_image_merged_filesname_v2(dsn=dataset_name, imn=label_name)}"
                )
                with open(new_label_name, "w") as f:
                    df_boxes[["class_id", "x_yolo", "y_yolo", "w_yolo", "h_yolo"]].to_csv(
                        f, index=False, header=False, sep=" "
                    )

                self.yolo_labels_names.append(new_label_name)

                ## FIXME we don't do this now. Because We want to use the tiff and not the jpg
                # # copy the image to the yolo folder
                # source_file = hasty_image_base_path.joinpath(image["dataset_name"]).joinpath(image_name)
                # target_file = yolo_base_path.joinpath(image_name)
                # source_file.parts[-1].split(".")[1] = ext
                # logger.info(f"copy {source_file} to {target_file}")
                # shutil.copyfile(source_file, target_file)

        ## this is a very complicated way of getting the mapping, because hasty already has seperate label classes
        # mapping = pd.concat(df_all_boxes_list)[["class_names", "class"]].drop_duplicates(["class", "class_names"]).sort_values(by="class", ascending=True)
        
        self.all_annotations = pd.concat(df_all_boxes_list)
        
        with open(self.yolo_base_path.joinpath("class_names.txt"), "w+") as f:
            json.dump(list(class_mapping.keys()), f)

        self.consistent_ids = list(class_mapping.keys())
        self.class_mapping = class_mapping
        return self.consistent_ids

    def convert_to_coco(self, annotation_file, output_file=None):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        with open(annotation_file, "r") as f:
            data = json.load(f)
            # print(data)
            hA = HastyAnnotation.from_dict(data)

        class_mapping = self.get_label_class_mapping(hA)

        annotation_id = 1
        category_mapping = {}
        for idx, category in enumerate(data['label_classes'], start=1):
            coco_format['categories'].append({
                "id": idx,
                "name": category['class_name']
            })
            category_mapping[category['class_name']] = idx

        for image in data['images']:
            coco_format['images'].append({
                "id": image['image_id'],
                "file_name": image['image_name'],
                "width": image['width'],
                "height": image['height']
            })
            for label in image.get('labels', []):
                bbox = label['bbox']
                # Convert bbox format from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
                bbox_coco = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                coco_format['annotations'].append({
                    "id": annotation_id,
                    "image_id": image['image_id'],
                    "category_id": category_mapping[label['class_name']],
                    "bbox": bbox_coco,
                    "area": bbox_coco[2] * bbox_coco[3],
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

        if output_file:
            with open(output_file, "w") as f:
                json.dump(coco_format, f)
            coco = COCO(output_file)

        return coco_format

    def transform_annotation(self):
        """
        if there are multiple label files go through them
        @return:
        """
        for annotation_file in self.get_unzipped_label_files():
            self.convert_to_coco(annotation_file=annotation_file, output_file=self.yolo_base_path.joinpath("coco_format.json"))
            self.transform_hasty_to_yolo(annotation_file=annotation_file)
            # TODO if there are multiple different label classes you have a problem

        logger.info(f"done with transforming the labels.")

    def prepare_output_folder_str(self, base_path: Path, yolo_base_path: Path):
        """

        :param base_path:
        :return:
        """
        # base_path.joinpath("test/images").mkdir(exist_ok=True, parents=True)
        # base_path.joinpath("test/labels").mkdir(exist_ok=True, parents=True)
        base_path.joinpath("train/images").mkdir(exist_ok=True, parents=True)
        base_path.joinpath("train/labels").mkdir(exist_ok=True, parents=True)
        base_path.joinpath("valid/images").mkdir(exist_ok=True, parents=True)
        base_path.joinpath("valid/labels").mkdir(exist_ok=True, parents=True)

        with open(yolo_base_path.joinpath("class_names.txt"), "r+") as f:
            class_names = json.load(f)

        data_yaml = {
            "train": "./train/images",
            "val": "./valid/images",
            "nc": len(class_names),
            "names": class_names,
        }
        with open(base_path.joinpath("data.yaml"), "w+") as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=False)

    def transform_yolo_to_hasty(self, param):
        raise ValueError("implement me!")

    def set_class_filter(self, keep_classes):
        self.class_filter = keep_classes


def delete_hidden_files(base_path):
    """pass"""
    for file in set(base_path.rglob("._*")):
        file.unlink()
    return True


def whole_workflow(
        base_path: Path,
        hasty_annotations_labels_zipped,
        hasty_annotations_images_zipped,
        amount_training_images=(5000000,),
        backgrounds=(0,),
        fulls=False,
        folds=5,
        slice_size=1280,
        prefix="rtu_tds_150_400",
        training_data_location="/training_data",
        keep_classes=None,
        extension="JPG",
        with_augmentations = 0):
    """
    with downloaded annotations prepare them to training data for yolo

    @type base_path: Path
    @param base_path
    @param hasty_annotations_labels_zipped:
    @param hasty_annotations_images_zipped:
    @return:
    :param amount_training_images:
    :param prefix:
    :param slice_size:
    :param folds:
    :param fulls:
    :param backgrounds:
    :param max_training_images:
    :param prefix:
    :param slice_size:
    :param folds:
    :param fulls:
    :param backgrounds:
    :param max_training_images:
    """

    tiled_path = base_path.joinpath("tiled")

    hYC = HastyToYOLOConverter(
        base_path=base_path,
        hasty_annotations_labels_zipped=hasty_annotations_labels_zipped,
        hasty_annotations_images_zipped=hasty_annotations_images_zipped,
    )
    hYC.set_class_filter(keep_classes)
    if fulls:
        hYC.prepare_zip_files()
        delete_hidden_files(base_path)
        hYC.merge_image_datasets()
    delete_hidden_files(base_path)
    hYC.transform_annotation()

    ## tile the images in a folder
    if fulls:
        results = list(
            YoloTiler.tile_folders(
                images_path=hYC.yolo_image_path,
                labels_path=hYC.yolo_label_path,
                tiled_path=tiled_path,
                slice_size=slice_size,
                extension=extension
            )
        )



    for n_background in backgrounds:
        for n_with_objects in amount_training_images:
            DATASET_NAME = f"{prefix}_{n_with_objects}_{n_background}"
            splitted_tiled_path = base_path.joinpath(DATASET_NAME)

            if n_background:
                """take as many images from the background folder"""
                k_folds_background, stats_bg = YoloTiler.splitter2(
                    source=tiled_path.joinpath(YoloTiler.IMAGES_WITHOUT_OBJECTS),
                    target=splitted_tiled_path,
                    consistent_ids=hYC.consistent_ids,
                    ext=extension,
                    amount_train_images=n_background,
                    folds=folds,
                )

            k_folds, stats = YoloTiler.splitter2(
                source=tiled_path.joinpath(YoloTiler.IMAGES_WITH_OBJECTS),
                target=splitted_tiled_path,
                consistent_ids=hYC.consistent_ids,
                ext=extension,
                amount_train_images=n_with_objects,
                folds=folds,
                training_data_location=training_data_location,
                with_augmentations=with_augmentations
            )
            # TODO augmentations?

            logger.info(
                f"Everything is ready. The Data can be found at {splitted_tiled_path}"
            )
            i = 1
            for fold in k_folds:
                yield {
                    "ds_sz": n_with_objects,
                    "bg": n_background,
                    "ds": DATASET_NAME,
                    "fold": i,
                }
                i += 1
