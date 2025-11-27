import pandas as pd

from active_learning.util.converter import coco2yolo
from loguru import logger

from pathlib import Path

from com.biospheredata.helper.filenames import get_dataset_image_merged_filesname, get_dataset_image_merged_filesname_v2
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def get_label_class_mapping(hA: HastyAnnotationV2) -> dict:
    index = 0
    mapping = {}

    for value in hA.label_classes:
        mapping[value.class_name] = index
        index += 1

    return mapping

def transform_hasty_to_yolo(hA: HastyAnnotationV2, yolo_base_path:Path, yolo_label_path:Path):
    """
    transform the image coordinate from hasty.ai json to yolo.

    HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point
    YOLO Format is x,y of the center point, width, height
    https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

    :param annotation_file:
    :return:
    """

    yolo_base_path.mkdir(parents=True, exist_ok=True)
    yolo_label_path.mkdir(parents=True, exist_ok=True)

    class_mapping = get_label_class_mapping(hA)

    df_all_boxes_list = []

    for image in hA.images:
        image_id = image.image_id,
        image_name = image.image_name
        dataset_name = image.dataset_name
        image_name_split = image_name.split(".")
        image_name_split[-1] = "txt"
        label_name = ".".join(image_name_split)
        logger.info(label_name)

        image_height = image.height
        image_width = image.width
        labels = image.labels
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



            df_all_boxes_list.append(df_boxes)
            # write the annotation to disk

            new_label_name = yolo_label_path.joinpath(
                f"{get_dataset_image_merged_filesname(dsn=dataset_name, imn=label_name, iid=image['image_id'])}"
            )
            new_label_name = yolo_label_path.joinpath(
                f"{get_dataset_image_merged_filesname_v2(dsn=dataset_name, imn=label_name)}"
            )
            with open(new_label_name, "w") as f:
                df_boxes[["class_id", "x_yolo", "y_yolo", "w_yolo", "h_yolo"]].to_csv(
                    f, index=False, header=False, sep=" "
                )

            yolo_labels_names.append(new_label_name)

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



if __name__ == "__main__":
    hA = HastyAnnotationV2.from_file(Path("/raid/cwinkelmann/training_data/eikelboom2019/eikelboom_512_overlap_0_ebFalse/eikelboom_train/train/hasty_format_crops_512_0.json"))

    yolo_base_path = Path("eikelboom2019_yolo")
    yolo_label_path = Path("eikelboom2019_yolo_labels")
    transform_hasty_to_yolo(hA, yolo_base_path, yolo_label_path)