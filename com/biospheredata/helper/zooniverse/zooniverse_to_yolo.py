import glob
import json
import random
import shutil
from matplotlib import pyplot as plt
from pathlib import Path
from csv import reader
import pandas as pd
from loguru import logger
from matplotlib import image
from matplotlib import pyplot as plt
import statistics
## first user marks
from matplotlib.pyplot import figure
import random

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import statistics

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from com.biospheredata.visualization.visualize_result import visualize_bounding_boxes
import hashlib

BOX_SIZE = 60

trustworthy_users = [
    "Pamelavans"
    "robert1601",
    "Darkstar1977",
    "H.axon",
    "Quynhanhdo"
    "Taylor_Q"
    "databanana"
    "Heuvelmans"
    "Big_Ade"
    "babyemperor"
    "HW1881"
]

def read_zooniverse_annotations(annotations_source, phase_tag, cache_dir):
    """

    @param annotations_source:
    @param phase_tag:
    @param cache_dir:
    @return:
    """
    cache_file = cache_dir.joinpath("cache_zooniverse_annotations.json")

    # index for certain information.
    idx_USER_NAME = 1
    idx_USER_ID = 2
    idx_PHASE = 5
    idx_LABEL_TIME = 7
    idx_TASK_INFORMATION = 11
    idx_IMAGE_INFORMATION = 12

    n = 0

    phases = []

    reduced_dataset = []
    flat_dataset = []
    if not Path(cache_file).is_file():
        logger.info(f"Cache file will be built, {cache_file}")

        # open file in read mode
        with open(annotations_source, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:

                    phase_information = row[idx_PHASE]
                    if phase_information not in phases:
                        phases.append(phase_information)
                        logger.info(f"found a new phase tag: {phase_information}")
                    user_id = row[idx_USER_ID]
                    user_name = row[idx_USER_NAME]
                    label_time = row[idx_LABEL_TIME]
                    phase_information_passed = False
                    if phase_information == phase_tag:
                        # logger.info(phase_information)
                        phase_information_passed = True
                        n = n + 1
                        if n % 10000 == 0:
                             logger.info(f"row number {n} of many, row itself: {row}")

                    task_information = json.loads(row[idx_TASK_INFORMATION])
                    if task_information[0][
                        "task"] == "T0" and phase_information_passed:  ## are there iguanas? ## FIXME do not commit this
                        if task_information[0]["value"] == "Yes":
                            # yes there are'

                            if len(task_information[1]["value"]) > 0:
                                image_information = json.loads(row[idx_IMAGE_INFORMATION])
                                for key, value in image_information.items():

                                    flight_site_code = value.get("flight_site_code",
                                                                 value.get("flight_code", value.get("Flight")))
                                    if flight_site_code is None:
                                        flight_site_code = value.get("site")
                                    image_name = value.get("image_name", value.get("Image_name"))
                                    # if image_name == "SRPB05-25-1-1_50.jpg":
                                    marks = task_information[1]['value']
                                    # logger.info(f"Flight sight code: {flight_site_code} and filename {image_name} with {len(task_information[1]['value'])} iguanas on it")

                                    ## TODO filter for the right class
                                    for mark in marks:
                                        flat_dataset.append({"flight_site_code": flight_site_code,
                                                             "image_name": image_name,
                                                             # "mark": mark,
                                                             "x": mark["x"],
                                                             "y": mark["y"],
                                                             "tool_label": mark["tool_label"],
                                                             "phase_tag": phase_information
                                                             })

                                    reduced_dataset.append(
                                        {
                                            "phase_tag": phase_information,
                                            "flight_site_code": flight_site_code,
                                            "image_name": image_name,
                                            "marks": marks,
                                            "user_id": user_id,
                                            "user_name": user_name,
                                            "label_time": label_time
                                        })

        logger.info("generating flat dataframe from the dataset")
        flat_dataset = pd.DataFrame(flat_dataset)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        flat_dataset.to_json(f"{Path(cache_dir).joinpath('flat_dataset.json')}")
        flat_dataset.to_csv(f"{Path(cache_dir).joinpath('flat_dataset.csv')}")
        logger.info(f"Wrote flat dataset to {cache_dir}")

        with open(cache_file, 'w+') as f:
            json.dump(reduced_dataset, f)
            logger.info(f"Wrote cache_file to {cache_file}")

    else:
        logger.info(f"cache file exists, {cache_file}")

        with open(cache_file, 'r') as f:
            reduced_dataset = json.load(f)

    return reduced_dataset


def read_zooniverse_annotations_v2(annotations_source, phase_tags, cache_dir):
    """

    @param annotations_source:
    @param phase_tag: "Iguanas 1st launch" or "Iguanas 2nd launch" or "Iguanas 3rd launch"
    @param cache_dir:
    @return:
    """
    cache_file = cache_dir.joinpath("cache_zooniverse_annotations.json")
    # https://help.zooniverse.org/next-steps/data-exports/
    # index for certain information.
    idx_USER_NAME = 1
    idx_USER_ID = 2
    idx_PHASE = 5
    idx_LABEL_TIME = 7
    idx_USER_INFORMATION = 10
    idx_TASK_INFORMATION = 11
    idx_IMAGE_INFORMATION = 12
    idx_SUBJECT_IDS = 13

    # TASK LABELS PHASE 1
    TASK_LABEL_ARE_THERE_ANY_IGUANAS = 0
    TASK_LABEL_MARK_ALL_IGUANAS = 1
    TASK_LABEL_DIFFICULTY_MARKING_IGUANAS = 2
    TASK_LABEL_ANYTHING_ELSE = 3



    n = 0

    phases = []

    reduced_dataset = []
    flat_dataset = []
    if not Path(cache_file).is_file():
        logger.info(f"Cache file will be built, {cache_file}")

        # open file in read mode
        with open(annotations_source, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:

                    phase_information = row[idx_PHASE]
                    if phase_information not in phases:
                        phases.append(phase_information)
                        logger.info(f"found a new phase tag: {phase_information}")
                    user_id = row[idx_USER_ID]
                    user_name = row[idx_USER_NAME]
                    label_time = row[idx_LABEL_TIME]
                    user_information = json.loads(row[idx_USER_INFORMATION])
                    labeling_started_at = user_information["started_at"]
                    labeling_finished_at = user_information["finished_at"]

                    phase_information_passed = False
                    if phase_information in phase_tags:
                        phase_information_passed = True
                        n = n + 1
                        if n % 10000 == 0:
                            logger.info(f"phase: {phase_information}, row number {n} of many, row itself: {row}")

                    task_information = json.loads(row[idx_TASK_INFORMATION])
                    subject_id = int(row[idx_SUBJECT_IDS])
                    ## Are there any Iguanas?
                    if task_information[TASK_LABEL_ARE_THERE_ANY_IGUANAS]["task"] == "T0" and phase_information in phase_tags:  ## is there anything?
                        if task_information[TASK_LABEL_ARE_THERE_ANY_IGUANAS]["value"] == "Yes":
                            # yes there are

                            if len(task_information[TASK_LABEL_MARK_ALL_IGUANAS]["value"]) > 0:
                                image_information = json.loads(row[idx_IMAGE_INFORMATION])
                                for key, image_information_value in image_information.items():

                                    ## this has been renamed quite a bit
                                    flight_site_code = image_information_value.get("flight_site_code",
                                                                 image_information_value.get("flight_code", image_information_value.get("Flight")))
                                    if flight_site_code is None:
                                        flight_site_code = image_information_value.get("site")


                                    image_name = image_information_value.get("image_name", image_information_value.get("Image_name", image_information_value.get("Filename")))
                                    if image_name is None:
                                        print(image_information_value)
                                    # if image_name == "SRPB05-25-1-1_50.jpg":
                                    marks = task_information[TASK_LABEL_MARK_ALL_IGUANAS]['value']
                                    # logger.info(f"Flight sight code: {flight_site_code} and filename {image_name} with {len(task_information[1]['value'])} iguanas on it")

                                    ## TODO filter for the right class
                                    for mark in marks:
                                        flat_dataset.append({"flight_site_code": flight_site_code,
                                                             "image_name": image_name,
                                                             "subject_id": subject_id,
                                                             # "mark": mark,
                                                             "x": mark["x"],
                                                             "y": mark["y"],
                                                             "tool_label": mark["tool_label"],
                                                             "phase_tag": phase_information,
                                                             "user_id": user_id,
                                                             "user_name": user_name,

                                                             })
                                        # TODO add more information like the time for labellin etc

                                    reduced_dataset.append(
                                        {
                                            "phase_tag": phase_information,
                                            "flight_site_code": flight_site_code,
                                            "image_name": image_name,
                                            "subject_id": subject_id,
                                            "marks": marks,
                                            "user_id": user_id,
                                            "user_name": user_name,
                                            "label_time": label_time,
                                            "labeling_started_at": labeling_started_at,
                                            "labeling_finished_at": labeling_finished_at,
                                        })

        logger.info("generating flat dataframe from the dataset")
        flat_dataset = pd.DataFrame(flat_dataset)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        flat_dataset.to_json(f"{Path(cache_dir).joinpath('flat_dataset.json')}")
        flat_dataset.to_csv(f"{Path(cache_dir).joinpath('flat_dataset.csv')}")
        logger.info(f"Wrote flat dataset to {cache_dir}")

        with open(cache_file, 'w+') as f:
            json.dump(reduced_dataset, f)
            logger.info(f"Wrote cache_file to {cache_file}")

    else:
        logger.info(f"cache file exists, {cache_file}")

        with open(cache_file, 'r') as f:
            reduced_dataset = json.load(f)
        with open(f"{Path(cache_dir).joinpath('flat_dataset.json')}", 'r') as f:
            flat_dataset = json.load(f)

    dict_result = {}
    dict_result["reduced_dataset"] = reduced_dataset
    dict_result["flat_dataset"] = flat_dataset

    return dict_result


def build_zooniverse_annotation_table(dict_result, phase_tags, output_dir):
    """
    take the results from
    :param reduced_dataset:
    :param flat_dataset:
    :return:
    """

    df_reduced_dataset = pd.DataFrame(dict_result["reduced_dataset"])
    df_flat_dataset = pd.DataFrame(dict_result["flat_dataset"])

    df_reduced_dataset["user_id"] = pd.to_numeric(df_reduced_dataset["user_id"])
    df_flat_dataset["user_id"] = pd.to_numeric(df_flat_dataset["user_id"])
    df_reduced_dataset = df_reduced_dataset[df_reduced_dataset["user_id"] > 0]
    df_flat_dataset = df_flat_dataset[df_flat_dataset["user_id"] > 0]

    phase_stats = {}

    for phase_tag in phase_tags:
        stats = {}

        df_rd_subset = df_reduced_dataset[df_reduced_dataset.phase_tag == phase_tag]
        df_fd_subset = df_flat_dataset[df_flat_dataset["phase_tag"] == phase_tag]



        df_rd_subset["labeling_started_at"] = pd.to_datetime(df_rd_subset["labeling_started_at"])
        df_rd_subset["labeling_finished_at"] = pd.to_datetime(df_rd_subset["labeling_finished_at"])
        df_rd_subset["labeling_duration"] = df_rd_subset["labeling_finished_at"] - df_rd_subset["labeling_started_at"]
        df_rd_subset["trusted_user"] = df_rd_subset["user_name"].isin(trustworthy_users)
        df_rd_subset["labeling_finished_at_day"] = df_rd_subset['labeling_finished_at'].apply(
            lambda x: x.strftime('%d%m%Y'))


        df_rd_subset["labeling_finished_at_day"] = df_rd_subset['labeling_finished_at'].apply(
            lambda x: x.strftime("%Y-%m-%d"))

        df_labeling_finished_at_day = df_rd_subset.groupby(by="labeling_finished_at_day").count()["phase_tag"].reset_index().sort_values(
            "labeling_finished_at_day", ascending=False)

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))

        X = df_labeling_finished_at_day["labeling_finished_at_day"]
        Y1 = df_labeling_finished_at_day["phase_tag"]
        plt.bar(x=X,
                height=Y1, facecolor='#9999ff', edgecolor='white')
        ax.xaxis.set_ticks(np.arange(0, len(X), 20), rotation=45, ha='right')
        ax.xaxis.set_ticklabels(X[np.arange(0, len(X), 20)], rotation=45, ha='right')
        ax.set_title('Daily Count of Labelled Images')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        plt.subplots_adjust(bottom=0.25)
        # for x, y in zip(X, Y1):
        #     plt.text(x + 0.4, y + 0.05, '%.2f' % str(y), ha='center', va='bottom')

        if output_dir is not None:
            plt.savefig(output_dir.joinpath(f"daily_count_image_labels_{phase_tag}"))
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        merged_dataset_grouped = df_rd_subset.groupby(by="image_name")["user_id"].agg(["count"])
        plt.hist(merged_dataset_grouped["count"], bins=25)

        ax.set_ylabel('Count')

        ax.set_xlabel('Amount of Users per Image')
        # ax.set_title('Amount')
        # ax.legend(title='Fruit color')
        if output_dir is not None:
            plt.savefig(Path(output_dir).joinpath(f"histogram_users_{phase_tag}.png"))
        # plt.show()
        plt.close()


        df_images = pd.DataFrame(df_rd_subset.groupby(by="image_name")["user_id"].count()).reset_index()


        stats["Images"] = df_rd_subset["image_name"].nunique()
        stats["Unique Users"] = df_fd_subset["user_name"].nunique()
        stats["Individual Marks by Users"] = df_fd_subset["tool_label"].count()

        stats["First Annotation"] = df_rd_subset["labeling_started_at"].min().strftime("%Y-%m-%d")
        stats["Last Annotation"] = df_rd_subset["labeling_finished_at"].max().strftime("%Y-%m-%d")
        stats["Total Labeling Duration [h]"] = int(df_rd_subset["labeling_duration"].sum().total_seconds()/3600)
        stats["Average Labeling Duration [s]"] = int(df_rd_subset["labeling_duration"].mean().total_seconds())
        stats["Days Live"] = int((df_rd_subset["labeling_started_at"].max() - df_rd_subset["labeling_finished_at"].min()).total_seconds() / (3600 * 24))

        stats["Images with more than 2 active users"] = len(df_images[df_images["user_id"] > 2])
        stats["Images with less or equal 2 active users"] = len(df_images[df_images["user_id"] <= 2])
        stats["Images Labelled by Trusted Users"] = len(df_rd_subset[df_rd_subset["trusted_user"] == True])

        stats["Average Amount of Marks per Image"] = df_fd_subset.groupby(by="image_name")["user_id"].count().mean()

        phase_stats[phase_tag] = stats

    pd.DataFrame(phase_stats).to_latex()
    return pd.DataFrame(phase_stats)


def transform_zooniverse_to_yolo(data, yolo_base_path, train_val_split=1, class_filter=None):
    """
    TODO Keep in mind there multiple users with different results.
    transform the image coordinate from hasty.at to yolo
    HASTY Format: "x1", "y1", "x2", "y2" - bottom left point, top right point - TODO double check this
    YOLO Format is x,y of the center point, width, height
    https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h


    :return:
    """
    consistent_ids = []
    filtered_and_processed_data = data
    data_key = 0
    for image in data:
        try:
            image_name = image["image_name"]
            user_id = int(image["user_id"])
            image_path = image["image_path"]
            if image_name is None:
                continue

            image_name_split = image_name.split(".")
            image_name_split[-1] = "txt"
            label_name = ".".join(image_name_split)
            # logger.info(label_name)

            image_height = image["height"]
            image_width = image["width"]

            # labels = image["labels"]
            if len(image["marks"]) > 0:

                df_boxes = pd.DataFrame(image["marks"])

                df_boxes = df_boxes[df_boxes.tool_label.isin(class_filter)]

                if df_boxes.shape[0] > 0:
                    for class_label in list(df_boxes["tool_label"].unique()):
                        if class_label in consistent_ids:
                            pass
                        else:
                            consistent_ids.append(class_label)

                    df_boxes = df_boxes.merge(pd.DataFrame(consistent_ids,
                                                           columns=["tool_label"]).reset_index(),
                                              on='tool_label', how='left')

                    df_boxes["h"] = BOX_SIZE
                    df_boxes["w"] = BOX_SIZE

                    ## TODO visualize the boxes before transformation

                    df_boxes["class"] = df_boxes["index"]
                    df_boxes["x_yolo"] = df_boxes["x"] / image_width
                    df_boxes["y_yolo"] = df_boxes["y"] / image_height

                    df_boxes["w_yolo"] = df_boxes["w"] / image_width
                    df_boxes["h_yolo"] = df_boxes["h"] / image_height

                    yolo_base_path.mkdir(exist_ok=True, parents=True)
                    ## decide if the image is a training or validation image
                    if random.random() < train_val_split:  # train
                        yolo_train_split_path = yolo_base_path.joinpath("train")

                    else:
                        yolo_train_split_path = yolo_base_path.joinpath("valid")
                    yolo_train_split_path.mkdir(exist_ok=True)
                    yolo_train_split_path.joinpath("labels").mkdir(exist_ok=True)

                    output_label_path = yolo_train_split_path.joinpath("labels").joinpath(f"{user_id}_{label_name}")
                    with open(output_label_path, 'w') as f:
                        logger.info(f"write yolo label file to {output_label_path}")
                        df_boxes[["class", "x_yolo", "y_yolo", "w_yolo", "h_yolo"]] \
                            .to_csv(f, index=False, header=False, sep=' ')

                        yolo_train_split_path.joinpath("images").mkdir(exist_ok=True)
                        # move the image in question from the source to the destination folder

                        output_image_path = yolo_train_split_path.joinpath("images").joinpath(image_name)
                        shutil.copyfile(image_path,
                                        output_image_path)
                        logger.info(f"write image file to {output_image_path}")

                    debug_path = yolo_base_path.joinpath("debug")
                    if debug_path:
                        visualize_bounding_boxes(imname=f"{image_name}", label_name=label_name,
                                                 basepath=yolo_train_split_path, output_path=debug_path,
                                                 suffix_images="images", suffix_labels="labels")

        except Exception as e:
            logger.error(f"An exception happened: {e}")
            pass

    with open(yolo_base_path.joinpath("class_names.txt"), 'w+') as f:
        json.dump(consistent_ids, f)

    return True


def get_all_image_paths(image_source: Path, cache_dir: Path) -> pd.DataFrame:
    """
    search for images in subfolders which we use to join the real path to the dataframe with classification report

    :type image_source: Path
    :type cache_dir: Path
    :param cache_dir:
    :param image_source:
    :return:
    """
    if image_source is None:
        return None
    metadata_file = cache_dir.joinpath("image_paths_metadata.json")

    if not Path(metadata_file).is_file():
        logger.info(f"WAIT: {metadata_file} file doesn't exist")

        # image_list = glob.glob(str(image_source.joinpath("*/*/*.jpg")))
        image_list = glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)

        if len(image_list) == 0:
            logger.warning(f"Found {len(image_list)} images in the folder {image_source}")

        images_split_list = [Path(x).parts for x in image_list]
        image_list = [{"mission_name": x[-2],
                       "image_name": x[-1],
                       "image_path": Path(*list(x))} for x in images_split_list]

        image_dict = {x[-1]: {"mission_name": x[-2],
                              "image_name": x[-1],
                              "image_path": str(Path(*list(x)))} for x in images_split_list}


        from PIL import Image
        for key, value in image_dict.items():
            # logger.info(f"load image: {value['image_path']}")
            im = Image.open(value["image_path"])
            width, height = im.size
            image_dict[key]["width"] = width
            image_dict[key]["height"] = height

            im.close()

        with open(metadata_file, 'w+') as f:
            json.dump(image_dict, f)
    else:
        logger.info(f"OK: {metadata_file} file does exist")

        with open(metadata_file, 'r') as f:
            image_dict = json.load(f)

    ## FIXME the image_list and the image_dict have a different length, meaning there are images in there twice

    logger.info("done with processing Zooniverse data.")
    return pd.DataFrame(image_dict).T


def process_phase_1():
    # ## TODO read the annotations from the classification csv
    # annotations_source = "/media/christian/2TB/ai-core/data/iguanas_from_above/zooniverse_iguanas-from-above-classifications.csv"
    # image_source = Path("/media/christian/Elements/data/iguanas_from_above/2020 Jan/Zooniverse project/All Images/Images for Website/First launch Aug_2020/Images uploaded")
    # target_upfolder = Path("/media/christian/2TB/ai-core/data/iguanas_from_above/zooniverse/iguana_yolo_sliced_train_split")
    #
    # # TODO: this only works with the raw images because they have GPS coordinates - image_metadata_list = image_dict.apply(lambda  x: image_coordinates(x["image_path"]), axis=1)
    # image_dict = get_all_image_paths(image_source)
    # zooniverse_annotation_dataset = read_zooniverse_annotations(annotations_source=annotations_source, image_source=image_source)
    # zooniverse_annotation_dataset = pd.DataFrame(zooniverse_annotation_dataset)
    # merged_dataset = zooniverse_annotation_dataset.merge(image_dict, on='image_name', how='left')
    # merged_dataset =  merged_dataset.to_dict(orient="records")
    #
    # transform_zooniverse_to_yolo(merged_dataset, target_upfolder)
    #
    # logger.info(zooniverse_annotation_dataset)

    ### Version
    ## TODO read the annotations from the classification csv
    annotations_source = "/media/christian/2TB/ai-core/data/iguanas_from_above/iguanas-from-above-classifications_2022-03-28.csv"
    image_source = Path(
        "/media/christian/Elements/data/iguanas_from_above/2020 Jan/Zooniverse project/All Images/Images for Website/First launch Aug_2020/Images uploaded")
    target_upfolder = Path(
        "/media/christian/2TB/ai-core/data/iguanas_from_above/zooniverse/iguana_yolo_sliced_train_split")

    # TODO: this only works with the raw images because they have GPS coordinates - image_metadata_list = image_dict.apply(lambda  x: image_coordinates(x["image_path"]), axis=1)
    zooniverse_annotation_dataset = read_zooniverse_annotations(annotations_source=annotations_source,
                                                                phase_tag="Iguanas 2nd launch")

    # image_dict = get_all_image_paths(image_source)
    # zooniverse_annotation_dataset = pd.DataFrame(zooniverse_annotation_dataset)
    # merged_dataset = zooniverse_annotation_dataset.merge(image_dict, on='image_name', how='left')
    # merged_dataset =  merged_dataset.to_dict(orient="records")
    #
    # transform_zooniverse_to_yolo(merged_dataset, target_upfolder)

    logger.info(zooniverse_annotation_dataset)


def deduplicate_entries(merged_dataset):
    """
    Each image is marked multiple times. Once per user, N marks for N iguanas.

    :param merged_dataset:
    :return:
    """

    image_index_mapping = pd.DataFrame(merged_dataset)["image_name"].reset_index(drop=False)
    # iterate over the dataset and merge the marks
    compacted_marks_per_data_frame = []
    marks_per_data_frame = pd.DataFrame(merged_dataset).reset_index(drop=False).groupby('image_name')['marks'].apply(list).reset_index()

    for image_with_list_of_list_of_mark in marks_per_data_frame.to_dict(orient="records"):
        image_name = image_with_list_of_list_of_mark["image_name"]
        compacted_marks = []
        for mark_group in image_with_list_of_list_of_mark["marks"]:
            for mark in mark_group:
                compacted_marks.append(mark)

        compacted_marks_per_data_frame.append({"image_name": image_name, "marks": compacted_marks})

    return compacted_marks_per_data_frame

def process_zooniverse_phases(annotations_source: Path,
                              image_source: Path,
                              cache_folder: Path,
                              image_names=None,
                              subject_ids=None,
                              phase_tag="Iguanas 2nd launch",
                              filter_func = None) -> pd.DataFrame:
    """
    merge Zooniverse part 2 data with the image dictionary and visulise the marks

    @param annotations_source:
    @param image_source:
    @param cache_folder:
    @return:
    """

    zooniverse_annotation_dataset = read_zooniverse_annotations_v2(annotations_source=annotations_source,
                                                                   phase_tags=phase_tag,
                                                                   cache_dir=cache_folder)
    zooniverse_annotation_dataset = pd.DataFrame(zooniverse_annotation_dataset["reduced_dataset"])

    if image_names:
        # a filter is applied
        logger.info(f"filtering the image dataset for  {len(image_names)} images")
        zooniverse_annotation_dataset = zooniverse_annotation_dataset[
            zooniverse_annotation_dataset.image_name.isin(image_names)]
        image_string = "__".join(image_names).encode()
        m = hashlib.sha256()
        m.update(image_string)
        zooniverse_merged_dataset_path = cache_folder.joinpath(f"{m.hexdigest()}_zooniverse_joined_records.json")

    if subject_ids:
        logger.info(f"filtering the image dataset for  {len(subject_ids)} subject_ids")
        zooniverse_annotation_dataset = zooniverse_annotation_dataset[
            zooniverse_annotation_dataset.subject_id.isin(subject_ids)]


        difference = set(subject_ids).difference(set(zooniverse_annotation_dataset.subject_id.unique()))
        if len(difference) > 0:
            logger.warning(f"Some of the subjects ids you used filter are not present in the set. These are: {difference}")


        image_string = "__".join([str(x) for x in subject_ids]).encode()
        m = hashlib.sha256()
        m.update(image_string)
        zooniverse_merged_dataset_path = cache_folder.joinpath(f"{m.hexdigest()}_zooniverse_joined_records.json")

    if subject_ids is None or image_names is None:
        # all images are used.
        logger.info("process all images")
        zooniverse_merged_dataset_path = cache_folder.joinpath(f"zooniverse_joined_records.json")

    image_dict = get_all_image_paths(image_source, cache_dir=cache_folder)
    if image_dict is not None:
        merged_dataset = zooniverse_annotation_dataset.merge(image_dict, on='image_name', how='left')
    else:
        # this should work if the the images are not available
        merged_dataset = zooniverse_annotation_dataset
    #
    # # TODO check if this should be refactored
    # # only use records of which we found images
    # # merged_dataset = merged_dataset[merged_dataset.height > 0]
    # # only logged in users
    # merged_dataset[["user_id"]] = merged_dataset[["user_id"]].apply(pd.to_numeric)
    # merged_dataset = merged_dataset[merged_dataset.user_id > 0]

    with open(cache_folder.joinpath(zooniverse_merged_dataset_path), 'w') as f:
        logger.info(f"Writing finished dataset to {cache_folder.joinpath(zooniverse_merged_dataset_path)}")
        json.dump(merged_dataset.to_dict(orient="records"), f)

    if filter_func:
        merged_dataset = filter_func(merged_dataset)

    return merged_dataset



def get_metadata_partitions(merged_dataset, n=None, threshold=None):
    """
    return list of records for only one image

    threshold means how many marks should exists. Otherwise it is considered noise.

    :return:
    """
    records_for_one_image = {}
    for record in merged_dataset:
        im = record["image_name"]
        if im not in records_for_one_image:
            records_for_one_image[im] = []
        records_for_one_image[im].append(record)

    if threshold is None:
        records_for_one_image = [r for i, r in records_for_one_image.items()]
    else:
        records_for_one_image = [r for i, r in records_for_one_image.items() if len(r) > threshold]
    ## TODO the, above is wrong. Because there is a saved record even if no mark is placed
    # slice the list to a shorter one
    if n is not None:
        records_for_one_image = records_for_one_image[:n]


    return records_for_one_image



def plot_zooniverse_user_marks(metadata_records, image_path, output_path: Path = None):
    """
    plot all the marks done by Zooniverse users and return the marks

    :param metadata_records:
    :return:
    """
    figure(figsize=(12, 12), dpi=80)

    ## visualise the image
    data = image.imread(image_path)
    implot = plt.imshow(data)

    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

    # one color per user
    colors = get_colors(len(metadata_records))  # sample return:  ['#8af5da', '#fbc08c', '#b741d0']

    marker_dict = {
        'Adult Male alone': "o",
        'Others (females, young males, juveniles)': "2",
        'Partial iguana': 3,
        'Adult Male with a lek': 4,
        'Adult Male not in a lek': 5, # Is this the same as 'Adult Male alone'
     }

    n = 0
    labels_all = []
    for image_metadata in metadata_records:
        # to read the image stored in the working directory
        try:
            user_id = int(image_metadata["user_id"])
        except:
            user_id = -1
        image_name = image_metadata["image_name"]

        x_list = [int(dm["x"]) for dm in image_metadata["marks"]]
        y_list = [int(dm["y"]) for dm in image_metadata["marks"]]

        labels = [dm["tool_label"] for dm in image_metadata["marks"]]
        try:
            marker_labels = [marker_dict[v] for v in labels]
        except Exception:
            logger.error(f"These labels are not known: {labels}")
            raise KeyError()

        labels_all = labels_all + marker_labels
        # put a red dot, size 40, at 2 locations:

        ## TODO plot the objects by the label with different markers. So scatter has to be called once for each marker
        plt.scatter(x=x_list, y=y_list, c=colors[n], s=40)

        n = n + 1


    if output_path is not None:
        plt.savefig(f"{output_path.joinpath(image_name)}_markers.png") #./labelimg_{user_id}.png")
        logger.info(f"{output_path.joinpath(image_name)}_markers.png")
        # plt.show()
        plt.close()

    plt.close()

    return 0


def get_mark_overview(metadata_records):
    """

    group all the marks by the zooniverse volunteers

    :param metadata_records:
    :return:

    """

    import random


    n = 0
    annotations_count = []
    labels_all = []
    for image_metadata in metadata_records:
        # to read the image stored in the working directory
        x_list = [int(dm["x"]) for dm in image_metadata["marks"]]
        labels = [dm["tool_label"] for dm in image_metadata["marks"]]
        labels_all = labels_all + labels
        # put a red dot, size 40, at 2 locations:
        annotations_count.append(len(x_list))
        n += 1

    return annotations_count





def reformat_marks(metadata_record_FMO01_68):
    """
    get one part from the flat structure

    :param metadata_record_FMO01_68:
    :return:
    """
    flat_struct = []

    for metadata_record in metadata_record_FMO01_68:
        for mark in metadata_record["marks"]:
            marks = {
                "x": int(mark["x"]),
                "y": int(mark["y"]),
            }
            flat_struct.append(marks)

    df_structure = pd.DataFrame(flat_struct)

    return df_structure

def get_estimated_SCAN_localizations(
        df_marks,
        image_name,
        params,
        output_path = None):


    X = df_marks.to_numpy()
    X = StandardScaler().fit_transform(X)
    result = {}

    eps, min_samples = params

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    #print("Estimated number of clusters: %d" % n_clusters_)
    #print("Estimated number of noise points: %d" % n_noise_)

    # print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
    # print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
    # print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
    # print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
    # print(
    #     "Adjusted Mutual Information:"
    #     f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
    # )
    # print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_} for eps={eps} and min_samples={min_samples}")
    if output_path:
        plt.savefig(output_path.joinpath(f"{image_name}_dbscan_{eps}_{min_samples}.png"))
    # plt.show()

    result[f"dbscan_count"] = n_clusters_
    result[f"dbscan_noise"] = n_noise_

    result["image_name"] = image_name
    return result

def get_estimated_iguana_count_localizations(df_marks, annotations_count,
                                             image_name,
                                             output_path: Path = None):
    """
    estimate the location
    """




    silhouettes = []
    X = df_marks.to_numpy()
    if max(annotations_count) in (0, 1):
        logger.warning(f"It seems there can be only one cluster")
        return {
            "image_name": image_name,
            "median_count": statistics.median(annotations_count),
            "mean_count": statistics.mean(annotations_count),
            "mode_count": statistics.mode(annotations_count),
            "sillouette_count": None
        }

    range_n_clusters = range(2, min(len(X), max(annotations_count)+2))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes.append(silhouette_avg)

        # logger.info(
        #     f"For n_clusters = {n_clusters}. The average silhouette_score is : {silhouette_avg}"
        # )
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Image pixel in X - Dimension")
        ax2.set_ylabel("Image pixel in Y - Dimension")

        plt.suptitle(
            f"Silhouette analysis for KMeans clustering on {image_name} n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

        if output_path:
            logger.info(f"save figure: {output_path.joinpath(image_name)}_n={n_clusters}.png")
            plt.savefig(f"{output_path.joinpath(image_name)}_n={n_clusters}.png")
            plt.close()

    # plt.show()
    plt.close()
    logger.info(f"finished annotations count for {image_name}")


    logger.info("visualise the optimal cluster count.")

    try:
        fig, ax = plt.subplots(1, 1)
        plt.plot(range_n_clusters, silhouettes)
        print(max(silhouettes))
        max_sillouette_score = max(zip(silhouettes, range_n_clusters))[1]

        plt.axvline(x=max_sillouette_score, color='b', label='axvline - full height')
        plt.text(max_sillouette_score - 3.5, 1.5, s=f"amount of cluster, max. sillouette score: {round(max_sillouette_score) ,3}",
                 bbox=dict(facecolor='red', alpha=0.5))
        if output_path is not None:
            plt.savefig(f"{output_path.joinpath(image_name)}_optimal_sillouette_score.png")
            logger.info(f"save figure: {output_path.joinpath(image_name)}_optimal_sillouette_score.png")
        # plt.show()
        plt.close()
    except Exception as e:
        logger.error(e)

    try:
        return {
            "image_name": image_name,
            # "median_count": statistics.median(annotations_count),
            # "mean_count": statistics.mean(annotations_count),
            # "mode_count": statistics.mode(annotations_count),
            "sillouette_count": max(zip(silhouettes, range_n_clusters))[1]
        }

    except ValueError as e:
        logger.error({
            "image_name": image_name,
            # "median_count": statistics.median(annotations_count),
            # "mean_count": statistics.mean(annotations_count),
            # "mode_count": statistics.mode(annotations_count),
            "sillouette_count": max(annotations_count)

        })
        return {
            "image_name": image_name,
            "median_count": statistics.median(annotations_count),
            "mean_count": statistics.mean(annotations_count),
            "mode_count": statistics.mode(annotations_count),
            "sillouette_count": max(annotations_count)
        }




if __name__ == '__main__':
    from pathlib import Path

    annotations_source = Path("/home/christian/Documents/iguanas/Zooniverse/iguanas-from-above-classifications.csv")
    # image_source = Path("/media/christian/Elements1/data/iguanas_from_above/Zooniverse_Phase_2_test_set/")
    image_source = Path(
        "/home/christian/Insync/christian.winkelmann@gmail.com/Google Drive/Datasets/IguanasFromAbove/Zooniverse Images Phase 2/")

    target_upfolder = Path(
        "/home/christian/Insync/christian.winkelmann@gmail.com/Google Drive/Datasets/IguanasFromAbove/Zooniverse Phase 2 Analysis")

    ## filter the annotations by this images for easier debugging
    image_name = "ESCG02-1_13.jpg"

    process_zooniverse_phases(annotations_source=annotations_source,
                              image_source=image_source,
                              cache_folder=target_upfolder,
                              image_name=image_name
                              )
