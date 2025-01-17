"""
rename images from DJI_XXXX.JPG to mission_date_mission_name_XXXX.JPG

Example:
    Parent Folder Name: Isabela
    Folder_name: ISBU03_19012023
    File_name before: DJI_0001.JPG
    File_name after: Isa_ISBU03_DJI_0132_19012023.JPG

    That format after is the Following


"""
import shutil

import pandas as pd
from pathlib import Path

import typing

from loguru import logger

from active_learning.config.mapping import prefix_mapping




def image_plausibility_check(image: Path):
    """

    :param image:
    :return:
    """
    # TODO check the date of the image
    # TODO check certain metadata like ISO, exposure time, aperture


def rename_single_image(island, mission_folder: str, image: Path):
    # test if the current image fits the schema

    image_name_stem = image.stem
    image_suffix = image.suffix

    assert isinstance(mission_folder, str), f"Mission folder {mission_folder} should be a string"

    image_name_split = image_name_stem.split("_")

    if (len(image_name_split) == 2 and "DJI" == image_name_split[0]
            and len(image_name_split[1]) == 4):
        # logger.info(f"Image: {image.name} is propbable a drone image directly from the drone")

        prefix = prefix_mapping[island]
        main_island, subisland = prefix.split("_")
        site_code, mission_date_ddmmYYYY = mission_folder.split("_")
        assert len(mission_date_ddmmYYYY) == 8, f"Mission date {mission_date_ddmmYYYY} should have 8 digits"
        assert mission_folder.startswith(
            subisland), f"Mission folder {mission_folder} does not start with the correct subisland prefix"

        # new_name Isa_ISBU03_DJI_0132_19012023
        new_image_name = f"{main_island}_{site_code}_{image_name_stem}_{mission_date_ddmmYYYY}{image_suffix}"

        return new_image_name

    if len(image_name_split) == 6 and "DJI" == image_name_split[2] and len(image_name_split[3]) == 4:
        # logger.info(f"Image {image.name} has probably a drone identifier in the name")
        # looking for that stuff Flo_FLM05_DJI_0111_030221_condor.SRT
        if image_name_split[5] in ["condor", "hawk"]:
            del image_name_split[5]
            image_name_split[4] = fix_date_format(image_name_split[4])
            image_name = "_".join(image_name_split)
            image_name = f"{image_name}{image_suffix}"
            return image_name
        else:
            logger.error(f"Image {image} has no valid drone identifier in the name")
            raise ValueError(f"Image {image} has no valid drone identifier in the name")


    if len(image_name_split) == 5 and "DJI" == image_name_split[2] and len(image_name_split[3]) == 4:
        logger.info(f"Image {image} already has the correct format")
        # looking for that stuff Flo_FLM05_DJI_0111_030221_condor.SRT
        return image.name

def run_renaming(df_changed_images, new_path):
    for _, row in df_changed_images.iterrows():
        full_old_image_path = new_path / row.island / row.mission_folder / row['old_name']
        full_new_image_path = new_path / row.island / row.mission_folder / row['new_name']
        logger.info(f"Moving {full_old_image_path} to {full_new_image_path}")
        # Create destination directory if it does not exist.
        shutil.move(full_old_image_path, full_new_image_path)

def rename_images(island, mission_folder, images_list: typing.List[Path]):
    """
    Rename images from DJI_XXXX.JPG to mission_date_mission_name_XXXX.JPG
    :param folder_name: str
    :param images_folder: Path
    :return: None
    """
    new_image_name_list = []

    for image in images_list:
        new_image_name = rename_single_image(island, mission_folder, image)

        new_image_name_list.append(new_image_name)

    df_rename = pd.DataFrame({"old_name": [image.name for image in images_list],
                              "island": island,
                              "mission_folder": mission_folder,
                              "new_name": new_image_name_list})

    return df_rename

def get_island_from_folder(folder: Path) -> str:
    """
    Get the island from the folder
    :param folder:
    :return:
    """
    mission_code_with_date = folder.parts[-1]
    island = folder.parts[-2]
    if not island in prefix_mapping.keys():
        raise ValueError(f"Island {island} not in prefix mapping")

    return island, mission_code_with_date

def raw_folder_to_mission(raw_folder: Path) -> pd.DataFrame:
    """

    :param path_raw_data:
    :return:
    """
    island, mission_code_with_date = get_island_from_folder(raw_folder)
    site_code, date = mission_code_with_date.split("_")

    if len(date) != 8:
        raise ValueError(f"Date {date} does not have 8 digits")
    images_list = list(raw_folder.glob("*.JPG"))
    df_rename = rename_images(island,
                              mission_folder=mission_code_with_date,
                              images_list=images_list)

    df_rename["full_path"] = raw_folder

    return df_rename

def rename_incorrect_folders(base_path: Path, new_base_path: Path):
    """

    :param base_path:
    :param new_base_path:
    :return:
    """
    islands = [i.stem for i in base_path.glob("*")]
    print(f"There are these islands: {islands}")
    # find each Folder which should contain Missions

    mission_folders = list(base_path.glob("*/*/"))
    mission_names = [m.stem for m in mission_folders]

    # get the date of each
    dates = [d.split("_") for d in mission_names]

    seperated_dates = [d[-1] for d in dates]

    def check_date_format(date_str: str) -> int:
        return len(date_str)

    string_lenth = [check_date_format(d) for d in seperated_dates]


    df_data = pd.DataFrame({"Mission": mission_names,"Date": seperated_dates,
                            "Date Format": string_lenth,
                            "island": [x.parent.stem for x in mission_folders],
                            "full_path_old": mission_folders})

    df_data_incorrect_date_mask = df_data["Date Format"] == 6

    df_data_changed = df_data[df_data_incorrect_date_mask]
    df_data_changed['Date_fixed'] = df_data_changed['Date'].apply(fix_date_format)

    df_data_changed["new_folder_path"] = df_data_changed.apply(
        lambda row: new_base_path / row['island'] / f"{row['Mission'].split('_')[0]}_{row['Date_fixed']}",
        axis=1
    )

    return df_data_changed


def move_folders(df_data_changed):
    for row in df_data_changed.itertuples(index=False):
        # Convert path columns to Path objects (if they aren't already)
        src = Path(row.full_path_old)
        dst = Path(row.new_folder_path)

        # Ensure that the parent directory for the destination exists.
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Copying {src} to {dst}")
        # Recursively copy the folder from source to destination.

        shutil.copytree(src, dst)

        print(f"Copied {src} to {dst}")


def fix_date_format(date_str: str) -> str:
    """
    If date_str is in ddmmyy format (6 digits), then convert it to ddmmyyyy format by inserting "20".
    For example, '150321' becomes '15032021'.
    If date_str is not 6 characters long, it is returned unchanged.
    """
    if len(date_str) == 6:
        # Extract day, month, and year portions.
        day = date_str[:2]
        month = date_str[2:4]
        year = date_str[4:]
        # Insert "20" (assuming the year is in the 2000s).
        return f"{day}{month}20{year}"
    return date_str
