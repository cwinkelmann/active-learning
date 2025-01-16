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

from active_learning.database import fix_date_format, rename_incorrect_folders, move_folders

prefix_mapping = {
    "Española": "Esp_E",
    "Fernandina": "Fer_FE",
    "Floreana": "Flo_FL",
    "Genovesa": "Gen_G",
    "Isabela": "Isa_IS",
    "Marchena": "Mar_M",
    "Pinta": "Pin_P",
    "Pinzón": "Pnz_PZ",
    "Rábida": "Rab_RA",
    "San Cristóbal": "Scris_SR",
    "Santa Cruz": "Scruz_SC",
    "Santa Fe": "SanFe_SF",
    "Santiago": "Snt_ST",
    "Wolf": "Wol_W",
    "Bartolomé (Santiago)": "Snt_BT",
    "Bainbridge (Santiago)": "Snt_BA",
    "Beagles (Santiago)": "Snt_BE",
    "Daphnes (Santa Cruz)": "Scruz_DA",
    "Plazas (Santa Cruz)": "Scruz_PL",
    "Seymour (Santa Cruz)": "Scruz_SE",
}

def image_plausibility_check(image: Path):
    """

    :param image:
    :return:
    """
    # TODO check the date of the image
    # TODO check certain metadata like ISO, exposure time, aperture


def rename_single_image(island, mission_folder, image: Path):
    # test if the current image fits the schema

    image_name_stem = image.stem
    image_suffix = image.suffix

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


if __name__ == "__main__":

    base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y")
    new_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/01_cleaned_photos_all")
    new_path.mkdir(exist_ok=True, parents=True)
    # encode each Mission as a GeoreferencedImage
    ## it should be as fast as possible

    # save the GeoreferencedImages in a database
    ##
    df_data_changed = rename_incorrect_folders(base_path, new_path)
    df_data_changed.to_csv(new_path / "folder_rename.csv")
    move_folders(df_data_changed)

    ### Now do the same for image names
    images_list = list(file for file in base_path.glob("*/*/*.JPG") if not file.name.startswith("._") )
    image_names = [i.stem for i in images_list]
    image_name_splits = [len(i.split("_")) for i in image_names]
    df_image_data = pd.DataFrame({"Image": image_names,
                                  "image_path": [i.parent.stem for i in images_list],
                                  "island": [x.parent.parent.stem for x in images_list],
                                  "Split": image_name_splits})


    images_to_rename = []
    for p in new_path.glob("*/*") :
        if p.name.startswith("._"):
            continue
        # p = Path("/Volumes/G-DRIVE/Iguanas_From_Above/01_cleaned_photos_all/Floreana/FLMO05_03022021")
        images_list = [file for file in p.glob("*" ) if not file.name.startswith("._")]
        island, mission_folder = p.parent.stem, p.stem

        df_rename = rename_images(island, mission_folder, images_list)
        print(f"Path: {p}")
        print(f"Would rename the following images: {[n.stem for n in images_list]}")
        print(f"New names {df_rename}")
        df_changed_images = df_rename[df_rename["new_name"] != df_rename["old_name"]]

        images_to_rename.append(df_changed_images)

    df_changed_images = pd.concat(images_to_rename)
    df_changed_images.to_csv(new_path / "images_rename.csv")

    for _, row in df_changed_images.iterrows():
        full_old_image_path = new_path / row.island / row.mission_folder / row['old_name']
        full_new_image_path = new_path / row.island / row.mission_folder / row['new_name']
        logger.info(f"Moving {full_old_image_path} to {full_new_image_path}")
        # Create destination directory if it does not exist.
        shutil.move(full_old_image_path, full_new_image_path)





