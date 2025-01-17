from pathlib import Path

import pytest

from active_learning.util.rename import rename_images, rename_single_image, raw_folder_to_mission, fix_date_format


def test_rename_images():
    images_list = [
        Path("Fernandina/FEC01_21012023/DJI_0001.JPG"),
        Path("Fernandina/FEC01_21012023/DJI_0002.JPG"),
        Path("Fernandina/FEC01_21012023/DJI_0003.JPG"),
        Path("Fernandina/FEC01_21012023/DJI_0004.JPG"),
    ]

    new_names = rename_images(island="Fernandina", mission_folder="FEC01_21012023", images_list=images_list)

    assert new_names == images_list


def test_rename_single_image():
    island = "Fernandina"
    mission_folder = "FEC01_21012023"
    image = Path("DJI_0001.JPG")

    new_image_name = rename_single_image(island, mission_folder, image)
    assert new_image_name == "Fer_FEC01_DJI_0001_21012023.JPG"

    image = Path("Fer_FEC01_DJI_0001_21012023_condor.JPG")
    new_image_name = rename_single_image(island, mission_folder, image)
    assert new_image_name == "Fer_FEC01_DJI_0001_21012023.JPG"


def test_fix_date_format():
    date_str = "120122"
    new_date_str = fix_date_format(date_str)

    assert new_date_str == "12012020"


def test_raw_folder_to_mission():
    """
    Test the function that converts a raw folder to a mission folder

    Assumes in Floreana/Flo_FLPC02_22012023 are images in the raw format DJI_0001.JPG, DJI_0002.JPG, DJI_0003.JPG, DJI_0004.JPG...

    :return:
    """
    # raw_folder = Path("Fernandina/FEC01_21012023")
    raw_folder = Path("/Users/christian/data/2TB/ai-core/data/fake_test_data/Floreana/Flo_FLPC02_22012023")

    df_rename = raw_folder_to_mission(raw_folder)

    assert sorted(list(df_rename.keys())) == ['full_path', 'island', 'mission_folder', 'new_name', 'old_name']

    df_rename