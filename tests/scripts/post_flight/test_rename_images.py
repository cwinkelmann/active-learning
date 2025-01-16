from pathlib import Path

import pytest

from active_learning.database import fix_date_format
from active_learning.util.rename import rename_images, rename_single_image


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

def test_rename_folder():
    """

    :return:
    """

