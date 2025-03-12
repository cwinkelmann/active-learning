"""
Iterate through the folders in the raw_photos_all_y directory and rename the folders that have the wrong format.
The original format is not changed
"""

import shutil

import pandas as pd
from pathlib import Path

import typing

from loguru import logger

from active_learning.util.rename import rename_images, rename_incorrect_folders, move_folders, fix_date_format, \
    run_renaming, rename_single_image

base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/2024_El Nino project')
# new_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/01_cleaned_photos_Floreana")


# new_path.mkdir(exist_ok=True, parents=True)
# encode each Mission as a GeoreferencedImage
## it should be as fast as possible

# save the GeoreferencedImages in a database
##
# df_data_changed = rename_incorrect_folders(base_path, new_path)
# df_data_changed.to_csv(new_path / "folder_rename.csv")
# move_folders(df_data_changed)
#
# ### Now do the same for image names
# images_list = list(file for file in base_path.glob("*/*/*.JPG") if not file.name.startswith("._") )
# image_names = [i.stem for i in images_list]
# image_name_splits = [len(i.split("_")) for i in image_names]
# df_image_data = pd.DataFrame({"Image": image_names,
#                               "image_path": [i.parent.stem for i in images_list],
#                               "island": [x.parent.parent.stem for x in images_list],
#                               "Split": image_name_splits})

import re

renaming_dict = []

if __name__ == "__main__":

    images_to_rename = []
    pattern = re.compile(r"DJI_\d{4}\.JPG", re.IGNORECASE)
    matching_files = [file for file in base_path.rglob("*.JPG") if pattern.match(file.name)]

    for p in matching_files:

        island, mission_folder = p.parent.parent.stem, p.parent.stem

        new_image_name = rename_single_image(island, mission_folder, p)

        renaming_dict.append({"island": island, "mission_folder": mission_folder, "old_name": p.name, "new_name": new_image_name})

        p.rename(base_path / island / mission_folder / new_image_name)

    df_rename = pd.DataFrame(renaming_dict)
    # df_rename.to_csv(new_path / "images_rename.csv")

    # run_renaming(df_rename, new_path)