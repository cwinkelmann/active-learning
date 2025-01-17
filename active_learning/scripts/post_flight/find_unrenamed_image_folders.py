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
    run_renaming

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

run_renaming(df_changed_images, new_path)