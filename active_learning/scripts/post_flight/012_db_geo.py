"""
This script is used to update the image metadata database with the new image names and paths.
"""



from pathlib import Path

import pandas as pd

from active_learning.util.rename import rename_single_image
from helper.image import get_image_gdf
import re

df_db_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.csv")
df_db = pd.read_csv(df_db_path)

for index, row in df_db.iterrows():
    pattern = re.compile(r"DJI_\d{4}\.JPG", re.IGNORECASE)
    image_name = row["image_name"]
    if not pattern.match(image_name):
        continue

    island = row["island"]
    mission_folder = row["mission_folder"]
    filepath = Path(row["filepath"])
    new_image_name = rename_single_image(island, mission_folder, filepath)
    if full_path_old.exists():
        # inplace rename

        renamed = full_path_old.parent / new_folder_path.name
        full_path_old.rename(renamed)


gdf_image_metadata_2 = get_image_gdf(df_db)
gdf_image_metadata_2.to_file(Path("/Volumes/G-DRIVE/Iguanas_From_Above") / Path(df_db_path).with_suffix(".geojson"),
                             driver="GeoJSON")
gdf_image_metadata_2.to_file(Path("/Volumes/G-DRIVE/Iguanas_From_Above") / Path(df_db_path).with_suffix(".geojson"),
                             driver="GeoJSON")

gdf_image_metadata_2.to_file(Path("/Volumes/G-DRIVE/Iguanas_From_Above") / Path(df_db_path).with_suffix(".shp"), driver="ESRI Shapefile")