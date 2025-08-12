## uncomment this if you want to copy the prefixed images to a folder in ordner to remove the prefixes
import shutil
from pathlib import Path

unedited_images = Path("/Users/christian/data/zooniverse/images/Zooniverse_Goldstandard_images/3rd launch")
editet_images = Path("/Users/christian/data/zooniverse/images/Zooniverse_Goldstandard_images/3rd launch_without_prefix")

# shutil.copytree(unedited_images, editet_images)

from com.biospheredata.helper.zooniverse.zooniverse_analysis import rename_2023_scheme_images_to_zooniverse, \
    rename_from_schema

df_renamed = rename_2023_scheme_images_to_zooniverse(editet_images)

rename_from_schema(df_renamed)