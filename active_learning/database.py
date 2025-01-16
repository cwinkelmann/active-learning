"""
Create a database of images
"""
import shutil

import pandas as pd
import typing
from loguru import logger

from pathlib import Path

from com.biospheredata.image.image_metadata import list_images
from com.biospheredata.types.GeoreferencedImage import GeoreferencedImage

"""
There is the old version: GeoreferencedImage.calculate_image_metadata
testLoadImage

And there is image_metadata.image_metadata_yaw_tilt()
Then there is flight_image_capturing_sim
"""

def get_dates(path: Path) -> typing.List[Path]:
    pass

def get_missions(path: Path) -> typing.List[Path]:
    pass


def image_database():
    """
    Create a database of images

    in the simplest case I just want to store the image path and the metadata
    The mission itself should know its date and location beyond the image_name


    :return:
    """
    p = Path("/Users/christian/data/2TB/ai-core/data/expedition")

    georeferenced_images: typing.List[GeoreferencedImage] = []

    for d in get_dates(path=p):
        for m in get_missions(d):
            for i in list_images(m, extension="JPG"):
                mission_date = d.stem
                mission_name = m.stem
                mission_code = "TODO" # break the name apart
                gi = GeoreferencedImage(image_path=i)
                georeferenced_images.append(gi)




    # TODO get list of dates
    # TODO get list of Missions
    # TODO get list Images

    return {"date": d}

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

def rename_incorrect_folders(base_path, new_base_path):
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


def main():
    base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y")
    new_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/01_cleaned_photos_all")
    new_path.mkdir(exist_ok=True, parents=True)
    # encode each Mission as a GeoreferencedImage
    ## it should be as fast as possible

    # save the GeoreferencedImages in a database
    ##
    df_data_changed = rename_incorrect_folders(base_path, new_path)
    df_data_changed.to_csv(new_path / "incorrect_dates.csv")
    move_folders(df_data_changed)

    ### Now do the same for image names
    images_list = list(base_path.glob("*/*/*.JPG"))
    image_names = [i.stem for i in images_list]
    image_name_splits = [len(i.split("_")) for i in image_names]
    df_image_data = pd.DataFrame({"Image": image_names,
                                  "image_path": [i.parent.stem for i in images_list],
                                  "island": [x.parent.parent.stem for x in images_list],
                                  "Split": image_name_splits})


    df_image_data

    return df_data, df_image_data
if __name__ == "__main__":
    df_data, df_image_data = main()

    print(df_data, df_image_data)