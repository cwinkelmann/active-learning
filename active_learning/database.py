"""
Create a database of images
"""

import typing

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


def build_image_database(p = Path("/Users/christian/data/2TB/ai-core/data/expedition")):
    """
    Create a database of images

    in the simplest case I just want to store the image path and the metadata
    The mission itself should know its date and location beyond the image_name


    :return:
    """

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

# def main():
#     """
#     run main function
#     :return:
#     """
#     base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y")
#     new_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/01_cleaned_photos_all")
#     new_path.mkdir(exist_ok=True, parents=True)
#     # encode each Mission as a GeoreferencedImage
#     ## it should be as fast as possible
#
#     # save the GeoreferencedImages in a database
#     ##
#     df_data_changed = rename_incorrect_folders(base_path, new_path)
#     df_data_changed.to_csv(new_path / "incorrect_dates.csv")
#     move_folders(df_data_changed)
#
#     ### Now do the same for image names
#     images_list = list(base_path.glob("*/*/*.JPG"))
#     image_names = [i.stem for i in images_list]
#     image_name_splits = [len(i.split("_")) for i in image_names]
#     df_image_data = pd.DataFrame({"Image": image_names,
#                                   "image_path": [i.parent.stem for i in images_list],
#                                   "island": [x.parent.parent.stem for x in images_list],
#                                   "Split": image_name_splits})
#
#
#     df_image_data
#
#     return df_data, df_image_data
# if __name__ == "__main__":
#     df_data, df_image_data = main()
#
#     print(df_data, df_image_data)