"""
example to extract image metadata from dji drone images
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from exif import Image
from loguru import logger


# to read xmp exempi has to be installed too brew install exempi
# https://libopenraw.freedesktop.org/exempi/
# https://python-xmp-toolkit.readthedocs.io/en/latest/installation.html
## on an arm mac you need to fix this:
# https://stackoverflow.com/questions/68869984/error-installing-exempi-2-5-2-on-m1-macbook-pro-running-big-sur


def list_images(path: Path, extension) -> list[Path]:
    """
    find images in a path and return list of paths

    :param path:
    :return:
    """
    logger.warning(f"This is deprecated, use the function frm the flight_image_capturing_sim repo")

    list_images(path, extension)

    if not isinstance(path, Path):
        path = Path(path)

    images_list = list(path.glob(f"*.{extension}"))
    images_list = [image_path for image_path in images_list if not str(image_path).startswith(".")]

    return images_list

def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def xmp_metadata(img_path):
    metadata = {}

    try:
        from libxmp import XMPFiles, consts
        xmpfile = XMPFiles(file_path=str(img_path), open_forupdate=True)
        xmp = xmpfile.get_xmp()
        metadata["format"] = xmp.get_property(consts.XMP_NS_DC, 'format')

        for xmp_key in [
            "drone-dji:GpsLatitude", "drone-dji:GpsLongitude",
            "drone-dji:GimbalYawDegree", "drone-dji:GimbalRollDegree",
            "drone-dji:GimbalPitchDegree",  # the pitch is the inclination with -90 == NADIR and 0 is horizontal
            "drone-dji:AbsoluteAltitude", "drone-dji:RelativeAltitude",
            "drone-dji:FlightRollDegree", "drone-dji:FlightYawDegree", "drone-dji:FlightPitchDegree"
        ]:
            metadata[xmp_key] = float(xmp.get_property("http://www.dji.com/drone-dji/1.0/", xmp_key))

    except Exception as e:
        logger.error(
            f"Problems with XMP library. Check https://python-xmp-toolkit.readthedocs.io/en/latest/installation.html")

    return metadata

def image_metadata_yaw_tilt(img_path):

    with open(img_path, 'rb') as src:
        # exif.Image, NOT PIL.Image
        img = Image(src)

    ## work on the EXIF data
    if img.has_exif:
        try:
            latitude = decimal_coords(img.gps_latitude, img.gps_latitude_ref)
            longitude = decimal_coords(img.gps_longitude, img.gps_longitude_ref)

            height = img.gps_altitude

            # print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
            # print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")
            metadata = {"height": height, "latitude": latitude, "longitude": longitude,
                        "datetime_original": img.datetime_original, "filepath": str(img_path)}

            supposedly_available_keys = ['image_width', 'image_height', 'bits_per_sample', 'image_description', 'make',
                                         'model', 'orientation',
                                         'samples_per_pixel', 'x_resolution', 'y_resolution', 'resolution_unit', 'software',
                                         'datetime',
                                         'y_and_c_positioning', '_exif_ifd_pointer', '_gps_ifd_pointer', 'xp_keywords',
                                         'compression',
                                         'jpeg_interchange_format', 'jpeg_interchange_format_length', 'exposure_time',
                                         'f_number',
                                         'exposure_program', 'photographic_sensitivity', 'exif_version',
                                         'datetime_original',
                                         'datetime_digitized', 'exposure_bias_value', 'max_aperture_value', 'metering_mode',
                                         'light_source',
                                         'focal_length', 'color_space', 'pixel_x_dimension', 'pixel_y_dimension',
                                         'exposure_mode',
                                         'white_balance', 'digital_zoom_ratio', 'focal_length_in_35mm_film',
                                         'scene_capture_type',
                                         'gain_control', 'contrast', 'saturation', 'sharpness', 'body_serial_number',
                                         'lens_specification',
                                         'gps_version_id', 'gps_latitude_ref', 'gps_latitude', 'gps_longitude_ref',
                                         'gps_longitude',
                                         'gps_altitude_ref', 'gps_altitude']

            # raw_exif = img.get_all()
            for key in supposedly_available_keys:
                metadata[key] = img.get(key)
            # try:
            #     del raw_exif["flash"]
            #     del raw_exif["xp_comment"]
            # except KeyError as e:
            #     logger.error(f"key not there, {e}")

            # metadata = {**metadata, **raw_exif}

            metadata["datetime_digitized"] = str(datetime.strptime(metadata["datetime_digitized"], "%Y:%m:%d %H:%M:%S"))



        except AttributeError:
            print('No Coordinates')
            image_metadata = {}
            return {}


    else:
        logger.warning(f"The Image {src} has no EXIF information")
        raise Exception("No Exif Data Available")

    dict_xmp_metadata = xmp_metadata(img_path)

    metadata = {**metadata, **dict_xmp_metadata}

    return metadata

if __name__ == "__main__":

    images = list_images(Path("/Users/christian/data/missions/Jan 2023/Floreana_small/FLBB01_28012023"), extension="JPG")
    # image_path = Path("/Users/christian/data/missions/Jan 2023/Floreana_small/FLBB01_28012023/DJI_0448.JPG")
    image_metadata = []

    for image_path in images:
        image_metadata.append(image_metadata_yaw_tilt(image_path))

    df_image_metadata = pd.DataFrame(image_metadata)
    print(df_image_metadata)

    print(image_metadata)