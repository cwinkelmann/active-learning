import typing

from datetime import datetime

import hashlib

import PIL
import pandas as pd
from exif import Image
from loguru import logger
from pathlib import Path

from pydantic import BaseModel, Field
from enum import Enum
from typing import Tuple, Optional

from util.util import get_image_id


class ColorSpace(Enum):
    SRGB = 1
    UNCALIBRATED = 65535


class ExposureMode(Enum):
    AUTO_EXPOSURE = 0
    MANUAL_EXPOSURE = 1
    AUTO_BRACKET = 2


class ExposureProgram(Enum):
    NOT_DEFINED = 0
    MANUAL = 1
    NORMAL_PROGRAM = 2
    APERTURE_PRIORITY = 3
    SHUTTER_PRIORITY = 4
    CREATIVE_PROGRAM = 5  # For depth of field
    ACTION_PROGRAM = 6    # For motion capture
    PORTRAIT_MODE = 7     # For close-up photos with the background out of focus
    LANDSCAPE_MODE = 8    # For landscape photos with the background in focus


class GpsAltitudeRef(Enum):
    ABOVE_SEA_LEVEL = 0
    BELOW_SEA_LEVEL = 1


class LightSource(Enum):
    UNKNOWN = 0
    DAYLIGHT = 1
    FLUORESCENT = 2
    TUNGSTEN = 3
    FLASH = 4
    FINE_WEATHER = 9
    CLOUDY_WEATHER = 10
    SHADE = 11
    DAYLIGHT_FLUORESCENT = 12
    DAY_WHITE_FLUORESCENT = 13
    COOL_WHITE_FLUORESCENT = 14
    WHITE_FLUORESCENT = 15
    STANDARD_LIGHT_A = 17
    STANDARD_LIGHT_B = 18
    STANDARD_LIGHT_C = 19
    D55 = 20
    D65 = 21
    D75 = 22
    D50 = 23
    ISO_STUDIO_TUNGSTEN = 24
    OTHER = 255


class MeteringMode(Enum):
    UNKNOWN = 0
    AVERAGE = 1
    CENTER_WEIGHTED_AVERAGE = 2
    SPOT = 3
    MULTISPOT = 4
    PATTERN = 5
    PARTIAL = 6
    OTHER = 255


class Orientation(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4
    LEFT_TOP = 5
    RIGHT_TOP = 6
    RIGHT_BOTTOM = 7
    LEFT_BOTTOM = 8


class ResolutionUnit(Enum):
    NONE = 1
    INCHES = 2
    CENTIMETERS = 3


class Saturation(Enum):
    NORMAL = 0
    LOW = 1
    HIGH = 2


class SceneCaptureType(Enum):
    STANDARD = 0
    LANDSCAPE = 1
    PORTRAIT = 2
    NIGHT_SCENE = 3


class Sharpness(Enum):
    NORMAL = 0
    SOFT = 1
    HARD = 2


class WhiteBalance(Enum):
    AUTO = 0
    MANUAL = 1

# Define the Pydantic model
class ExifData(BaseModel):
    _exif_ifd_pointer: int
    _gps_ifd_pointer: int
    bits_per_sample: int
    body_serial_number: str
    color_space: ColorSpace
    compression: int
    contrast: int
    datetime: str
    datetime_digitized: str
    datetime_original: str
    digital_zoom_ratio: float
    exif_version: str
    exposure_bias_value: float
    exposure_mode: ExposureMode
    exposure_program: ExposureProgram
    exposure_time: float
    f_number: float
    filepath: str
    focal_length: float
    focal_length_in_35mm_film: int
    gain_control: int
    gps_altitude: float
    gps_altitude_ref: GpsAltitudeRef
    gps_latitude: Tuple[float, float, float]
    gps_latitude_ref: str
    gps_longitude: Tuple[float, float, float]
    gps_longitude_ref: str
    gps_version_id: int
    height: float
    image_description: str
    image_height: int
    image_width: int
    jpeg_interchange_format: int
    jpeg_interchange_format_length: int
    latitude: float
    lens_specification: Tuple[float, float, float, float]
    light_source: LightSource
    longitude: float
    make: str
    max_aperture_value: float
    metering_mode: MeteringMode
    model: str
    orientation: Orientation
    photographic_sensitivity: int
    pixel_x_dimension: int
    pixel_y_dimension: int
    resolution_unit: ResolutionUnit
    samples_per_pixel: int
    saturation: Saturation
    scene_capture_type: SceneCaptureType
    sharpness: Sharpness
    software: str
    white_balance: WhiteBalance
    x_resolution: float
    xp_keywords: str
    y_and_c_positioning: int
    y_resolution: float

class ExtendImageMetaData(ExifData):
    image_id: typing.Union[str, int] = Field(alias='image_id')
    image_name: str = Field(alias='image_name', description="Name of the image file")


class XMPMetaData(BaseModel):
    format: Optional[str] = Field(None, alias="format")
    drone_dji_GpsLatitude: Optional[float] = Field(None, alias="drone-dji:GpsLatitude")
    drone_dji_GpsLongitude: Optional[float] = Field(None, alias="drone-dji:GpsLongitude")
    drone_dji_GimbalYawDegree: Optional[float] = Field(None, alias="drone-dji:GimbalYawDegree")
    drone_dji_GimbalRollDegree: Optional[float] = Field(None, alias="drone-dji:GimbalRollDegree")
    drone_dji_GimbalPitchDegree: Optional[float] = Field(None, alias="drone-dji:GimbalPitchDegree")
    drone_dji_AbsoluteAltitude: Optional[float] = Field(None, alias="drone-dji:AbsoluteAltitude")
    drone_dji_RelativeAltitude: Optional[float] = Field(None, alias="drone-dji:RelativeAltitude")
    drone_dji_FlightRollDegree: Optional[float] = Field(None, alias="drone-dji:FlightRollDegree")
    drone_dji_FlightYawDegree: Optional[float] = Field(None, alias="drone-dji:FlightYawDegree")
    drone_dji_FlightPitchDegree: Optional[float] = Field(None, alias="drone-dji:FlightPitchDegree")

    class Config:
        # Allow population by field alias. This means that if you pass a dict with keys
        # like "drone-dji:GpsLatitude", they will be properly converted.
        allow_population_by_field_name = True
        # Optionally, you can use:
        # use_enum_values = True
        # if you use Enums in some cases.


def list_images(path: Path, extension):
    """
    find images in a path

    :param path:
    :return:
    """

    images_list = list(path.glob(f"*.{extension}"))
    images_list = [image_path for image_path in images_list if not str(image_path).startswith(".")]

    return images_list

def get_metadata_dataframe(image_list: list[Path]):
    """
    get the image metadata as a dataframe

    :param image_list:
    :return:
    """
    image_metadata = []

    for image_path in image_list:
        image_metadata.append(get_exif_metadata(image_path))

    df_image_metadata = pd.DataFrame(image_metadata)
    df_image_metadata["image_name"] = [Path(filepath).name for filepath in df_image_metadata["filepath"]]
    return df_image_metadata

def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def xmp_metadata(img_path) -> XMPMetaData:
    metadata = {}

    try:
        from libxmp import XMPFiles, consts
        xmpfile = XMPFiles(file_path=str(img_path), open_forupdate=True)
        xmp = xmpfile.get_xmp()
        metadata["format"] = xmp.get_property(consts.XMP_NS_DC, 'format')


        for xmp_key in [
            "drone-dji:GpsLatitude",
            "drone-dji:GpsLongitude",
            "drone-dji:GimbalYawDegree",
            "drone-dji:GimbalRollDegree",
            "drone-dji:GimbalPitchDegree",  # the pitch is the inclination with -90 == NADIR and 0 is horizontal
            "drone-dji:AbsoluteAltitude",
            "drone-dji:RelativeAltitude",
            "drone-dji:FlightRollDegree",
            "drone-dji:FlightYawDegree",
            "drone-dji:FlightPitchDegree"
        ]:
            try:
                metadata[xmp_key] = float(xmp.get_property("http://www.dji.com/drone-dji/1.0/", xmp_key))
                metadata_model = XMPMetaData(**metadata)
            except Exception as e:
                ## with phantom 4, someone wrote the metadata tag wrong: 'drone-dji:GpsLongitude' instead of 'drone-dji:GPSLongitude'
                logger.error(f"Problem with {xmp_key}, {e}")
        return metadata_model

    except Exception as e:
        logger.error(
            f"Problems with XMP library. Check https://python-xmp-toolkit.readthedocs.io/en/latest/installation.html and propably https://stackoverflow.com/questions/68869984/error-installing-exempi-2-5-2-on-m1-macbook-pro-running-big-sur")
        logger.error(f"Modify exempi.py f path is None: \
                        m1_path = '/opt/homebrew/lib/libexempi.dylib' \
                        if os.path.exists(m1_path): \
                            path = m1_path" )
        logger.error(e)



def get_exif_metadata(img_path) -> ExifData:
    """
    extract the image exif and xmp metadata from a dji drone image
    :param img_path:
    :return:
    """
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
            exif_metadata = {"height": height, "latitude": latitude, "longitude": longitude,
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
                exif_metadata[key] = img.get(key)
            # try:
            #     del raw_exif["flash"]
            #     del raw_exif["xp_comment"]
            # except KeyError as e:
            #     logger.error(f"key not there, {e}")

            # metadata = {**metadata, **raw_exif}

            exif_metadata["datetime_digitized"] = str(datetime.strptime(exif_metadata["datetime_digitized"], "%Y:%m:%d %H:%M:%S"))

        except AttributeError:
            print('No Coordinates')
            return {}

    else:
        logger.warning(f"The Image {src} has no EXIF information")
        raise Exception(f"No Exif Data Available of image name {src}")

    exif_instance = ExifData(**exif_metadata)

    return exif_instance


def get_image_size(img_path):
    import os

    file_stats = os.stat(img_path)

    return file_stats.st_size / (1024 * 1024)