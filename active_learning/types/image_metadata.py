import PIL.Image as PIL_Image
import geopandas as gpd
import hashlib
import pandas as pd
import pydantic_core
import time
import tqdm
import typing
from datetime import datetime
from enum import Enum
from exif import Image as ExifImage
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field
from shapely.geometry import Point
from typing import Tuple, Optional
import hashlib

from active_learning.util.image import get_image_id


class ColorSpace(Enum):
    SRGB = 1
    UNCALIBRATED = 65535


class ExposureMode(Enum):
    AUTO_EXPOSURE = 0
    MANUAL_EXPOSURE = 1
    AUTO_BRACKET = 2


class ExposureProgram(Enum):
    """

    """
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
    GpsLatitude: float = Field(alias="GpsLatitude")
    GpsLongitude: float = Field( alias="GpsLongitude")
    GimbalYawDegree: float = Field(alias="GimbalYawDegree")
    GimbalRollDegree: float = Field(alias="GimbalRollDegree")
    GimbalPitchDegree: float = Field(alias="GimbalPitchDegree")
    AbsoluteAltitude: float = Field(alias="AbsoluteAltitude")
    RelativeAltitude: float = Field(alias="RelativeAltitude")
    FlightRollDegree: float = Field(alias="FlightRollDegree")
    FlightYawDegree: float = Field(alias="FlightYawDegree")
    FlightPitchDegree: float = Field(alias="FlightPitchDegree")

    class Config:
        populate_by_name = True

class ImageMetaData(BaseModel):
    datetime_digitized: str = Field(..., alias="datetime_digitized")
    folder_name: str = Field(..., alias="folder_name")
    image_hash: str = Field(..., alias="image_hash")
    image_name: str = Field(..., alias="image_name")
    island: str = Field(..., alias="island")
    mission_folder: str = Field(..., alias="mission_folder")

class CompositeMetaData(BaseModel):
    datetime_digitized: datetime = Field(..., alias="datetime_digitized")
    folder_name: str = Field(..., alias="folder_name")
    image_hash: str = Field(..., alias="image_hash")
    image_name: str = Field(..., alias="image_name")
    island: str = Field(..., alias="island")
    mission_folder: str = Field(..., alias="mission_folder")

    _exif_ifd_pointer: int
    _gps_ifd_pointer: int
    bits_per_sample: int
    body_serial_number: str
    color_space: ColorSpace
    compression: int
    contrast: int
    datetime: str
    datetime_digitized: datetime
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

    GpsLatitude: float = Field(alias="GpsLatitude")
    GpsLongitude: float = Field(alias="GpsLongitude")
    GimbalYawDegree: float = Field(alias="GimbalYawDegree")
    GimbalRollDegree: float = Field(alias="GimbalRollDegree")
    GimbalPitchDegree: float = Field(alias="GimbalPitchDegree")
    AbsoluteAltitude: float = Field(alias="AbsoluteAltitude")
    RelativeAltitude: float = Field(alias="RelativeAltitude")
    FlightRollDegree: float = Field(alias="FlightRollDegree")
    FlightYawDegree: float = Field(alias="FlightYawDegree")
    FlightPitchDegree: float = Field(alias="FlightPitchDegree")

    def to_series(self) -> pd.Series:
        data = self.model_dump()
        return pd.Series(data)

    def to_serialisable_series(self) -> pd.Series:
        data = self.model_dump()
        data = {k: (v.name if isinstance(v, Enum) else v) for k, v in data.items()}
        return pd.Series(data)

    @classmethod
    def from_serialisable_series(cls, series):
        data = series[1].to_dict()  # Extract row data
        for field, field_type in cls.__annotations__.items():
            try:
                issubclass(field_type, Enum)
            except TypeError:
                pass
            if typing.get_origin(field_type) is tuple:
                continue
            elif issubclass(field_type, Enum) and field in data:
                data[field] = field_type[data[field]]
        return cls(**data)

class DerivedImageMetaData(CompositeMetaData):
    # Fields specific to drone imagery
    flight_code: str = Field(..., description="Flight code identifier")
    island_code: str = Field(..., description="Island code identifier")
    site_code: str = Field(..., description="Site code identifier")
    drone_name: str = Field(..., description="Name of the drone")
    YYYYMMDD: str = Field(..., description="Date in YYYY-MM-DD format")

    # GSD (Ground Sample Distance) calculations
    ground_height_m: float = Field(..., description="Ground height in meters")
    ground_width_m: float = Field(..., description="Ground width in meters")
    gsd_abs_avg_cm: float = Field(..., description="Average absolute ground sample distance in cm")
    gsd_abs_height_cm: float = Field(..., description="Height absolute ground sample distance in cm")
    gsd_abs_width_cm: float = Field(..., description="Width absolute ground sample distance in cm")
    gsd_rel_avg_cm: float = Field(..., description="Average relative ground sample distance in cm")
    gsd_rel_height_cm: float = Field(..., description="Height relative ground sample distance in cm")
    gsd_rel_width_cm: float = Field(..., description="Width relative ground sample distance in cm")

    # Flight data fields that might be None/NaN
    bearing_to_prev: Optional[float] = Field(None, description="Bearing to previous image location")
    distance_to_prev: Optional[float] = Field(None, description="Distance to previous image location")
    flight_direction: Optional[str] = Field(None, description="Direction of flight")
    forward_overlap_pct: Optional[float] = Field(None, description="Forward overlap percentage")
    risk_score: Optional[float] = Field(None, description="Risk score")
    shift_mm: Optional[float] = Field(None, description="Shift in millimeters")
    shift_pixels: Optional[float] = Field(None, description="Shift in pixels")
    speed_m_per_s: Optional[float] = Field(None, description="Speed in meters per second")
    time_diff_seconds: Optional[float] = Field(None, description="Time difference in seconds")

    def to_serialisable_series(self) -> pd.Series:
        """
        Convert this model to a pandas Series for serialization.
        Override this method to customize the serialization process.
        """
        data_dict = self.model_dump()
        # Handle special types like Enums, Points, etc.
        for key, value in data_dict.items():
            if isinstance(value, Enum):
                data_dict[key] = value.name
            elif isinstance(value, Point):
                # Keep geometry as is for GeoDataFrame
                data_dict[key] = (value.x, value.y)

        return data_dict

    @classmethod
    def from_dataframe_row(cls, row):
        """
        Create an instance from a dataframe row, handling geometry creation
        from latitude and longitude if necessary.
        """
        row_dict = row[1].to_dict()
        # Remove geometry if present as we'll handle it separately
        if 'geometry' in row_dict:
            del row_dict['geometry']

        return cls(**row_dict)

def list_images(path: Path, extension, recursive=False):
    """
    find images in a path

    :param extension:
    :param recursive:
    :return:
    :param path:
    :return:
    """
    if recursive:
        images_list = list(path.rglob(f"*.{extension}"))
    else:
        images_list = list(path.glob(f"*.{extension}"))

    # remove hidden files which are especially annoying on a Mac
    images_list = [image_path for image_path in images_list if not str(image_path.name).startswith(".")]

    return images_list

def get_image_metadata(image_list: list[Path]) -> typing.List[CompositeMetaData]:
    """
    This is the most recent version of the function ( 2025-03-17 )
    get the image metadata as a dataframe

    :param image_list:
    :return:
    """
    image_metadata: typing.List[CompositeMetaData] = []

    # Initialize tqdm progress bar
    pbar = tqdm.tqdm(total=len(image_list), desc="Processing Images", unit="img")

    # Track time
    start_time = time.time()

    for i, image_path in enumerate(image_list):
        #try:
            # Log progress every 100 images
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            avg_time_per_image = elapsed / i
            remaining_time = avg_time_per_image * (len(image_list) - i)
            logger.info(
                f"Extracting metadata: {i}/{len(image_list)} ({image_path.stem}) - "
                f"Elapsed: {elapsed:.2f}s, Estimated remaining: {remaining_time:.2f}s"
            )

        # Process image metadata
        image_metadata.append(image_metadata_yaw_tilt(image_path))

        # except Exception as e:
        #     logger.error(f"Problem with {image_path}, {e}")

        # Update tqdm progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Final runtime
    total_time = time.time() - start_time
    logger.info(f"Processing complete in {total_time:.2f} seconds.")

    return image_metadata


def convert_to_serialisable_dataframe(
        image_metadata: typing.List[CompositeMetaData] | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert the image metadata to a GeoDataFrame, which can be serialised easily
    :param image_metadata:
    :return:
    """
    lst_image_metadata = []

    crs = "EPSG:4326"
    if isinstance(image_metadata, gpd.GeoDataFrame):
        # For GeoDataFrame, extract each row as a series
        crs = image_metadata.crs

        for idx, row in image_metadata.iterrows():
            data = {k: (v.name if isinstance(v, Enum) else v) for k, v in row.items()}
            lst_image_metadata.append(pd.Series(data))
    else:
        for item in image_metadata:
            lst_image_metadata.append(item.to_serialisable_series())

    # Convert to GeoDataFrame to preserve geometry
    gdf_image_metadata = gpd.GeoDataFrame(lst_image_metadata)

    gdf_image_metadata.set_crs(crs, inplace=True)

    # Ensure required columns exist
    cols_to_move = ['image_name', 'folder_name']
    existing_cols = [col for col in cols_to_move if col in gdf_image_metadata.columns]

    # Reorder the columns
    new_order = existing_cols + [col for col in gdf_image_metadata.columns if col not in existing_cols]
    gdf_image_metadata = gdf_image_metadata[new_order]

    return gdf_image_metadata

def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def xmp_metadata_PIL(img_path: Path):

    # Open image
    image = PIL_Image.open(img_path)

    # Extract XMP metadata
    xmp_data = image.getxmp()
    metadata = {}
    # Print XMP metadata
    if xmp_data:

        xmp_data = xmp_data["xmpmeta"]["RDF"]["Description"]

        for xmp_key in [
            "GpsLatitude",
            "GpsLongitude",
            "GimbalYawDegree",
            "GimbalRollDegree",
            "GimbalPitchDegree",  # the pitch is the inclination with -90 == NADIR and 0 is horizontal
            "AbsoluteAltitude",
            "RelativeAltitude",
            "FlightRollDegree",
            "FlightYawDegree",
            "FlightPitchDegree"
        ]:
            try:
                metadata[xmp_key] = float(xmp_data.get(xmp_key))
            except:
                metadata[xmp_key] = None
    else:
        print("No XMP metadata found")

    return metadata


def image_hash(img_path: Path):
    """
    hash an image
    TODO: check if this is the best way to hash an image and if it can be sped up because it just reads the image a second time
    :param img:
    :return:
    """
    logger.warning(f"Deprecated, use image_id instead")
    img = PIL_Image.open(img_path)
    md5hash = hashlib.md5(img.tobytes())
    return md5hash.hexdigest()


def image_metadata_yaw_tilt(img_path: Path) -> CompositeMetaData:
    """
    extract the image exif and xmp metadata from a dji drone image
    :param img_path:
    :return:
    """
    with open(img_path, 'rb') as src:
        # exif.Image, NOT PIL.Image
        img = ExifImage(src)


    ## work on the EXIF data
    if img.has_exif:
        try:
            latitude = decimal_coords(img.gps_latitude, img.gps_latitude_ref)
            longitude = decimal_coords(img.gps_longitude, img.gps_longitude_ref)

            height = img.gps_altitude

            # print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
            # print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")
            exif_metadata = {"height": height, "latitude":
                latitude, "longitude": longitude,
                        "datetime_original": img.datetime_original,
                             "filepath": str(img_path)}

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
                                        'exposure_bias_value', 'max_aperture_value', 'metering_mode',
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
            img.get("datetime_digitized")

            obj_exif_metadata = ExifData(**exif_metadata)



        except AttributeError as e:
            logger.error(f"No Coordinates, {e} with image {img_path}")
            image_metadata = {}
            return {}
        except pydantic_core._pydantic_core.ValidationError:
            logger.error("Image has no complete Exif data, with image {img_path}")

    else:
        logger.warning(f"The Image {src} has no EXIF information")
        raise Exception(f"No Exif Data Available of image name {src}")
    # TODO use the XMPMetaData

    dict_xmp_metadata = xmp_metadata_PIL(img_path)
    obj_xmp_metadata = XMPMetaData(**dict_xmp_metadata) # TODO check if this works

    metadata = {}
    metadata["image_hash"] = get_image_id(img_path)
    metadata["image_name"] = img_path.name
    metadata["mission_folder"] = img_path.parent.name
    metadata["island"] = img_path.parent.parent.name
    metadata["datetime_digitized"] = datetime.strptime(exif_metadata["datetime_original"], "%Y:%m:%d %H:%M:%S")
    metadata["folder_name"] = img_path.parent.name

    # obj_img_metadata = ImageMetaData(**metadata)
    metadata = {**metadata, **exif_metadata, **dict_xmp_metadata}
    cIMD = CompositeMetaData(**metadata)

    # TODO create an object from this

    return cIMD

def find_leaky_images(df_image_metadata):
    """
    find images which are the same
    :param df_image_metadata:
    :return:
    """
    duplicate_hashes = df_image_metadata[df_image_metadata['image_hash'].duplicated()]
    id_redundant_images = list(duplicate_hashes['image_hash'].unique())



    return id_redundant_images




def find_spatially_close_images(gdf: gpd.GeoDataFrame, radius=50):
    """
    find images which are too close to each other
    :param df_image_metadata:
    :return:
    """
    # TODO implement this
    gdf['buffer'] = gdf['geometry'].buffer(radius)

    # Initialize a flag to check if any points are within the radius of another

    # Iterate through each point and check for overlaps
    for idx, row in gdf.iterrows():
        # Get the buffer of the current row
        buffer = row['buffer']

        # Get the other rows
        others = gdf.drop(idx)

        # Check for overlaps
        overlaps = others['buffer'].overlaps(buffer)

        # If there are overlaps, print the indices of the overlapping points
        if overlaps.any():
            no_overlap = False
            print(f"Point {row.image_name} overlaps with points {others[overlaps].image_name.tolist()}")

def get_exif_metadata(img_path) -> ExifData:
    """
    extract the image exif and xmp metadata from a dji drone image
    :param img_path:
    :return:
    """
    with open(img_path, 'rb') as src:
        # exif.Image, NOT PIL.Image
        img = ExifImage(src)

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

# def get_image_id(filename):
#     """
#     generate an id from the image itself which can be used to find images which are exactly the same
#     @param filename:
#     @return:
#     """
#     logger.warning(f"DEPRECTEAD: Use get_image_id from image.py")
#     with open(filename, "rb") as f:
#         bytes = f.read()  # read entire file as bytes
#         readable_hash = hashlib.sha256(bytes).hexdigest()
#     return readable_hash


def get_mosaic_slice_name(mission_name, creation_date):
    return f"{mission_name}_{creation_date}_m"


def get_image_gdf(df: pd.DataFrame, projected_CRS: str = "EPSG:4326", original_CRS: str = "EPSG:4326"):
    """

    build a geodataframe from the dataframe of images
    :return:
    """
    ## remote outliers
    df_zero_zero = df[(df["latitude"] == 0.0) & (df["longitude"] == 0.0)]

    if len(df_zero_zero) > 0:
        logger.error(
            f"There are elements in the image list, which have a latitude of 0 and longitude of 0. Will remote those")
        df = df[(df["latitude"] != 0.0) & (df["longitude"] != 0.0)]

    try:
        format = "%Y-%m-%d %H:%M:%S"
        df["datetime_digitized"] = pd.to_datetime(df["datetime_digitized"],
                                                  format=format)
    except ValueError:
        format = "%Y:%m:%d %H:%M:%S"
        logger.warning(f"Dateformat %Y-%m-%d %H:%M:%S was wrong, try again with {format}")
        df["datetime_digitized"] = pd.to_datetime(df["datetime_digitized"],
                                                  format=format)

    image_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=original_CRS
        # the coordinates in the images should always be WGS 84
    )
    image_gdf = image_gdf.to_crs(crs=projected_CRS)

    return image_gdf