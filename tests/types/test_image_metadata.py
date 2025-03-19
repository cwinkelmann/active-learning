import pytest
from exif import Orientation, ResolutionUnit, Saturation, SceneCaptureType, Sharpness, WhiteBalance, MeteringMode, \
    LightSource, GpsAltitudeRef, ExposureProgram, ExposureMode, ColorSpace

from active_learning.types.image_metadata import ExifData, XMPMetaData

@pytest.fixture
def sample_data():
    sample_data = {
        "_exif_ifd_pointer": 332,
        "_gps_ifd_pointer": 746,
        "bits_per_sample": 866,
        "body_serial_number": "0K8TG920120534",
        "color_space": ColorSpace.SRGB,
        "compression": 7,
        "contrast": 0,
        "datetime": "2021:02:03 09:31:43",
        "datetime_digitized": "2021-02-03 09:31:43",
        "datetime_original": "2021:02:03 09:31:43",
        "digital_zoom_ratio": 1.0,
        "exif_version": "0230",
        "exposure_bias_value": -0.7,
        "exposure_mode": ExposureMode.AUTO_EXPOSURE,
        "exposure_program": ExposureProgram.NORMAL_PROGRAM,
        "exposure_time": 0.003125,
        "f_number": 5.0,
        "filepath": "/Users/christian/data/2TB/ai-core/data/fake_test_data/Floreana/FLPC02_22012023/Flo_FLPC02_DJI_0833_22012023.JPG",
        "focal_length": 10.26,
        "focal_length_in_35mm_film": 28,
        "gain_control": 0,
        "gps_altitude": 35.0,
        "gps_altitude_ref": GpsAltitudeRef.ABOVE_SEA_LEVEL,
        "gps_latitude": (1.0, 19.0, 16.3759),
        "gps_latitude_ref": "S",
        "gps_longitude": (90.0, 30.0, 32.21),
        "gps_longitude_ref": "W",
        "gps_version_id": 2,
        "height": 35.0,
        "image_description": "default",
        "image_height": 3648,
        "image_width": 5472,
        "jpeg_interchange_format": 21130,
        "jpeg_interchange_format_length": 30394,
        "latitude": -1.3212155277777777,
        "lens_specification": (28.0, 28.0, 2.8, 11.0),
        "light_source": LightSource.DAYLIGHT,
        "longitude": -90.50894722222222,
        "make": "Hasselblad",
        "max_aperture_value": 2.971,
        "metering_mode": MeteringMode.CENTER_WEIGHTED_AVERAGE,
        "model": "L1D-20c",
        "orientation": Orientation.TOP_LEFT,
        "photographic_sensitivity": 100,
        "pixel_x_dimension": 5472,
        "pixel_y_dimension": 3648,
        "resolution_unit": ResolutionUnit.INCHES,
        "samples_per_pixel": 3,
        "saturation": Saturation.NORMAL,
        "scene_capture_type": SceneCaptureType.STANDARD,
        "sharpness": Sharpness.NORMAL,
        "software": "10.00.12.07",
        "white_balance": WhiteBalance.MANUAL,
        "x_resolution": 72.0,
        "xp_keywords": "singl",
        "y_and_c_positioning": 1,
        "y_resolution": 72.0
    }
    return sample_data

def test_exif_image_metadata(sample_data):


    exif_instance = ExifData(**sample_data)
    json_str = exif_instance.model_dump_json()

    photo_loaded = ExifData.model_validate_json(json_str)

    assert isinstance(photo_loaded, ExifData)
    assert photo_loaded.compression == 7

def test_exif_meta_with_image():
    """
    Load an image and extract the exif metadata
    :return:
    """
    raise NotImplementedError

    assert isinstance(photo_loaded, ExtendImageMetaData)
    assert photo_loaded.compression == 7

def test_xmp_metadata_with_image():

    raise NotImplementedError
    assert isinstance(photo_loaded, XMPMetaData)
    assert photo_loaded.compression == 7


