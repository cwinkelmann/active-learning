
import json
from pathlib import Path

from com.biospheredata.types.SimpleImage import SimpleImage


def get_image_size(img_path):
    """
    Image size in MB
    :param img_path:
    :return:
    """
    import os

    file_stats = os.stat(img_path)

    return file_stats.st_size / (1024 * 1024)


class GeoreferencedImage(SimpleImage):
    """
    retrieve metadata from a georeferenced mavic 2 pro image
    """

    image_path: str = None
    image_metadata: {} = None
    image_name: str = None
    image_hash: str = None

    def __init__(self, image_path: Path, image_metadata=None):
        super().__init__(image_path, image_metadata)

        self.image_name = Path(image_path).name

        if image_metadata is None:
            self.calculate_image_metadata(image_path)

    def __str__(self):
        super.__str__(self)
        return self.to_json()

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "image_metadata": self.image_metadata,
            "image_path": str(self.image_path),
            "image_name": str(self.image_name),
            "image_hash": str(self.image_hash),
        }


    @staticmethod
    def from_dict(self, dict_representation):
        """
        generate a Georeferenced Image from a persisted JSON document

        @param self:
        @param image_metadata:
        @param image_path:
        @return:
        """
        gi = GeoreferencedImage(image_path=dict_representation["image_path"],
                                  image_metadata=dict_representation["image_metadata"])
        gi.set_image_hash(dict_representation["image_hash"])

        return gi

    def decimal_coords(self, coords, ref):
        """
        calculate the decimal degrees
        :param coords:
        :param ref:
        :return:
        """
        decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
        if ref == "S" or ref == "W":
            decimal_degrees = -decimal_degrees
        return decimal_degrees

    # def calculate_image_metadata(self, img_path: Path):
    #     """
    #     load gps coordinates from the image
    #
    #     https://medium.com/spatial-data-science/how-to-extract-gps-coordinates-from-images-in-python-e66e542af354
    #
    #     :param img_path:
    #     :return:
    #     """
    #     with open(img_path, 'rb') as src:
    #         # exif.Image, NOT PIL.Image
    #         img = Image(src)
    #
    #
    #     # This is a bit of a performance bottleneck because the image will be opened and read a second time.
    #     self.image_hash = get_image_id(img_path)
    #     self.image_size = get_image_size(img_path)
    #
    #     # exif.Image
    #     if img.has_exif:
    #         try:
    #             logger.warning(f"There are multiple version of this, try finding the right one")
    #             latitude = self.decimal_coords(img.gps_latitude, img.gps_latitude_ref)
    #             longitude = self.decimal_coords(img.gps_longitude, img.gps_longitude_ref)
    #
    #             height = img.gps_altitude
    #
    #             # print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
    #             # print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")
    #             metadata = {"height": height, "latitude": latitude, "longitude": longitude,
    #                         "datetime_original": img.datetime_original, "filepath": str(img_path)}
    #
    #             supposedly_available_keys = ['image_width', 'image_height', 'bits_per_sample', 'image_description', 'make', 'model', 'orientation',
    #              'samples_per_pixel', 'x_resolution', 'y_resolution', 'resolution_unit', 'software', 'datetime',
    #              'y_and_c_positioning', '_exif_ifd_pointer', '_gps_ifd_pointer', 'xp_keywords', 'compression',
    #              'jpeg_interchange_format', 'jpeg_interchange_format_length', 'exposure_time', 'f_number',
    #              'exposure_program', 'photographic_sensitivity', 'exif_version', 'datetime_original',
    #              'datetime_digitized', 'exposure_bias_value', 'max_aperture_value', 'metering_mode', 'light_source',
    #              'focal_length', 'color_space', 'pixel_x_dimension', 'pixel_y_dimension', 'exposure_mode',
    #              'white_balance', 'digital_zoom_ratio', 'focal_length_in_35mm_film', 'scene_capture_type',
    #              'gain_control', 'contrast', 'saturation', 'sharpness', 'body_serial_number', 'lens_specification',
    #              'gps_version_id', 'gps_latitude_ref', 'gps_latitude', 'gps_longitude_ref', 'gps_longitude',
    #              'gps_altitude_ref', 'gps_altitude']
    #
    #             # raw_exif = img.get_all()
    #             for key in supposedly_available_keys:
    #
    #                 metadata[key] = img.get(key)
    #
    #             metadata["datetime_digitized"] = str(datetime.strptime(metadata["datetime_digitized"], "%Y:%m:%d %H:%M:%S"))
    #
    #         except AttributeError:
    #             print('No Coordinates')
    #             self.image_metadata = {}
    #             return {}
    #
    #     else:
    #         logger.warning(f"The Image {src} has no EXIF information")
    #
    #     dict_xmp_metadata = xmp_metadata(img_path)
    #
    #     metadata = {**metadata, **dict_xmp_metadata}
    #     self.image_metadata = metadata
    #     return metadata
    #
    # def set_image_hash(self, image_hash):
    #     if image_hash is None or image_hash == 'None':
    #         raise ValueError("The image hash may not be None")
    #     else:
    #         self.image_hash = image_hash
