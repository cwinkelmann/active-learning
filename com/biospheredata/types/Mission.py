from __future__ import annotations
import json
import pandas as pd
import uuid

from pathlib import Path
import pathlib

from loguru import logger
from com.biospheredata.types.GeoreferencedImage import GeoreferencedImage
from com.biospheredata.types.GeoreferencedImageList import GeoreferencedImageList
from com.biospheredata.types.HastyAnnotation import HastyAnnotation

class NoImagesException(Exception):
    pass




class Mission(object):
    """
    FIXME: this deprecated because way too complex. The type itself should be written by a PyDantic Class. Modifications should be done with a seperate class
    abstraction of a mission
    @deprecated
    """
    # Version of the format
    VERSION = 1.0

    raw_image_files = []
    creation_date = None
    orthophoto_path = None
    CRS = None
    metadata_fp = "mission_metadata.json"
    json_representation_path = None
    georeferenced_imagelist_metadata: Path

    def __init__(self, mission_name: str, base_path: Path, CRS, creation_date=None):

        self.uuid = str(uuid.uuid1())
        self.raw_image_files = []
        self.CRS = CRS

        self.georeferenced_image_metadata = GeoreferencedImageList(CRS)  # then n-th element refers to the n-th element of the raw_image
        self.set_creation_date(creation_date)

        self.mission_name = mission_name
        self.expedition_name = None
        self.base_path = base_path
        self.json_representation_path = base_path.joinpath(self.metadata_fp)
        self.orthophoto_path = None
        self.hasty_annotation_file = None
        self.hasty_annotation_file_mosaic = None
        self.georeferenced_imagelist_metadata = None

    def set_base_path(self, base_path):
        self.base_path = Path(base_path)

    @staticmethod
    def link_additional_configs(mission: Mission) -> Mission:
        """
        resolve other settings if it is suitable.
        :param mission:
        :return:
        """

        if Path(mission.base_path.joinpath("output/odm_orthophoto/odm_orthophoto.tif")).is_file():
            logger.info(f"found an orthophoto in odm_orthophoto")
            mission.orthophoto_path = Path("output/odm_orthophoto/odm_orthophoto.tif")
        if Path(mission.base_path.joinpath("output/odm_orthophoto/odm_orthophoto.tiff")).is_file():
            logger.info(f"found an orthophoto in odm_orthophoto")
            mission.orthophoto_path = Path("output/odm_orthophoto/odm_orthophoto.tiff")

        if Path(mission.base_path.joinpath("hasty_annotation.json")).is_file():
            mission.hasty_annotation_file = Path("hasty_annotation.json")

        if Path(mission.base_path.joinpath(GeoreferencedImageList.metadata_fp)).is_file():
            mission.georeferenced_imagelist_metadata = GeoreferencedImageList.metadata_fp

        return mission

    @staticmethod
    def init(base_path: Path, suffix, CRS) -> Mission:
        """
        open a whole mission from disk
        @return: Mission
        """
        assert type(base_path) is Path or pathlib.PosixPath, "base_path must be a Path Object"
        base_path = Path(base_path)
        mission_name = Path(base_path).parts[-1]
        mission_date = Path(base_path).parts[-2]
        expedition_name = Path(base_path).parts[-3]

        mission = Mission(mission_name, base_path, CRS=CRS)
        mission.creation_date = mission_date
        mission.set_expedition_name(expedition_name)

        mission.add_raw_images(suffix=suffix)
        mission.extract_meta_data()
        mission.calculate_flight_statistics()

        mission = Mission.link_additional_configs(mission)

        return mission

    @staticmethod
    def open(base_path) -> Mission:
        """
        open a whole mission from disk.

        @return:
        """
        assert type(base_path) is Path or pathlib.PosixPath, "base_path must be a Path Object"
        if not isinstance(base_path, Path):
            base_path = Path(base_path)

        if not base_path.exists():
            raise Exception(f"Path {base_path} does not exist")

        mission_name = Path(base_path).parts[-1]
        mission_date = Path(base_path).parts[-2]
        expedition_name = Path(base_path).parts[-3]

        mission = Mission(mission_name, base_path, CRS=None)
        mission.creation_date = mission_date
        mission.set_expedition_name(expedition_name)

        mission.load_meta_data()
        mission.calculate_flight_statistics()
        mission = Mission.link_additional_configs(mission)

        return mission

    @staticmethod
    def from_json_file(filepath: Path):
        """
        loat a persisted mission from disk
        :param filepath:
        :return:
        """
        with open(filepath, 'r') as f:
            job_dict = json.load(f)

        return Mission.from_dict(job_dict)

    @staticmethod
    def from_dict(job_dict):

        mission_uuid = job_dict["mission_uuid"]
        mission_name = job_dict["mission_name"]
        raw_image_files = job_dict["raw_image_files"]
        creation_date = job_dict["creation_date"]
        expedition_name = job_dict["expedition_name"]
        base_path = Path(job_dict["base_path"])

        georeferenced_image_metadata = job_dict["metadata_dictionaries"]
        CRS = job_dict["CRS"]

        mission = Mission(mission_name=mission_name, base_path=base_path, CRS=CRS)
        mission.uuid = mission_uuid
        mission.raw_image_files = raw_image_files
        mission.creation_date = creation_date
        mission.set_expedition_name(expedition_name)
        mission.georeferenced_image_metadata = GeoreferencedImageList.from_dict(georeferenced_image_metadata,
                                                                                projected_CRS=CRS)
        mission.orthophoto_path = job_dict["orthophoto_path"]

        hasty_annotation_file = job_dict.get("hasty_annotation_file", None)
        hasty_annotation_file = Path(hasty_annotation_file) if (
                    hasty_annotation_file != 'None' and hasty_annotation_file is not None) else None
        mission.set_hasty_annotation_file(hasty_annotation_file)

        hasty_annotation_file_mosaic = job_dict.get("hasty_annotation_file_mosaic", None)
        hasty_annotation_file_mosaic = Path(
            hasty_annotation_file_mosaic) if hasty_annotation_file_mosaic is not None else None
        mission.set_hasty_annotation_file_mosaic(hasty_annotation_file_mosaic)

        georeferenced_imagelist_metadata = job_dict.get("georeferenced_imagelist_metadata", None)
        georeferenced_imagelist_metadata = Path(
            georeferenced_imagelist_metadata) if georeferenced_imagelist_metadata is not None else None
        mission.set_georeferenced_imagelist_metadata(georeferenced_imagelist_metadata)

        return mission

    def __str__(self):
        super.__str__(self)
        return self.to_json()

    def to_json(self):
        return json.dumps(self.to_dict())

    def __dict___(self):
        return self.to_dict()

    def to_dataframe(self):
        """
        build a flat representation of the mission
        :return:
        """
        columns = [
            "height", "latitude", "longitude", "datetime_original", "filepath", "image_width", "image_height",

            "image_description",  # I left some columns here
            "orientation",
            "software",
            "datetime",
            'y_and_c_positioning',
            '_exif_ifd_pointer', 'xp_keywords',
            'compression', 'exposure_time', 'f_number',
            'exposure_program',
            'photographic_sensitivity', 'exif_version',
            'datetime_digitized',
            'exposure_bias_value',
            'max_aperture_value',
            'metering_mode',
            'light_source',
            'focal_length',
            'color_space',
            'pixel_x_dimension',
            'pixel_y_dimension',
            'exposure_mode',
            'white_balance',
            # 'gain_control', 'contrast', 'saturation', 'sharpness',
            'body_serial_number',
            'gps_version_id',
            'gps_altitude_ref',
            'gps_altitude',
            'geometry',

            ## the XMP keys

            'drone-dji:GpsLatitude',
            'drone-dji:GpsLongitude',
            'drone-dji:GimbalYawDegree',
            'drone-dji:GimbalRollDegree',
            'drone-dji:GimbalPitchDegree',
            'drone-dji:AbsoluteAltitude',
            'drone-dji:RelativeAltitude',
            'drone-dji:FlightRollDegree',
            'drone-dji:FlightYawDegree',
            'drone-dji:FlightPitchDegree'

        ]

        ## TODO image metadata together with mission name and date
        flat_representation = self.georeferenced_image_metadata.to_dataframe(columns)
        flat_representation["mission_date"] = self.creation_date
        flat_representation["mission_name"] = self.mission_name

        flat_representation_with_annotations = pd.concat(
            [flat_representation,
             self.get_annotations_per_image()],
            axis=1)

        """
        df_statistics = self.get_annotations_statistics()
        if df_statistics is not None:
            flat_representation = pd.concat([flat_representation, df_statistics], axis=1).dropna()
        """
        return flat_representation_with_annotations

    def get_annotations_per_image(self):
        """
        if there are hasty annotations this will be merged so you have an overview.
        :return:
        """
        if self.hasty_annotation_file is not None and self.hasty_annotation_file != "None" :
            hA = HastyAnnotation.from_file(self.get_hasty_annotation_file(absolute=True))
            df_statistics = hA.image_summary_to_dataframe()

            return df_statistics

    def get_annotations_statistics(self):
        """
        export based on the orthophotos, labeled slices and annotations a statistic how many of X where found on which orthophoto

        TODO: this should be generated in the missions.

        :return:
        """

        mission_base_path = []
        statistics_for_all = []
        # hasty_annotations = []
        #
        # annotation_dictionary = {
        #     "hasty_prediction_files": [x.get_hasty_annotation_file_mosaic() for x in self.get_missions()],
        #     "mission_paths": [x.base_path for x in self.get_missions()]
        # }
        #
        # ## TODO left join the hasty annotations.
        # ## TODO left join candidate proposals
        #
        # annotation_df = pd.DataFrame.from_records(annotation_dictionary)
        #
        # ## TODO if image names would be unique we could join them now
        df_statistics = None

        if self.hasty_annotation_file is not None:
            hA = HastyAnnotation.from_file(self.get_hasty_annotation_file(absolute=True))
            df_statistics = hA.get_statistic(global_count=False)

        df_statistics
        # statistics_df = pd.concat([annotation_df, statistics_df], axis=1)
        #
        # statistics_df = statistics_df.fillna(0)
        # # convert the statistic columns to integer
        # cols = [i for i in statistics_df.columns if i not in ["hasty_prediction_files", "mission_paths"]]
        # for col in cols:
        #     statistics_df[col] = statistics_df[col].astype(int)
        return df_statistics

    def __repr__(self):
        return self.to_json()

    def to_dict(self):
        return {
            "version": self.VERSION,
            "mission_uuid": self.uuid,
            "mission_name": self.mission_name,
            "raw_image_files": [str(x) for x in self.raw_image_files],
            "creation_date": self.creation_date,
            "expedition_name": self.expedition_name,
            "metadata_dictionaries": self.georeferenced_image_metadata.to_dict(),
            "base_path": str(self.base_path),
            "orthophoto_path": str(self.orthophoto_path),
            "hasty_annotation_file": str(self.hasty_annotation_file),
            "hasty_annotation_file_mosaic": str(self.hasty_annotation_file_mosaic),
            "georeferenced_imagelist_metadata": str(self.georeferenced_imagelist_metadata),
            "CRS": str(self.CRS)
        }

    def calculate_flight_statistics(self):
        """
        calculate statistics of the set of images in the mission
        :return:
        """
        self.georeferenced_image_metadata.calculate_images_statistics()
        self.georeferenced_image_metadata.persist(self.base_path)

    def add_georeferenced_image(self, gi: GeoreferencedImage):
        """
        add a georeferenced image to the mission
        :param gi:
        :return:
        """
        self.georeferenced_image_metadata.add_image(gi)

    def add_raw_images(self, suffix="JPG"):
        """
        load all images in the path
        @param suffix:
        @return:
        """

        path = Path(self.base_path)
        new_images_tmp = [str(x.parts[-1]) for x in path.glob(f"*.{suffix}")]
        new_images = []
        for n in new_images_tmp:
            # These hidden files break the process. So they need to be removed.
            if not n.startswith("."):
                new_images.append(n)

        if len(new_images) == 0:
            raise NoImagesException(f"no images found in {path} with suffix {suffix}")
        self.raw_image_files = self.raw_image_files + new_images

    def set_raw_images(self, images_filepaths):

        self.raw_image_files = [str(Path(x).parts[-1]) for x in images_filepaths]

    def get_images(self, absolute=True):
        """

        @param absolute:
        @return:
        """
        if absolute:
            return [self.base_path.joinpath(x) for x in self.raw_image_files]
        else:
            return self.raw_image_files

    def extract_meta_data(self, refresh=False):
        """
        extract the medadata of all images of a mission
        """

        n = len(self.get_images(absolute=True))
        if n < 3:
            raise NoImagesException(f"not enough images found in {self.base_path}")
        k = 1
        if len(self.georeferenced_image_metadata) != len(self.raw_image_files):

            for image_path in self.get_images(absolute=True):
                logger.info(f"working through image number {k}/{n}, path: {image_path}")
                k = k + 1

                # get the image coordinates
                gi = GeoreferencedImage(image_path)
                gi.calculate_image_metadata(image_path)
                self.add_georeferenced_image(gi)

            return True
        else:
            return False

    def persist(self):
        """
        generate a dictionary representation and persist this as a json document
        @return:
        """
        self.json_representation_path = self.base_path.joinpath(self.metadata_fp)
        self.base_path.mkdir(exist_ok=True, parents=True)
        if len(self.raw_image_files) > 0:
            with open(self.json_representation_path, 'w+') as f:
                json.dump(self.to_dict(), f)

            return self.json_representation_path
        else:
            logger.warning(f"Not enough images")
            return None

    def set_orthophoto_path(self, orthophoto_path):
        self.orthophoto_path = orthophoto_path

    def get_geotiff_path(self, absolute=True):

        if absolute:
            try:
                return self.base_path.joinpath(self.orthophoto_path)
            except TypeError:
                return None
        else:
            return self.orthophoto_path

    def load_meta_data(self):
        """
        load the persisted json if it exists
        @return:
        """
        try:
            with open(self.base_path.joinpath(self.metadata_fp), 'r') as f:
                smm = json.load(f)

            self.CRS = smm.get("CRS")
            self.raw_image_files = smm.get("raw_image_files")
            self.uuid = smm.get("mission_uuid")
            self.georeferenced_image_metadata = GeoreferencedImageList.from_dict(smm.get("metadata_dictionaries", []),
                                                                                 projected_CRS=self.CRS)
            self.hasty_annotation_file = smm.get("hasty_annotation_file", None)
            self.hasty_annotation_file_mosaic = smm.get("hasty_annotation_file_mosaic", None)
            self.orthophoto_path = smm.get("orthophoto_path", None)

        except FileNotFoundError as e:
            logger.error(f"File: {self.metadata_fp} not found. You will need to generate first.")
            raise ValueError("Mission not initialised")

    def zip(self):
        """
        TODO implement me
        :return:
        """
        pass

    def set_hasty_annotation_file(self, annotation_file):
        self.hasty_annotation_file = annotation_file

    def get_center(self):
        """
        locate the center of the mission
        :return:
        """
        return self.georeferenced_image_metadata.center

    def get_hasty_annotation_file(self, absolute=True):
        if absolute:
            return self.base_path.joinpath(self.hasty_annotation_file)
        else:
            return self.hasty_annotation_file

    def set_hasty_annotation_file_mosaic(self, annotation_file):
        self.hasty_annotation_file_mosaic = annotation_file

    def get_hasty_annotation_file_mosaic(self, absolute=True):
        if absolute:
            return self.base_path.joinpath(self.hasty_annotation_file_mosaic)
        else:
            return self.hasty_annotation_file_mosaic

    def set_expedition_name(self, expedition_name):
        self.expedition_name = expedition_name

    def get_georeferenced_image_metadata(self) -> GeoreferencedImageList:
        return self.georeferenced_image_metadata

    def set_georeferenced_imagelist_metadata(self, georeferenced_imagelist_metadata):
        self.georeferenced_imagelist_metadata = georeferenced_imagelist_metadata

    def set_creation_date(self, creation_date):
        logger.warning("implement date checking. Often this is not a date.")
        self.creation_date = creation_date


if __name__ == '__main__':
    ## create a mission from a folder of just images
    mission_path = Path(
        "/media/christian/2TB/ai-core/data/iguanas_from_above/amy_big_disk/2021 Jan/Photos from drone/Floreana/02.02.21/FMO01")

    # Open a Mission from a Folder
    CRS = "EPSG:3395"
    mission_object = Mission.init(base_path=Path(mission_path), CRS=CRS, suffix="JPG")
