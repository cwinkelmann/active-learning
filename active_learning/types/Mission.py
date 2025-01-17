from __future__ import annotations


import pathlib

import typing
import uuid

import pandas as pd
from loguru import logger

from pathlib import Path
import geopandas as gpd
from active_learning.types.GeoreferencedImage import GeoreferencedImage


class NoImagesException(Exception):
    pass

class MissionV2(object):
    """
    new better version of the Mission
    """

    def __init__(self, mission_name: str,
                 base_path: Path,
                 CRS):
        """
        Full Path to full renamed images
        :param mission_name:
        :param base_path:
        :param CRS:
        :param creation_date:
        """
        self.uuid = str(uuid.uuid1())
        self.base_path = base_path
        self.raw_image_files = []
        self.CRS = CRS

        self.georeferenced_images: typing.List[GeoreferencedImage] = []  # then n-th element refers to the n-th element of the raw_image
        self.mission_name = mission_name

    @staticmethod
    def init(base_path: Path, suffix, CRS) -> MissionV2:
        """
        open a whole mission from disk
        @return: Mission
        """
        assert type(base_path) is Path or pathlib.PosixPath, "base_path must be a Path Object"
        base_path = Path(base_path)
        mission_name = Path(base_path).parts[-1]

        mission = MissionV2(mission_name, base_path, CRS=CRS)

        mission.add_raw_images(suffix=suffix)
        return mission

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
                self.georeferenced_images.append(GeoreferencedImage.from_path(path / n))

        if len(new_images) == 0:
            raise NoImagesException(f"no images found in {path} with suffix {suffix}")
        self.raw_image_files = self.raw_image_files + new_images


    def get_images(self, absolute=True):
        """

        @param absolute:
        @return:
        """
        if absolute:
            return [self.base_path.joinpath(x) for x in self.raw_image_files]
        else:
            return self.raw_image_files

    def get_geodata_frame(self, projected_CRS="EPSG:32715") -> gpd.GeoDataFrame:
        """
        get a GeoDataFrame from the images
        @return:
        """
        if len(self.georeferenced_images) == 0:
            raise NoImagesException("no images loaded")

        df_images = pd.DataFrame([x.exif_metadata.model_dump() for x in self.georeferenced_images])
        df_images = df_images[(df_images["latitude"] != 0.0) & (df_images["longitude"] != 0.0)]

        df_images.sort_values("datetime_original")

        df_images["uuid"] = self.uuid
        df_images["mission_name"] = self.mission_name

        image_gdf = gpd.GeoDataFrame(
            df_images, geometry=gpd.points_from_xy(df_images.longitude, df_images.latitude), crs="EPSG:4326"
            # the coordinates in the images should always be WGS 84
        )
        gdf_images = image_gdf.to_crs(crs=projected_CRS)

        return gdf_images