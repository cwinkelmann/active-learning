"""
@author: Christian Winkelmann
"""
import json
from datetime import datetime
import datetime as dtt
from geopandas import GeoDataFrame
import geopandas as gpd
from loguru import logger
from pathlib import Path

import pandas as pd
from shapely.geometry import MultiPoint

from com.biospheredata.types.GeoreferencedImage import GeoreferencedImage


class GeoreferencedImageList(object):
    """
    List of Georeferenced Images
    """
    photo_series_length = float
    metadata_fp = "georeferenced_imagelist_metadata.json"

    def __init__(self, projected_CRS):
        """

        """

        self.image_gdf: GeoDataFrame = None
        self.center = None
        self.image_gdf: GeoDataFrame
        n = 0
        self.image_list = []
        self.projected_CRS = projected_CRS

    def __str__(self):
        super.__str__(self)
        return self.to_json()

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "image_list": [x.to_dict() for x in self.image_list],
        }

    def __dict___(self):
        return self.to_dict()

    def __repr__(self):
        return self.to_json()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.image_list):
            self.n += 1
            return self.image_list[self.n]
        else:
            raise StopIteration

    def len(self):
        return len(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def append(self, item: GeoreferencedImage):
        """
        @deprecated
        append an image to the list
        @param item:
        @return:
        """
        self.add_image(item)

    def add_image(self, gi: GeoreferencedImage):
        """

        @param gi:
        @return:
        """
        self.image_list.append(gi)
        # logger.warning(f"add_image is quite slow over time with longer lists")
        # ## TODO this ia bit inefficient for huge lists
        # df = pd.DataFrame([x.image_metadata for x in self.image_list])
        # image_gdf = gpd.GeoDataFrame(
        #     df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        #     # the coordinates in the images should always be WGS 84
        # )
        # image_gdf = image_gdf.to_crs(crs=self.projected_CRS)
        #
        # self.image_gdf = image_gdf

    def get_image_gdf(self):
        """
        TODO depprecated this moved to flight image sim...
        :return:
        """
        ## on read cache, which gets invalidated when the list has a different length

        logger.warning("This is deprecated and moved to flight image sim")

        if (self.image_gdf is None or len(self.image_gdf) != len(self.image_list)) and len(self.image_list) > 0:
            df = pd.DataFrame([x.image_metadata for x in self.image_list if x.image_metadata is not None] )

            ## remote outliers
            df_zero_zero = df[(df["latitude"] == 0.0) & (df["longitude"] == 0.0)]
            if len(df_zero_zero) > 0:
                logger.error(f"There are elements in the image list, which have a latitude of 0 and longitude of 0. Will remote those")
                df = df[(df["latitude"] != 0.0) & (df["longitude"] != 0.0)]

            try:
                format = "%Y-%m-%d %H:%M:%S"
                df["datetime_digitized"] = pd.to_datetime(df["datetime_digitized"],
                                                                       format=format)
            except ValueError:
                format = "%Y:%m:%d %H:%M:%S"
                logger.warning(f"Dateformat {format} was wrong, try again with ")
                df["datetime_digitized"] = pd.to_datetime(df["datetime_digitized"],
                                                                       format=format)

                df["timedelta_deviation_from_median"] = abs(df["datetime_digitized"] - df["datetime_digitized"].median())
                df["timedelta_deviation_from_median"] = df["timedelta_deviation_from_median"].apply(lambda x: x.total_seconds())

                if len(df[df["timedelta_deviation_from_median"] > 60 * 17]) > 0:
                    logger.error(f"It seems there was an image in the set which does not belong there. It was shot long before or after the flight.")
                    logger.error(df[df["timedelta_deviation_from_median"] > 60 * 17])
                    df = df[df["timedelta_deviation_from_median"] < 60 * 17]

            image_gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
                # the coordinates in the images should always be WGS 84
            )
            image_gdf = image_gdf.to_crs(crs=self.projected_CRS)

            self.image_gdf = image_gdf

        return self.image_gdf

    def calculate_images_statistics(self):
        """
        distance, other notable information on the dataset
        TODO this moved to flight image sim...
        :param image_list: list
        :return:
        """
        from shapely.geometry import Point, LineString

        logger.warning("This is deprecated and moved to flight image sim")

        self.avg_height = self.get_image_gdf()["height"].mean()
        self.image_gdf_projected = self.get_image_gdf().to_crs(self.projected_CRS)

        self.flight_path = flight_path = LineString(self.image_gdf_projected["geometry"])

        self.multi_pnt1 = multi_pnt1 = MultiPoint(self.image_gdf_projected["geometry"])
        self.convex_hull = multi_pnt1.convex_hull
        self.center = (self.convex_hull.centroid.y, self.convex_hull.centroid.x)
        self.area = self.convex_hull.area

        self.distance = round(flight_path.length, 2)
        first_photo_time = self.get_image_gdf()["datetime_original"].min()
        last_photo_time = self.get_image_gdf()["datetime_original"].max()

        self.first_photo_time = datetime.strptime(first_photo_time, '%Y:%m:%d %H:%M:%S')
        self.last_photo_time = datetime.strptime(last_photo_time, '%Y:%m:%d %H:%M:%S')

        self.photo_series_length = (self.last_photo_time - self.first_photo_time).total_seconds()



    @staticmethod
    def from_dict(dict_rep, projected_CRS):

        gil = GeoreferencedImageList(projected_CRS=projected_CRS)
        for i in dict_rep["image_list"]:
            gi = GeoreferencedImage(image_path=i["image_path"], image_metadata=i["image_metadata"])
            gi.set_image_hash(i["image_hash"])
            gil.add_image(gi)

        return gil

    def add_image_list(self, image_list):
        """
        TODO implement me
        :param image_list:
        :return:
        """
        pass


    def persist(self, path: Path):
        with open(path.joinpath(self.metadata_fp), 'w') as f:

            json.dump(self.__dict___(), f)

    def to_dataframe(self, columns):
        metadata_gdf = self.get_image_gdf()[columns].copy(deep=True)
        metadata_gdf["image_hashes"] = pd.Series([gi.image_hash for gi in self.image_list])
        metadata_gdf["image_name"] = pd.Series([gi.image_name for gi in self.image_list])

        return metadata_gdf