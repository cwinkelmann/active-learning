from datetime import datetime

from shapely.geometry.multipoint import MultiPoint
from shapely.geometry import Point, LineString

from active_learning.types.Mission import MissionV2


class MissionStatistics:
    def __init__(self, mission: MissionV2, projected_CRS="EPSG:32715"):
        self.mission = mission
        self.projected_CRS = projected_CRS

        self.calculate_images_statistics()

    def calculate_images_statistics(self):
        """
        distance, other notable information on the dataset

        :param image_list: list
        :return:
        """

        self.avg_height = self.mission.get_geodata_frame(projected_CRS=self.projected_CRS)["height"].mean()
        self.image_gdf_projected = self.mission.get_geodata_frame(projected_CRS=self.projected_CRS).to_crs(self.projected_CRS)

        self.flight_path = LineString(self.image_gdf_projected["geometry"])
        self.distance = round(self.flight_path.length, 2)

        self.multi_pnt1 = MultiPoint(self.image_gdf_projected["geometry"])
        self.convex_hull = self.multi_pnt1.convex_hull
        self.center = (self.convex_hull.centroid.y, self.convex_hull.centroid.x)
        self.area = self.convex_hull.area


        first_photo_time = self.mission.get_geodata_frame(projected_CRS=self.projected_CRS)["datetime_original"].min()
        last_photo_time = self.mission.get_geodata_frame(projected_CRS=self.projected_CRS)["datetime_original"].max()

        self.first_photo_time = datetime.strptime(first_photo_time, '%Y:%m:%d %H:%M:%S')
        self.last_photo_time = datetime.strptime(last_photo_time, '%Y:%m:%d %H:%M:%S')

        self.photo_series_length = (self.last_photo_time - self.first_photo_time).total_seconds()