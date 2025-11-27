import math
import statistics

import numpy as np
import pandas as pd
import shapely
from loguru import logger
import geopandas as gpd

from helper.geometry import get_angle, point_from_angle, shift_to_vector_center, get_length, \
    get_angle_between_two_vectors_360


class FlightPath(object):

    def __init__(self, max_height, max_width, projected_CRS = None):
        self.max_height = max_height
        self.max_width = max_width
        self.projected_CRS = projected_CRS

    def _remove_duplicates(self, lst):
        """
        if a duplicate coordindate is in the list, remove it
        :param lst:
        :return:
        """

        res = []
        [res.append(x) for x in lst if x not in res]

        return res

    def generate_linear_flight_path(self,
                                    stride_x: int, stride_y: int,
                                    slice_size=1280,
                                    start_x=0, start_y=0,
                                    end_x=None, end_y=None
                                    ):
        """
        generate polygons along a linear flight path across the virtual map
        stop when the cutout leaves the image
        :return:
        """
        assert stride_x != 0 or stride_y != 0
        if end_y is None:
            end_y = self.max_height - slice_size
        if end_x is None:
            end_x = self.max_width - slice_size

        if stride_y == 0:
            slice_boxes_x = [x for x in range(start_x, end_x, stride_x)]
            slice_boxes_y = [0 for x in slice_boxes_x]
        # if stride_y > 0:
        #     slice_boxes_y = [y for y in range(start_y, self.max_height - slice_size, stride_y)]
        else:  # opposite direction
            slice_boxes_y = [y for y in range(start_y, end_y, stride_y)]

        if stride_x == 0:
            slice_boxes_x = [0 for y in slice_boxes_y]
        elif stride_x > 0:
            slice_boxes_x = [x for x in range(start_x, end_x, stride_x)]
        else:  # opposite direction
            slice_boxes_x = [x for x in range(start_x, end_x, stride_x)]

        x = slice_boxes_x
        y = slice_boxes_y
        xy = list(zip(slice_boxes_x, slice_boxes_y))

        self.xy = xy
        return xy



    def generate_u_flight_path(self, stride_x, stride_y,
                               turn_distance, turn_shift, start_x=700, start_y=100,
                               slice_size=1280):
        """

        :param stride_x:
        :param stride_y:
        :param turn_distance:
        :param turn_shift:
        :return:
        """
        forward = self.generate_linear_flight_path(stride_x=stride_x, stride_y=stride_y,
                                                   start_x=start_x, start_y=start_y)

        backward = self.generate_linear_flight_path(stride_x=stride_x * -1,
                                                    stride_y=stride_y * -1,
                                                    start_x=forward[-1][0] + int(slice_size / 2),
                                                    start_y=forward[-1][1],
                                                    end_x=0,
                                                    end_y=0
                                                    )
        self.xy = forward + backward
        return forward + backward

    def generate_sine_flight_path(self, stride_x, stride_y, turn_distance, turn_shift):
        """
        sine wave shaped flight path
        :param stride_x:
        :param stride_y:
        :param turn_distance:
        :param turn_shift:
        :return:
        """
        pass

    def generate_circle_flight_path(self, center, radius, a, cycles_degrees=360):
        """
        # TODO get the docsting right
        :type center: object
        :return:
        """
        raise Exception("implement me!")

    def fly_by_coordinates(self, POIs: list, interval=100) -> list[shapely.Point]:
        """
        fly from coordinate to coordinate
        POIs: Points of interest
        :return:
        """
        flight_path = []
        for a, b in [(POIs[k], POIs[k+1]) for k in range(len(POIs)-1)]:
            flight_path.append( self.corridor_flight(a, b, interval=interval) )
        # TODO with len(POIs) = 2 and either a_x = b_x or a_y = b_y this is equal to self.generate_linear_flight_path(
        flight_path = [item for row in flight_path for item in row]
        flight_path = self._remove_duplicates(flight_path)
        self.xy = flight_path
        return flight_path


    def corridor_flight(self, a, b, interval, norm_vector = (1, 0)):
        """
        fly a straight line from a to b, but save a x,y tuple at every distance=interval and b in the end
        :param a: tuple of (x, y)
        :param b: tuple of (x, y)
        :param interval:
        :param norm_vector: tuple of (x, y) which is the base vector
        :return:


        (x_a,y_a) -> (x_1, y_1) -> ... (x_b, y_b)
        """

        ## TODO how do I calculate that?
        """
        WE have a 2D space, a,b create a triangle
        i.e. 0,0 and 100,100 with 0,0 at the top left and 0 degrees is horizontal to the right means 135 degrees
        """

        flight_path = [shapely.Point(a)]


        b_norm = shift_to_vector_center(np.array(a), np.array(b))
        angle = get_angle(norm_vector, b_norm)
        angle = get_angle_between_two_vectors_360(norm_vector, b_norm)

        while get_length(flight_path[0], flight_path[-1]) < get_length(a, b):
            next_point = point_from_angle(flight_path[-1], angle, radius=interval)

            ## FIXME don't append it if it surpassed the
            if get_length(a, b) > get_length(a, next_point):
                flight_path.append( next_point )
            else:
                flight_path.append(shapely.Point(b))

        flight_path = self._remove_duplicates(flight_path)
        self.xy = flight_path

        return flight_path

    def stats(self):
        """ calculate some stats on the flight path """

        sgm_lth = statistics.mean([shapely.LineString(
            [self.xy[idx], self.xy[idx+1]]).length for idx in range(0, len(self.xy)-1)]
        )

        stats = {
            "length": shapely.LineString(self.xy),
            "avg_segment_length": sgm_lth
        }

        return stats

    def get_image_gdf(self, df: pd.DataFrame):
        """
        build a geodataframe from the dataframe of images
        :return:
        """
        # df = pd.DataFrame([x.image_metadata for x in self.image_list if x.image_metadata is not None] )

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

        return image_gdf

    def get_points(self):
        return self.xy

    def get_df(self):
        return pd.DataFrame( [{"x": p.x, "y": p.y} for p in self.xy] )


if __name__ == "__main__":

    fP = FlightPath(max_height=1900, max_width=1300)

    for i in range(1, 100, 10):
        i = 1 * i
        x = - 1
        print(f"(x,y){x},{i}: {round( np.degrees(np.arctan2((i), (x))), 3 )}")


