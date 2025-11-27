import shutil

from pathlib import Path
import fiona
import geopandas as gpd

from com.biospheredata.types.GeoreferencedImageList import GeoreferencedImageList


class GeoreferencedImageListConverter(object):

    fiona.supported_drivers['KML'] = 'rw'
    columns = ["height", "latitude", "geometry", "datetime_digitized", "filepath"]

    @staticmethod
    def convert_points(gil: GeoreferencedImageList, path: Path, prefix: str = None):
        """
        take the georeferenced image list and persist it as a shapefile
        """
        path.mkdir(exist_ok=True, parents=True)

        # .to_file(path.joinpath(f"photo_points.geojson"), driver="GeoJSON")
        gil.get_image_gdf()[GeoreferencedImageListConverter.columns]\
            .to_file(str(Path(path).joinpath(f"./{prefix}photo_points.geojson")), driver="GeoJSON")

        ## TODO read this
        # with open(str(path.joinpath("photo_points.kml")), 'w') as fb:
        #
        #     fiona.supported_drivers['KML'] = 'rw'
        #     gil.image_gdf[GeoreferencedImageListConverter.columns]\
        #         .to_file(str(path.joinpath("photo_points.kml")), driver='KML')

        return True

    @staticmethod
    def get_convex_hull(gil: GeoreferencedImageList, path: Path, prefix: str = None):
        """
        save the convex hull of the photos

        @param prefix:
        @param gil:
        @param path:
        @return:
        """
        path.mkdir(exist_ok=True, parents=True)

        ## TODO read this
        fiona.supported_drivers['KML'] = 'rw'
        # gil.image_gdf.convex_hull.to_file(path.joinpath("convex_hull.kml"), driver='KML')
        gil.get_image_gdf().convex_hull.to_file(
            path.joinpath(f"./{prefix}convex_hull.geojson"), driver='GeoJSON')

