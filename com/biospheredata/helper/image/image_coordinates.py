"""
Code to load an image and project pixel coordinates on the coordinates

"""
from pathlib import Path

from loguru import logger
from osgeo import gdal
import geopandas


def local_coordinates_to_wgs84(image_name, sliced_image_path: Path, annotations, detection_model_category_names):
    """

    :param sliced_image_path:
    :param image_name:
    :param annotations:
    :return:
    """
    georeferenced_tiff = str(sliced_image_path.joinpath(image_name))
    logger.info(f"trying to open this file: {georeferenced_tiff}")
    if not Path.is_file(Path(georeferenced_tiff)):
        raise ValueError(f"File {georeferenced_tiff} is not there.")
    src = gdal.Open(georeferenced_tiff)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    # ulx, uly is the upper left corner, lrx, lry is the lower right corner
    #
    #The osr library (part of gdal) can be used to transform the points to any coordinate system. For a single point:
    print(lry)
    src.GetProjection()

    from shapely.geometry import Point, LineString, Polygon
    from shapely.geometry import Point

    d = {}

    d = {'category_id': [],
        'category_name': [],
         'geometry': [
         ]}
    for annotation in annotations:
        d['category_id'].append(annotation['category_id'])
        d['category_name'].append(detection_model_category_names[annotation['category_id']])

        x = ( annotation["bbox"][0] + annotation["bbox"][2]/2) * xres
        y = ( annotation["bbox"][1] + annotation["bbox"][3]/2) * yres
        d['geometry'].append(Point(ulx+x, uly+y))
        logger.info(f"logged iguana position")

    gdf = geopandas.GeoDataFrame(d)
    gdf = gdf.set_crs(src.GetProjection(), allow_override=True)

    print(f"got this crs: {gdf.crs}")
    #gdf.set_crs(4326, allow_override=True)
    gdf_proj = gdf.to_crs({'init': 'epsg:4326'})

    try:
        gdf_path = str(sliced_image_path.joinpath(f"{image_name}_.geojson"))
        gdf_proj.to_file(gdf_path, driver="GeoJSON")

        return gdf_path

    except Exception as e:
        logger.error(f"cannot write geodata frame: {gdf_path}")
