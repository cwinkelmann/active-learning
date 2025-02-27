from pathlib import Path
import geopandas as gpd
import typing

import shapely
from shapely import affinity

from com.biospheredata.types.HastyAnnotationV2 import ImageLabel

from shapely import Polygon, affinity, box
from loguru import logger

def add_offset_centroid(point: shapely.Point,
                        image_height: int,
                        image_width: int,
                        # image_cutout_box: shapely.Polygon,
                        offset: int = 50, angle = 0) -> shapely.Polygon:
    """
    add an offset to the point. If the point is too close to the image boarder, the offset is adjusted so the box has the same size
    :param box:
    :param img_height:
    :param img_width:
    :return:
    """
    # Step 2: Calculate the minx, miny, maxx, maxy for the bounding box
    minx = max(point.x - offset, 0)
    maxx = min(point.x + offset, image_width)
    miny = max(point.y - offset, 0)
    maxy = min(point.y + offset, image_height)

    # Ensure the box remains of size offset*2 wide and offset*2 high
    # Adjust if it goes out of bounds
    if maxx - minx < offset * 2:
        if minx == 0:
            maxx = minx + offset * 2
        elif maxx == image_width:
            minx = maxx - offset * 2

    if maxy - miny < offset * 2:
        if miny == 0:
            maxy = miny + offset * 2
        elif maxy == image_height:
            miny = maxy - offset * 2

    # Step 3: Create the bounding box using the box function
    bounding_box = box(minx, miny, maxx, maxy)

    return bounding_box


def add_offset_to_box(box: typing.List[int], img_height: int, img_width: int, offset: int = 50):
    """
    add an offset to the cutout
    :param box:
    :param img_height:
    :param img_width:
    :return:
    """

    x1, y1, x2, y2 = box
    x1 -= offset
    x2 += offset
    y1 -= offset
    y2 += offset

    return (max(0, x1), max(0, y1), min(x2, img_width), min(y2, img_height))


def project_point_to_crop(point: shapely.Point, crop_box: shapely.Polygon) -> shapely.Point:
    """
    Projects a point from full-image coordinates into a cropped image's coordinate system.

    The crop_box is assumed to be a Shapely polygon (e.g. created via shapely.box)
    with bounds (minx, miny, maxx, maxy). The new coordinate system will have
    (minx, miny) of the crop_box as the new (0,0).

    Parameters:
        point (shapely.geometry.Point): Point in the full image.
        crop_box (shapely.geometry.Polygon): The crop box defining the new boundary.

    Returns:
        shapely.geometry.Point: The point in the crop's coordinate system.
    """
    # Get the lower-left corner coordinates of the crop box.
    minx, miny, maxx, maxy = crop_box.bounds

    # Translate the point by subtracting the crop's minimum coordinates.
    projected_point = affinity.translate(point, xoff=-minx, yoff=-miny)
    return projected_point

def project_label_to_crop(label: ImageLabel, crop_box: shapely.Polygon) -> ImageLabel:
    """
    similar to project_point_to_crop but for ImageLabel, offset all coordinates to the new origin
    :param label:
    :param crop_box:
    :return:
    """
    raise NotImplementedError("Not yet implemented")

def reframe_bounding_box(cutout_box: typing.Union[shapely.Polygon, shapely.box],
                         label: typing.Union[shapely.Polygon, ImageLabel, shapely.Point],
                         angle = 0, fit_to_box=True) -> typing.Union[shapely.Polygon, ImageLabel]:
    """ Reframe the bounding box to the new cutout box.
    The cutout box is the new image boundary, so its x_min, y_min is the new origin, therefore all coordinates need
    to be translated to this new origin.

    :param cutout_box:
    :param annotation:
    :param image_boundary:
    :param fit_to_box: if True the bounding box is fitted to the cutout box,
    :return:
    """
    if isinstance(label, ImageLabel):
        bbox_polygon = label.bbox_polygon # TODO extend this for marks

    elif isinstance(label, shapely.Polygon):
        # TODO implement this
        bbox_polygon = label

    else:
        raise ValueError("label must be either ImageLabel or shapely.Polygon")
    assert isinstance(bbox_polygon, shapely.Polygon), "bbox_polygon must be a shapely.Polygon"

    if angle != 0:
        bbox_polygon = shapely.affinity.rotate(bbox_polygon, -1 * angle, origin=cutout_box.centroid, use_radians=False)


    if bbox_polygon.intersects(cutout_box):
        # change coordinates to the label by using the coordinates of the cutout box
        # translate the cutout box to the origin
        bbox_polygon_t = affinity.translate(bbox_polygon, -cutout_box.bounds[0], -cutout_box.bounds[1])
        # print(bbox_polygon_t)
        # ensure the bounding boxes are within the cutout box
        xmin, ymin, xmax, ymax = bbox_polygon_t.bounds

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, cutout_box.bounds[2] - cutout_box.bounds[0])
        ymax = min(ymax, cutout_box.bounds[3] - cutout_box.bounds[1])

        if xmin < 0 or ymin < 0 or xmax > cutout_box.bounds[2] or ymax > cutout_box.bounds[3]:
            raise ValueError("bbox_polygon is not within the cutout box")

        if isinstance(label, ImageLabel):
            label.bbox = xmin, ymin, xmax, ymax
            return label
        else:
            return shapely.box(xmin, ymin, xmax, ymax)
    else:
        logger.warning("bbox_polygon is not within the cutout box")


def reframe_polygon(cutout_box: shapely.Polygon,
                    label: typing.Union[shapely.Polygon, ImageLabel],
                    fit_to_box: bool = True) -> typing.Union[shapely.Polygon, ImageLabel]:
    """ Reframe the bounding box to the new cutout box.
    The cutout box is the new image boundary, so its x_min, y_min is the new origin, therefore all coordinates need
    to be translated to this new origin.
    TODO: this could replace reframe_bounding_box because of the intersection this should work for all polygons
    # TODO refactor this so it can handly segmentation polygons
    :param cutout_box:
    :param annotation:
    :param image_boundary:
    :return:
    """

    if isinstance(label, ImageLabel):
        bbox_polygon = label.bbox_polygon
    else:

        assert isinstance(label, shapely.Polygon)
        bbox_polygon = label

    if bbox_polygon.intersects(cutout_box):
        if fit_to_box:
            bbox_polygon = bbox_polygon.intersection(cutout_box)
            assert bbox_polygon.within(cutout_box), "The polygon must be within the cutout box now"
        else:
            assert bbox_polygon.intersects(cutout_box), "The polygon must be within the cutout box now"

        bbox_polygon_t = affinity.translate(bbox_polygon, -cutout_box.bounds[0], -cutout_box.bounds[1])


        if isinstance(label, ImageLabel):
            label.bbox = bbox_polygon_t.bounds
            return label
        else:
            return bbox_polygon_t

    else:
        logger.warning("bbox_polygon is outside the cutout box. Because of the rotation it sometimes happens")

def convert_shapefile2usable(shapefile_path: Path):
    parent_path = shapefile_path.parent
    # Convert to GeoJSON
    gdf = gpd.read_file(shapefile_path)
    geojson_path = parent_path / "output.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    # Convert to CSV (geometry as WKT format)
    csv_path = parent_path / "output.csv"
    gdf.to_csv(csv_path, index=False)

    print(f"GeoJSON saved to: {geojson_path}")
    print(f"CSV saved to: {csv_path}")

    return gdf

    # if polygon.within(cutout_box):
    #     # change coordinates to the label by using the coordinates of the cutout box
    #     # translate the cutout box to the origin
    #
    #     bbox_polygon_t = affinity.translate(polygon, -cutout_box.bounds[0], -cutout_box.bounds[1])
    #     return bbox_polygon_t
    #
    # elif polygon.intersects(cutout_box):
    #
    #     polygon = polygon.intersection(cutout_box)
    #
    #     bbox_polygon_t = affinity.translate(polygon, -cutout_box.bounds[0], -cutout_box.bounds[1])
    #
    #     return bbox_polygon_t
