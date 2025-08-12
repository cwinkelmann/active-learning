"""
Grid operations for image annotation
"""

import numpy as np
from typing import List

import random
import shapely
from shapely import affinity, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from com.biospheredata.types.annotationbox import BboxXYXY
from typing import List, Tuple, Dict
from shapely.geometry import Polygon
import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict

def create_regular_raster_grid(max_x: int, max_y: int,
                               slice_height: int, slice_width: int,
                               overlap: int = 0) -> Tuple[List[Polygon], List[Dict]]:
    """
    Create a regular raster grid of bounding boxes starting from the top-left corner of the image.

    Parameters:
        max_x (int): Width of the image.
        max_y (int): Height of the image.
        slice_height (int): Height of each tile.
        slice_width (int): Width of each tile.
        overlap (int): Overlap between tiles in pixels.

    Returns:
        Tuple[List[Polygon], List[Dict]]:
            - A list of shapely Polygons representing the grid tiles.
            - A list of dictionaries with tile metadata (indices and coordinates).
    """
    tiles = []
    tile_coordinates = []

    # Calculate step size (tile size minus overlap)
    step_x = slice_width - overlap
    step_y = slice_height - overlap

    # Iterate over grid positions
    for y1 in range(0, max_y, step_y):
        for x1 in range(0, max_x, step_x):
            x2 = x1 + slice_width
            y2 = y1 + slice_height

            # Create a polygon for the tile
            pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            if x2 <= max_x and y2 <= max_y:
                # Append to results
                tiles.append(pol)
                tile_coordinates.append({
                    "height_i": y1 // step_y,
                    "width_j": x1 // step_x,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # Break when the tile reaches the max dimensions
            if x2 == max_x:
                break
        if y2 == max_y:
            break

    return tiles, tile_coordinates




def create_regular_raster_grid_from_center(max_x: int, max_y: int,
                                           slice_height: int,
                                           slice_width: int,
                                           overlap: int = 0) -> [List[Polygon], List[Dict]]:
    """
    Create a regular raster grid of oriented bounding boxes, starting from the center of the image.
    """
    tiles = []
    tile_coordinates = []

    center_x = max_x / 2
    center_y = max_y / 2

    # Calculate the number of tiles needed in each direction from the center
    num_tiles_x = int(np.ceil(max_x / slice_width))
    num_tiles_y = int(np.ceil(max_y / slice_height))

    # Generate x and y positions starting from the center
    x_positions = []
    y_positions = []

    # Starting positions for x and y
    x_start = center_x - slice_width / 2
    y_start = center_y - slice_height / 2

    # Function to generate positions along one axis
    def generate_positions(start, max_value, slice_size):
        positions = []
        positions_set = set()
        # Center position
        if 0 <= start <= max_value - slice_size:
            positions.append(start)
            positions_set.add(start)
        # Move outwards from the center
        offset = slice_size
        while True:
            added = False
            # Left/Up position
            pos1 = start - offset
            if 0 <= pos1 <= max_value - slice_size and pos1 not in positions_set:
                positions.append(pos1)
                positions_set.add(pos1)
                added = True
            # Right/Down position
            pos2 = start + offset
            if 0 <= pos2 <= max_value - slice_size and pos2 not in positions_set:
                positions.append(pos2)
                positions_set.add(pos2)
                added = True
            if not added:
                break
            offset += slice_size
        return positions

    # Generate x and y positions
    x_positions = generate_positions(x_start, max_x, slice_width)
    y_positions = generate_positions(y_start, max_y, slice_height)

    # Sort positions to maintain order from top-left to bottom-right
    x_positions = sorted(x_positions)
    y_positions = sorted(y_positions)

    # Generate tiles and calculate distances from the center
    tiles_info = []

    for height_i, y1 in enumerate(y_positions):
        y2 = y1 + slice_height
        for width_j, x1 in enumerate(x_positions):
            x2 = x1 + slice_width

            # Ensure tiles are within image boundaries
            x1_clipped = max(x1, 0)
            y1_clipped = max(y1, 0)
            x2_clipped = min(x2, max_x)
            y2_clipped = min(y2, max_y)

            # Adjust tile size if it goes beyond boundaries
            if x2_clipped - x1_clipped <= 0 or y2_clipped - y1_clipped <= 0:
                continue

            # Calculate the center of the tile
            tile_center_x = (x1_clipped + x2_clipped) / 2
            tile_center_y = (y1_clipped + y2_clipped) / 2

            # Calculate the Euclidean distance from the tile center to the image center
            distance = ((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2) ** 0.5

            # Append tile information
            tiles_info.append((distance, height_i, width_j, x1_clipped, y1_clipped, x2_clipped, y2_clipped))

    # Sort the tiles based on their distance from the image center (closest first)
    tiles_info_sorted = sorted(tiles_info, key=lambda x: x[0])

    # Generate tiles starting from the center
    for info in tiles_info_sorted:
        distance, height_i, width_j, x1, y1, x2, y2 = info

        # Create the polygon for the tile
        pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        # Append the polygon and its coordinates to the lists
        tiles.append(pol)
        tile_coordinates.append({
            "height_i": height_i,
            "width_j": width_j,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return tiles, tile_coordinates



def create_regular_raster_grid_from_center_2(max_x: int, max_y: int,
                                           slice_height: int,
                                           slice_width: int,
                                           overlap: int = 0) -> [List[Polygon], List[Dict]]:
    """
    Create a regular raster grid of oriented bounding boxes, starting from the center of the image,
    with the center point located at the intersection of four tiles.
    """
    tiles = []
    tile_coordinates = []

    center_x = max_x / 2
    center_y = max_y / 2

    # Function to generate positions along one axis
    def generate_positions(center, max_value, slice_size):
        positions = []
        positions_set = set()
        # Positive direction from center
        offset = 0
        while center + offset < max_value:
            pos = center + offset
            if pos not in positions_set and 0 <= pos <= max_value - slice_size:
                positions.append(pos)
                positions_set.add(pos)
            offset += slice_size
        # Negative direction from center
        offset = -slice_size
        while center + offset >= 0:
            pos = center + offset
            if pos not in positions_set and 0 <= pos <= max_value - slice_size:
                positions.append(pos)
                positions_set.add(pos)
            offset -= slice_size
        return sorted(positions)

    # Generate x and y positions
    x_positions = generate_positions(center_x, max_x, slice_width)
    y_positions = generate_positions(center_y, max_y, slice_height)

    # Generate tiles and calculate distances from the center
    tiles_info = []

    for height_i, y1 in enumerate(y_positions):
        y2 = y1 + slice_height
        for width_j, x1 in enumerate(x_positions):
            x2 = x1 + slice_width

            # Ensure tiles are within image boundaries
            x1_clipped = max(x1, 0)
            y1_clipped = max(y1, 0)
            x2_clipped = min(x2, max_x)
            y2_clipped = min(y2, max_y)

            # Adjust tile size if it goes beyond boundaries
            if x2_clipped - x1_clipped <= 0 or y2_clipped - y1_clipped <= 0:
                continue

            # Calculate the center of the tile
            tile_center_x = (x1_clipped + x2_clipped) / 2
            tile_center_y = (y1_clipped + y2_clipped) / 2

            # Calculate the Euclidean distance from the tile center to the image center
            distance = ((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2) ** 0.5

            # Append tile information
            tiles_info.append((distance, height_i, width_j, x1_clipped, y1_clipped, x2_clipped, y2_clipped))

    # Sort the tiles based on their distance from the image center (closest first)
    tiles_info_sorted = sorted(tiles_info, key=lambda x: x[0])

    # Generate tiles starting from the center
    for info in tiles_info_sorted:
        distance, height_i, width_j, x1, y1, x2, y2 = info

        # Create the polygon for the tile
        pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        # Append the polygon and its coordinates to the lists
        tiles.append(pol)
        tile_coordinates.append({
            "height_i": height_i,
            "width_j": width_j,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return tiles, tile_coordinates




def create_random_oriented_bounding_boxes(
                                          x_max: int, y_max: int,
                                          box_width: int, box_height: int) -> (shapely.Polygon, int):
    # TODO do I want only full boxes or also partial boxes? A partial box has a part outside the boundaries

    """ create random oriented bounding boxes which reflect the cutout regions """
    x_c = random.randint(0, x_max - box_width)
    y_c = random.randint(0, y_max - box_height)
    angle = random.randint(0, 360)
    # create the 4 corners, then rotate them by a random angle
    bbox_obb = shapely.Polygon([(0, 0), (box_width, 0), (box_width, box_height), (0, box_height)])

    bbox_obb = shapely.affinity.translate(bbox_obb, xoff=x_c, yoff=y_c)

    bbox_obb = shapely.affinity.rotate(bbox_obb, angle, origin='centroid', use_radians=False)

    return bbox_obb, angle


def find_point_within_polygon(polygon: shapely.Polygon) -> shapely.Point:
    """
    Find the closest point within a polygon to a given point
    """
    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Check if the centroid is inside the polygon
    if polygon.contains(centroid):
        closest_point = centroid
    else:
        # Find the closest point within the polygon
        closest_point = polygon.exterior.interpolate(polygon.exterior.project(centroid))

    closest_point = shapely.Point( int(round(closest_point.x)), int(round(closest_point.y)))

    return closest_point


def chebyshev_center(coords: list) -> shapely.Point:
    """
    find the Chebyshev center of a polygon, which is not the centroid but the point that is furthest away from the edges
    In a Snake shaped polygon, the centroid would be in the middle of the snake, but the Chebyshev center would be at the head of the snake

    :param coords:
    :return:
    """
    # Compute the Voronoi diagram for the vertices of the triangles
    points = np.array(coords)
    vor = Voronoi(points)

    polygon = Polygon(coords)

    # Find the Voronoi vertices inside the polygon
    voronoi_vertices = [shapely.Point(vertex) for vertex in vor.vertices if polygon.contains(shapely.Point(vertex))]

    # Calculate the distance to the polygon's edges for each vertex
    max_distance = 0
    center_point = None
    for vertex in voronoi_vertices:
        distance = vertex.distance(polygon.exterior)
        if distance > max_distance:
            max_distance = distance
            center_point = vertex

    center_point = shapely.Point(int(round(center_point.x)), int(round(center_point.y)))
    return center_point