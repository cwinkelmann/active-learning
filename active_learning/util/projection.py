
def world_to_pixel(geo_transform, x, y):
    """
    Converts georeferenced coordinates to pixel coordinates.

    Parameters:
    - geo_transform: Geotransform tuple of the dataset.
    - x: Georeferenced x-coordinate (e.g., longitude).
    - y: Georeferenced y-coordinate (e.g., latitude).

    Returns:
    - (pixel_x, pixel_y): Corresponding pixel coordinates.
    """
    originX = geo_transform[0]
    pixelWidth = geo_transform[1]
    originY = geo_transform[3]
    pixelHeight = geo_transform[5]

    pixel_x = int((x - originX) / pixelWidth)
    pixel_y = int((y - originY) / pixelHeight)

    return pixel_x, pixel_y

def pixel_to_world(geo_transform, pixel_x, pixel_y):
    """
    Converts pixel coordinates to georeferenced coordinates.

    Parameters:
    - geo_transform: Geotransform tuple of the dataset.
    - pixel_x: Pixel x-coordinate (column).
    - pixel_y: Pixel y-coordinate (row).

    Returns:
    - (x, y): Corresponding georeferenced coordinates.
    """
    x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return x, y



def