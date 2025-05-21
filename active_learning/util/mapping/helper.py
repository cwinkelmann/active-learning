"""

"""

import geopandas as gpd
# Packages used by this tutorial
import matplotlib.pyplot as plt  # visualization
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
# Importing the main package
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box
import pyproj
import numpy as np
import numpy as np
from loguru import logger


def format_lat_lon(value, pos, is_latitude=True, rounding=1):
    """Format latitude or longitude as degrees with cardinal direction"""
    if is_latitude:
        direction = "S" if value < 0 else "N"
    else:
        direction = "W" if value < 0 else "E"
    # Absolute value and round to 1 decimal place
    value = abs(round(value, rounding))
    return f"{value}°{direction}"

# First convert the axis limits from Web Mercator to WGS84 for proper lat/lon labeling
def get_geographic_ticks(ax, epsg_from=3857, epsg_to=4326, n_ticks=5):
    """Convert projected coordinates to geographic coordinates for axis ticks"""
    transformer = pyproj.Transformer.from_crs(epsg_from, epsg_to, always_xy=True)

    # Get current axis limits in projected coordinates
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Create evenly spaced ticks in projected space
    x_ticks_proj = np.linspace(x_min, x_max, n_ticks)
    y_ticks_proj = np.linspace(y_min, y_max, n_ticks)

    # Transform to geographic coordinates
    x_ticks_geo = []
    y_ticks_geo = []

    for x in x_ticks_proj:
        lon, _ = transformer.transform(x, 0)  # y value doesn't matter for longitude
        x_ticks_geo.append(lon)

    for y in y_ticks_proj:
        _, lat = transformer.transform(0, y)  # x value doesn't matter for latitude
        y_ticks_geo.append(lat)

    return x_ticks_proj, x_ticks_geo, y_ticks_proj, y_ticks_geo

def draw_accurate_scalebar(ax, islands_wm, location=(0.1, 0.05),
                           length_km=100, segments=4, height=200,
                           web_mercator_projection_epsg=3857, WSG84_projection_epsg=4326):
    """
    Draw a scalebar that accounts for the scale variation in Web Mercator projection
    by using the center latitude of the map.
    """
    # Get map bounds in Web Mercator
    x_min, y_min, x_max, y_max = islands_wm.total_bounds

    # Get the center latitude in geographic coordinates
    center_y = (y_min + y_max) / 2
    transformer = pyproj.Transformer.from_crs(web_mercator_projection_epsg, WSG84_projection_epsg, always_xy=True)
    _, center_lat = transformer.transform(0, center_y)

    # Calculate the scale factor at this latitude (1.0 at equator, increases with latitude)
    # Web Mercator scale factor formula: sec(lat) = 1/cos(lat)
    scale_factor = 1.0 / np.cos(np.radians(center_lat))

    # Calculate the length in projected coordinates
    # For Web Mercator at the equator, 1 degree is approximately 111.32 km
    meters_per_unit = 1  # Web Mercator uses meters
    length_proj = length_km * 1000 / meters_per_unit  # Convert km to projection units

    # Apply the scale factor correction
    length_proj_corrected = length_proj / scale_factor

    # Now draw the segmented scalebar
    x0, y0 = location
    segment_length = length_proj_corrected / segments

    for i in range(segments):
        x = x0 + i * segment_length
        color = 'black' if i % 2 == 0 else 'white'
        rect = plt.Rectangle((x, y0), segment_length, height, facecolor=color, edgecolor='black', zorder=5)
        ax.add_patch(rect)

        # Labels at every other segment
        if i % 2 == 0:
            display_length = i * length_km / segments
            ax.text(x, y0 - height * 0.6, f"{display_length:.0f} km",
                    ha='center', va='top', fontsize=8)

    # Final label
    ax.text(x0 + segments * segment_length, y0 - height * 0.6,
            f"{length_km:.0f} km", ha='center', va='top', fontsize=8)

def draw_segmented_scalebar(ax, start=(0.1, 0.05), segments=4, segment_length=1000, height=200,
                            crs_transform=None, units="m", label_step=2):
    """
    Draws a segmented scale bar directly on a matplotlib axis.
    """

    x0, y0 = start
    for i in range(segments):
        x = x0 + i * segment_length
        color = 'black' if i % 2 == 0 else 'white'
        rect = plt.Rectangle((x, y0), segment_length, height, facecolor=color, edgecolor='black', zorder=5)
        ax.add_patch(rect)
        if i % label_step == 0:
            # Convert segment length from meters to kilometers if larger than 1000m
            display_length = i * segment_length
            display_units = units
            if units == "m" and display_length >= 1000:
                display_length = display_length / 1000
                display_units = "km"

            ax.text(x, y0 - height * 0.6, f"{display_length:.0f} {display_units}",
                    ha='center', va='top', fontsize=8)

    # Final label at the end
    final_length = segments * segment_length
    final_units = units
    if units == "m" and final_length >= 1000:
        final_length = final_length / 1000
        final_units = "km"

    ax.text(x0 + segments * segment_length, y0 - height * 0.6,
            f"{final_length:.0f} {final_units}", ha='center', va='top', fontsize=8)


def add_text_box(ax, text, location='lower left', fontsize=8, alpha=0.8, pad=0.5, frameon=True):
    """Add a text box to the map"""
    text_box = AnchoredText(
        text,
        loc=location,
        frameon=frameon,
        prop=dict(fontsize=fontsize, backgroundcolor='white', alpha=alpha),
        pad=pad,
        borderpad=pad
    )
    ax.add_artist(text_box)
    return text_box



def get_largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)
    return None


# Function to find the closest island
def find_closest_island(point_geometry, islands_gdf, name_col):
    distances = islands_gdf.distance(point_geometry)
    min_distance_idx = distances.idxmin()
    closest_island = islands_gdf.iloc[min_distance_idx]
    return closest_island[name_col]


