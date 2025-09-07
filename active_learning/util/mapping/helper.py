"""

"""

import geopandas as gpd
# Packages used by this tutorial
import pandas as pd
import pyproj
import shapely
from matplotlib.offsetbox import AnchoredText
# Importing the main package
from shapely.geometry import Polygon, MultiPolygon, box
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib import cm


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
                           length_km=100, segments=4, height_fraction=0.015,
                           web_mercator_projection_epsg=3857, WSG84_projection_epsg=4326):
    """
    Draw a scalebar that accounts for the scale variation in Web Mercator projection
    by using the center latitude of the map.

    Parameters:
    - height_fraction: float, height as fraction of map extent (default 0.015 = 1.5%)
    """
    # Get map bounds in Web Mercator
    x_min, y_min, x_max, y_max = islands_wm.total_bounds

    # Get current axis limits (actual displayed extent)
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()

    # Calculate height based on current map display extent
    map_height = ax_ylim[1] - ax_ylim[0]
    height = map_height * height_fraction

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

    # Convert location from relative to absolute coordinates
    if isinstance(location[0], float) and location[0] <= 1.0:
        # Location given as fractions - convert to map coordinates
        map_width = ax_xlim[1] - ax_xlim[0]
        x0 = ax_xlim[0] + location[0] * map_width
        y0 = ax_ylim[0] + location[1] * map_height
    else:
        # Location given as absolute coordinates
        x0, y0 = location

    # Now draw the segmented scalebar
    segment_length = length_proj_corrected / segments

    for i in range(segments):
        x = x0 + i * segment_length
        color = 'black' if i % 2 == 0 else 'white'
        rect = plt.Rectangle((x, y0), segment_length, height, facecolor=color, edgecolor='black', zorder=5)
        ax.add_patch(rect)

        # Labels at every other segment
        if i % 2 == 0 and i > 0:
            display_length = i * length_km / segments
            ax.text(x, y0 - height * 0.6, f"{display_length:.0f} km",
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                              edgecolor='gray', alpha=0.9))

    # Final label
    ax.text(x0 + segments * segment_length, y0 - height * 0.6,
            f"{length_km:.0f} km", ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor='gray', alpha=0.9))


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


def add_text_box(ax, text, location='lower left', fontsize=8, alpha=0.8, pad=0.5, frameon=True, xy=None):
    """Add a text box to the map with flexible positioning"""

    if xy is not None:
        # Use custom coordinates (x, y) - can be in data coordinates or axes fraction
        text_box = ax.text(
            xy[0], xy[1], text,
            fontsize=fontsize,
            bbox=dict(
                boxstyle=f"round,pad={pad}",
                facecolor='white',
                alpha=alpha,
                edgecolor='black' if frameon else 'none'
            ),
            transform=ax.transAxes if isinstance(xy[0], float) and 0 <= xy[0] <= 1 else ax.transData,
            verticalalignment='top',
            horizontalalignment='left'
        )
        return text_box
    else:
        # Use original AnchoredText approach
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

# Corrected Galápagos Island UTM Zone Assignments
# Using proper hemisphere-based UTM zones
island_utm_zones = {
    "Bartolome": "15S",    # ~-90.56°, south of equator
    "Caamaño": "15S",      # ~-90.3°, south of equator near Santa Cruz
    "Santiago": "15S",     # ~-90.8°, straddles equator but mostly south
    "Wolf": "15N",         # ~-91.8°, north of equator
    "San Cristobal": "16S", # ~-89.6°, south of equator
    "Lobos": "16S",        # ~-89.6°, south of equator, near San Cristobal
    "Santa Fé": "15S",     # ~-90.4°, south of equator
    "Santa Cruz": "15S",   # ~-90.3°, south of equator
    "Rabida": "15S",       # ~-90.7°, south of equator
    "Pinzón": "15S",       # ~-90.6°, south of equator
    "Pinta": "15N",        # ~-90.75°, north of equator
    "Marchena": "15N",     # ~-90.5°, north of equator
    "Isabela": "15S",      # ~-91.1°, straddles equator but mostly south
    "Tortuga": "15S",      # ~-91.4°, south of equator
    "Genovesa": "16N",     # ~-89.95°, north of equator
    "Fernandina": "15S",   # ~-91.6°, south of equator
    "Floreana": "15S",     # ~-90.3°, south of equator
    "Gardner por Floreana": "15S", # ~-90.3°, south of equator near Floreana
    "Caldwell": "16S",     # ~-89.6°, south of equator near Española
    "Albany": "16S",       # ~-89.6°, south of equator near Española
    "Española": "16S",     # ~-89.5°, south of equator
    "Daphne Major": "15S", # ~-90.3°, south of equator near Santa Cruz
}

def get_utm_epsg(utm_zone):
    """
    Get the EPSG code for a given UTM zone
    UTM zones are in the format '15N', '15S', etc.

    For UTM north zones: EPSG = 32600 + zone_number
    For UTM south zones: EPSG = 32700 + zone_number
    """
    zone_number = int(utm_zone[:-1])
    hemisphere = utm_zone[-1]

    if hemisphere == 'N':
        return 32600 + zone_number
    elif hemisphere == 'S':
        return 32700 + zone_number
    else:
        raise ValueError(f"Invalid UTM zone format: {utm_zone}")

# Dictionary mapping islands to their correct UTM zones
# Corrected to use proper N/S hemisphere designations
island_utm_zones = {
    "Bartolome": "15S",    # ~-90.56°, south of equator
    "Caamaño": "15S",      # ~-90.3°, south of equator near Santa Cruz
    "Santiago": "15S",     # ~-90.8°, straddles equator but mostly south
    "Wolf": "15N",         # ~-91.8°, north of equator
    "Darwin": "15N",         # ~-91.8°, north of equator
    "San Cristobal": "16S", # ~-89.6°, south of equator
    "Lobos": "16S",        # ~-89.6°, south of equator, near San Cristobal
    "Santa Fé": "15S",     # ~-90.4°, south of equator
    "Santa Cruz": "15S",   # ~-90.3°, south of equator
    "Baltra": "15S",   # ~-90.3°, south of equator
    "Rabida": "15S",       # ~-90.7°, south of equator
    "Pinzón": "15S",       # ~-90.6°, south of equator
    "Pinta": "15N",        # ~-90.75°, north of equator
    "Marchena": "15N",     # ~-90.5°, north of equator
    "Isabela": "15S",      # ~-91.1°, straddles equator but mostly south
    "Tortuga": "15S",      # ~-91.4°, south of equator
    "Genovesa": "16N",     # ~-89.95°, north of equator
    "Fernandina": "15S",   # ~-91.6°, south of equator
    "Floreana": "15S",     # ~-90.3°, south of equator
    "Gardner por Floreana": "15S", # ~-90.3°, south of equator near Floreana
    "Caldwell": "16S",     # ~-89.6°, south of equator near Española
    "Albany": "16S",       # ~-89.6°, south of equator near Española
    "Española": "16S",     # ~-89.5°, south of equator
    "Daphne Major": "15S", # ~-90.3°, south of equator near Santa Cruz
}

island_plot_config = {


    "Bartolome": {
        "utm_zone": "16M",
        "scalebar": {"location": (1200, 30), "length_km": 1, "segments": 4, "height": 50},
    },
    "Caamaño": {
        "utm_zone": "16M",
        "scalebar": {"location": (0, 20), "length_km": 0.1, "segments": 4, "height": 50},
    },
    "Santiago": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
    },
    "Wolf": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "San Cristobal": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
    },
    "Lobos": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Santa Fé": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
    },
    "Santa Cruz": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
    },
    "Rabida": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Pinzón": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Pinta": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Marchena": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Isabela": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
    },
    "Tortuga": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Genovesa": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Fernandina": {
        "utm_zone": "15M",
        "scalebar": {"location": (100, 100), "length_km": 10, "segments": 4, "height": 150},
    },
    "Floreana": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
    },
    "Gardner por Floreana": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Caldwell": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Albany": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    "Española": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 5, "segments": 4, "height": 150},
    },
    "Daphne Major": {
        "utm_zone": "16M",
        "scalebar": {"location": (100, 100), "length_km": 1, "segments": 4, "height": 150},
    },
    }


def get_largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)
    return None


def get_islands(
    gpkg_path = "/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
    fligth_database_path = "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet",
    output_path = "galapagos_map.png",
    dpi = 300,
    web_mercator_projection_epsg= 3857):
    """
    Creates a map of the Galápagos Islands showing data locations and annotations.
    """

    name_col = "nombre"




    # Load GeoPackage layers
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')
    # TODO use the geoparquet instead
    flight_database = gpd.read_parquet(fligth_database_path).to_crs(epsg=web_mercator_projection_epsg)

    # add year to dataframe to simplify grouping later
    flight_database['year'] = flight_database['datetime_digitized'].dt.year

    # Prepare base plot
    islands_wm = islands.to_crs(epsg=web_mercator_projection_epsg)

    islands_isla = islands_wm[islands_wm['tipo'] == 'Isla']
    #
    # Dictionary to store folder -> island mapping
    folder_island_map = {}
    # Determine name column
    if not name_col:
        raise ValueError("No valid name column found for labeling.")
    islands_wm_f = islands_isla.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

    # group by island
    islands_wm_f = islands_wm_f.dissolve(by=name_col, as_index=False)

    return islands_wm_f


# Function to find the closest island
def find_closest_island(point_geometry, islands_gdf, name_col):
    distances = islands_gdf.distance(point_geometry)
    min_distance_idx = distances.idxmin()
    closest_island = islands_gdf.iloc[min_distance_idx]
    return closest_island[name_col]


def mission_extend(gdf_mission) -> gpd.GeoDataFrame:
    """
    Extend the mission by 10% in each direction
    :param gdf_mission: GeoDataFrame with the mission
    :return: extended GeoDataFrame
    """

    # Get the bounding box of the mission
    minx, miny, maxx, maxy = gdf_mission.total_bounds

    # Calculate the width and height of the bounding box
    width = maxx - minx
    height = maxy - miny

    # Calculate the extension in each direction (10% of width/height)
    x_extension = 0.1 * width
    y_extension = 0.1 * height

    # Create a new bounding box with the extension
    extended_bbox = box(minx - x_extension, miny - y_extension, maxx + x_extension, maxy + y_extension)

    # Create a new GeoDataFrame with the extended bounding box
    gdf_extended = gpd.GeoDataFrame(geometry=[extended_bbox], crs=gdf_mission.crs)

    return gdf_extended

def get_mission_flight_length(gdf_mission: gpd.GeoDataFrame) -> float:
    """
    Calculate the flight length of the mission
    :param gdf_mission: GeoDataFrame with the mission
    :return: flight length in meters
    """
    # Get the geometry of the mission
    # sort by timestamp if available
    gdf_mission = gdf_mission.sort_values(by=['datetime_digitized'], ascending=True)
    mission_geometry = shapely.LineString(gdf_mission.geometry)

    # Calculate the length of the mission in meters
    flight_length = mission_geometry.length

    return flight_length

def get_mission_type(gdf_mission: gpd.GeoDataFrame) -> str:
    """
    Determine if the mission is a oblique cliff or nadir flights
    """
    # Get the geometry of the mission
    # when most of the images are oblique, the mission is oblique
    oblique_percentage = gdf_mission['is_oblique'].mean()

    if oblique_percentage > 0.5:
        return "oblique"
    else:
        return "nadir"


