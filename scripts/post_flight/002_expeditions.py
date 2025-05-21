"""
Plots the Expedition Data for multiple years

TODO next map: where are the data points for volunteers

TODO: One map per island
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

web_mercator_projection_epsg = 3857


def format_lat_lon(value, pos, is_latitude=True):
    """Format latitude or longitude as degrees with cardinal direction"""
    if is_latitude:
        direction = "S" if value < 0 else "N"
    else:
        direction = "W" if value < 0 else "E"
    # Absolute value and round to 1 decimal place
    value = abs(round(value, 1))
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

def draw_accurate_scalebar(ax, islands_wm, location=(0.1, 0.05), length_km=100, segments=4, height=200):
    """
    Draw a scalebar that accounts for the scale variation in Web Mercator projection
    by using the center latitude of the map.
    """
    # Get map bounds in Web Mercator
    x_min, y_min, x_max, y_max = islands_wm.total_bounds

    # Get the center latitude in geographic coordinates
    center_y = (y_min + y_max) / 2
    transformer = pyproj.Transformer.from_crs(web_mercator_projection_epsg, 4326, always_xy=True)
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

# Function to find the closest island
def find_closest_island(point_geometry, islands_gdf, name_col):
    distances = islands_gdf.distance(point_geometry)
    min_distance_idx = distances.idxmin()
    closest_island = islands_gdf.iloc[min_distance_idx]
    return closest_island[name_col]


def get_largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)
    return None


def create_galapagos_map(
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        fligth_database_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet",
        output_path="galapagos_map.png",
        dpi=300):
    """
    Creates a map of the Galápagos Islands showing data locations and annotations.
    """

    name_col = "gr_isla"
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

    for folder_name, group in flight_database.groupby("folder_name"):
        # Calculate the centroid of all points in this folder
        # This is more representative than a single point
        points_unary = group.geometry.union_all()

        # If we got a single point, use it directly; otherwise get the centroid
        if hasattr(points_unary, 'centroid'):
            representative_point = points_unary.centroid
        else:
            representative_point = points_unary  # It's already a point

        # Find the closest island to this representative point
        closest_island = find_closest_island(representative_point, islands_wm_f, name_col)

        # Store in our mapping dictionary
        folder_island_map[folder_name] = closest_island

        logger.info(f"Folder {folder_name}: Closest island is {closest_island}")

    # assign the folder name to the islands dataframe
    flight_database['main_island'] = flight_database['folder_name'].map(folder_island_map)

    for idx, isla in islands_wm_f.iterrows():
        island_name = isla[name_col]  # Get the island name from the row

        # Filter flight database for this island
        flight_database_island = flight_database[flight_database['main_island'] == island_name]

        # If no data for this island, skip it
        if flight_database_island.empty:
            print(f"No data for island: {island_name}")
            continue

        # Get distinct years for this island
        distinct_years = sorted(flight_database_island['year'].unique())
        if not distinct_years:
            logger.error(f"No year data for island: {island_name}")
            continue

        # Create a figure with subplots - one for each year
        fig, axes = plt.subplots(1, len(distinct_years), figsize=(6 * len(distinct_years), 8),
                                 squeeze=False)
        # Loop through each year for this island
        for i, year in enumerate(distinct_years):
            ax = axes[0, i]  # Get the appropriate subplot

            # Filter data for this year
            year_data = flight_database_island[flight_database_island['year'] == year]

            # Create a GeoDataFrame for just this island
            gdf_island = gpd.GeoDataFrame(pd.DataFrame([isla]), geometry="geometry", crs=islands_wm_f.crs)

            # Plot the island
            gdf_island.plot(ax=ax, alpha=0.7, edgecolor='black', color='lightgrey')

            # Plot the points for this year
            if not year_data.empty:
                year_data.plot(ax=ax, marker='o', color='red', markersize=5,
                               label=f"{len(year_data)} photos")

            # Set title and labels
            ax.set_title(f'{year} ({len(year_data)} photos)', fontsize=12)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend()

            # Add grid lines
            ax.grid(alpha=0.3)

        # fig.suptitle(f'Photo Locations on {island_name} by Year', fontsize=16)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)  # Make room for suptitle
        plt.subplots_adjust(top=0.6)  # Uncomment this line and put it AFTER tight_layout

        # Save the figure
        output_file = f"island_{island_name.replace(' ', '_')}_by_year.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot for {island_name} to {output_file}")
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

    # Plot all islands
    islands_wm.plot(ax=ax, alpha=0.7, edgecolor='black', color='lightgrey')

    # Dictionary of label offsets (dx, dy) in map units
    # Positive values: dx = right, dy = up
    # Negative values: dx = left, dy = down
    label_offsets = {
        "Santiago": (30000, 10000),
        "Wolf": (1000, 5000),
        "Santa Fé": (00, -9000),
        "Española": (10000, 9000),
        "Isabela": (-10000, 0),
        "Fernandina": (-15000, -21000),
        "Floreana": (00, -14000),
        "Santa Cruz": (35000, 10000),
        "San Cristóbal": (70000, -10000),
        "Genovesa": (10000, 10000),
        "San Cristobal": (10000, 20000),
        "Marchena": (10000, 10000),
        "Pinta": (10000, 10000),


    }

    # Place labels with manual offsets
    for idx, row in islands_wm_f.iterrows():
        # Get the largest polygon for this island
        island_poly = get_largest_polygon(row.geometry)
        if island_poly is None:
            continue

        # Get centroid of island
        centroid = island_poly.centroid

        # Get island name
        island_name = row['nombre']

        # Get offset from dictionary (or default if not defined)
        offset = label_offsets.get(island_name, (0, 0))

        # Apply offset to centroid
        label_x = centroid.x + offset[0]
        label_y = centroid.y + offset[1]

        # Add the label (no box, no arrow)
        ax.text(label_x, label_y, island_name,
                fontsize=10, ha='center', va='center')



    # Title


    # Add overview map - Fixed approach
    # Create a new axes for the inset map with absolute positioning
    # axins = fig.add_axes([0.6, 0.7, 0.25, 0.25])  # [left, bottom, width, height] in figure coordinates

    # Try to load South America (or at least Ecuador)
    # Load countries of South America
    countries = gpd.read_file(gpkg_path, layer='ne_10m_admin_0_countries')

    # Filter to Ecuador and surrounding countries
    south_america = countries[countries['CONTINENT'].isin(['South America', 'North America'])]
    area_of_interest = gpd.read_file(gpkg_path, layer='broader_area')
    bounds = area_of_interest.geometry.union_all().bounds

    # Create a box from the bounds
    box_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
    south_america = south_america.clip(box_polygon)

    south_america = south_america.to_crs(epsg=web_mercator_projection_epsg)

    # # Plot South America
    # south_america.plot(ax=axins, color='lightgrey', edgecolor='black', linewidth=0.5, alpha=0.7)
    #
    # # Highlight Ecuador
    ecuador = south_america[south_america['ADMIN'] == 'Ecuador']
    # if not ecuador.empty:
    #     ecuador.plot(ax=axins, color='lightgrey', edgecolor='black', linewidth=0.8)

    info_text = """Data source: (Sanchez, 2024), )
Base map: Islands data from Ecuadorian National Geographic Institute
Survey period: 2020-2024
Map projection: Web Mercator (EPSG:3857)
Marine Iguana population assessment in the Galápagos Archipelago
HNEE/Winkelmann, 2025"""

    # Add information text box
    add_text_box(
        ax,
        info_text,
        location='upper right',
        fontsize=8,
        alpha=0.8,
        pad=0.5
    )

    # Title
    ax.set_title('Galápagos Islands - Marine Iguana Survey Data', fontsize=14)



    # And then in your main function, replace the tick formatter code with:

    x_ticks_proj, x_ticks_geo, y_ticks_proj, y_ticks_geo = get_geographic_ticks(ax,
                                                                                epsg_from=web_mercator_projection_epsg,
                                                                                epsg_to=4326)

    # Set the tick positions to the projected coordinates
    ax.set_xticks(x_ticks_proj)
    ax.set_yticks(y_ticks_proj)

    # Format the tick labels with the geographic coordinates
    ax.set_xticklabels([format_lat_lon(x, 0, is_latitude=False) for x in x_ticks_geo])
    ax.set_yticklabels([format_lat_lon(y, 0, is_latitude=True) for y in y_ticks_geo])

    # Get the union of all Galápagos islands
    galapagos_union = islands_wm.geometry.union_all()

    # Get the bounds of the unified geometry
    galapagos_bounds = galapagos_union.bounds  # This gives (minx, miny, maxx, maxy)

    # # Set appropriate tick intervals based on map extent
    # x_span = galapagos_bounds[2] - galapagos_bounds[0]
    # y_span = galapagos_bounds[3] - galapagos_bounds[1]
    #
    # # Calculate nice intervals (about 3-5 ticks in each direction)
    # x_interval = round(x_span / 4, 1)  # Round to nearest 0.1 degree
    # y_interval = round(y_span / 4, 1)
    #
    # # Make sure intervals are at least 0.5 degrees
    # x_interval = max(0.5, x_interval)
    # y_interval = max(0.5, y_interval)
    #
    # # Set tick positions
    # x_ticks = mticker.MultipleLocator(x_interval)
    # y_ticks = mticker.MultipleLocator(y_interval)
    # ax.xaxis.set_major_locator(x_ticks)
    # ax.yaxis.set_major_locator(y_ticks)

    # Add a light grid
    ax.grid(linestyle='--', alpha=0.4, zorder=0)

    # Style the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8)


    # Adding an inset map to the plot
    iax = inset_map(ax, location="upper left", size=2.5, pad=0.1, xticks=[], yticks=[])
    # Plotting alaska in the inset map
    south_america.plot(ax=iax, color='lightgrey', edgecolor='black', linewidth=0.5, alpha=0.7)
    if not ecuador.empty:
        ecuador.plot(ax=iax, color='#AAAAFF', edgecolor='black', linewidth=0.8)

    # Creating the extent indicator, which appears by-default as a red square on the map
    indicate_extent(iax, ax, web_mercator_projection_epsg, web_mercator_projection_epsg)

    na = NorthArrow(location="center left", rotation={"degrees": 0}, scale=0.5)
    # Note that you have to use .copy() here too!
    ax.add_artist(na.copy())

    # # Add scale bar
    # ax.add_artist(ScaleBar(1, dimension='si-length', length_fraction=0.25,
    #                 scale_loc='top',
    #                 units='m',
    #                 location='lower left',
    #                 color='black',
    #                 box_color='white',
    #                 box_alpha=0.8))

    # Replace your draw_segmented_scalebar call with:
    draw_accurate_scalebar(ax, islands_wm,
                           location=(islands_wm.total_bounds[0] + 100,
                                     islands_wm.total_bounds[1] + 100),
                           length_km=100,  # Total length in km
                           segments=4,
                           height=6500)

    # draw_segmented_scalebar(ax, start=(islands_wm.total_bounds[0], islands_wm.total_bounds[1]),
    #                         segments=3, segment_length=30000,
    #                         height=6500, units="km")

    # TODO add legend

    # Cleanup main map
    # ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    logger.info(f"Map saved to {output_path}")


if __name__ == "__main__":
    create_galapagos_map(gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg")