import geopandas as gpd
# Packages used by this tutorial
import matplotlib.pyplot as plt  # visualization
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
# Importing the main package
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box

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
            ax.text(x, y0 - height * 0.6, f"{i * segment_length:.0f} {units}",
                    ha='center', va='top', fontsize=8)

    # Final label at the end
    ax.text(x0 + segments * segment_length, y0 - height * 0.6,
            f"{segments * segment_length} {units}", ha='center', va='top', fontsize=8)


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


def create_fancy_north_arrow(ax, x, y, size=0.08, zorder=10):
    """Create a fancy north arrow at the given position"""
    # Get axis dimensions to scale arrow properly
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    arrow_size = min(x_range, y_range) * size

    # Create the arrow
    arrow = FancyArrowPatch(
        (x, y),
        (x, y + arrow_size),
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=1.5,
        color='black',
        zorder=zorder
    )
    ax.add_patch(arrow)

    # Add the 'N' letter
    ax.text(
        x, y + arrow_size * 1.2,
        'N',
        fontsize=12,
        ha='center',
        va='center',
        fontweight='bold',
        zorder=zorder
    )

    # Add a circle at the base
    circle_radius = arrow_size * 0.15
    circle = plt.Circle(
        (x, y),
        circle_radius,
        color='white',
        ec='black',
        lw=1.5,
        zorder=zorder
    )
    ax.add_patch(circle)

def get_largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)
    return None


def create_galapagos_map(
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        output_path="galapagos_map.png",
        dpi=300):
    """
    Creates a map of the Galápagos Islands showing data locations and annotations.
    """
    # Load GeoPackage layers
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')

    # Prepare base plot
    fig, ax = plt.subplots(figsize=(12, 10))
    islands_wm = islands.to_crs(epsg=web_mercator_projection_epsg)
    islands_wm = islands_wm[islands_wm['tipo'] == 'Isla']

    # Filter for these islands to plot nicely
    important_islands = ["Santiago", "Wolf", "Santa Fé", "Española", "Isabela",
                         "Fernandina", "Floreana", "Santa Cruz", "San Cristóbal",
                          "Genovesa", "San Cristobal", "Marchena", "Pinta"]

    islands_wm_f = islands_wm[islands_wm['nombre'].isin(important_islands)]
    # Determine name column
    name_col = next((col for col in ['nombre', 'NAME', 'Island'] if col in islands.columns), None)
    if not name_col:
        raise ValueError("No valid name column found for labeling.")
    islands_wm_f = islands_wm_f.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

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
    ax.set_title('Galápagos Islands - Marine Iguana Survey Data', fontsize=14)

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

    info_text = """Data source: Iguanas From Above project (MacLeod et al., 2023)
Base map: Islands data from Ecuadorian National Geographic Institute
Survey period: 2020-2024
Map projection: Web Mercator (EPSG:web_mercator_projection_epsg)
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

    # ===== CUSTOMIZE GRID AND TICKS =====
    # Format latitude and longitude ticks
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_lat_lon(x, pos, is_latitude=False)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: format_lat_lon(y, pos, is_latitude=True)))

    # Get the union of all Galápagos islands
    galapagos_union = islands_wm.geometry.unary_union

    # Get the bounds of the unified geometry
    galapagos_bounds = galapagos_union.bounds  # This gives (minx, miny, maxx, maxy)

    # Set appropriate tick intervals based on map extent
    x_span = galapagos_bounds[2] - galapagos_bounds[0]
    y_span = galapagos_bounds[3] - galapagos_bounds[1]

    # Calculate nice intervals (about 3-5 ticks in each direction)
    x_interval = round(x_span / 4, 1)  # Round to nearest 0.1 degree
    y_interval = round(y_span / 4, 1)

    # Make sure intervals are at least 0.5 degrees
    x_interval = max(0.5, x_interval)
    y_interval = max(0.5, y_interval)

    # Set tick positions
    x_ticks = mticker.MultipleLocator(x_interval)
    y_ticks = mticker.MultipleLocator(y_interval)
    ax.xaxis.set_major_locator(x_ticks)
    ax.yaxis.set_major_locator(y_ticks)

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

    draw_segmented_scalebar(ax, start=(islands_wm.total_bounds[0], islands_wm.total_bounds[1]),
                            segments=3, segment_length=30000,
                            height=6500, units="km")

    # TODO add legend

    # Cleanup main map
    # ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    create_galapagos_map()