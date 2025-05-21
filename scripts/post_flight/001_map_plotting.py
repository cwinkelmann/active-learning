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
import pyproj
import numpy as np

from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks

web_mercator_projection_epsg = 3857





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
    important_islands = ["Santiago", "Wolf", "Darwin", "Santa Fé", "Española", "Isabela",
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
        "Darwin": (1000, 5000),
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

    info_text = """Data source: (Sanchez, 2024), )
Inset map: Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.
Map projection: Web Mercator (EPSG:3857)
Marine Iguana population assessment in the Galápagos Archipelago
HNEE/Winkelmann, 2025"""

    # # Add information text box
    # add_text_box(
    #     ax,
    #     info_text,
    #     location='lower right',
    #     fontsize=8,
    #     alpha=0.8,
    #     pad=0.5
    # )

    # Title
    ax.set_title('Galápagos Islands - Marine Iguana Survey Data', fontsize=14)



    # And then in your main function, replace the tick formatter code with:
    import numpy as np
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
    galapagos_union = islands_wm.geometry.unary_union

    # Add a light grid
    ax.grid(linestyle='--', alpha=0.5, zorder=0)

    # Style the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8)


    # Adding an inset map to the plot
    iax = inset_map(ax, location="upper right", size=2.5, pad=0.1, xticks=[], yticks=[])
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
                           segments=3,
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
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    create_galapagos_map(gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg")