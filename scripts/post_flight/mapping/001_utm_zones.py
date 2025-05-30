import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box, LineString
import pyproj
import numpy as np

from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks, island_utm_zones

web_mercator_projection_epsg = 3857





def create_utm_and_equator_lines(islands_gdf):
    """
    Create boundary lines for UTM zones 15/16 and the equator
    """
    # Get the bounds of the map in WGS84
    islands_wgs84 = islands_gdf.to_crs(epsg=4326)
    bounds = islands_wgs84.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    # Buffer the bounds a bit to ensure lines extend beyond islands
    lon_buffer = (max_lon - min_lon) * 0.1
    lat_buffer = (max_lat - min_lat) * 0.1

    # UTM Zone 15 and 16 boundary is at 90°W (or -90 in decimal degrees)
    utm_boundary_lon = -90

    # Create a line from min_lat to max_lat at the boundary longitude
    utm_boundary = LineString([
        (utm_boundary_lon, min_lat - lat_buffer),
        (utm_boundary_lon, max_lat + lat_buffer)
    ])

    # Create equator line (latitude 0)
    equator = LineString([
        (min_lon - lon_buffer, 0),
        (max_lon + lon_buffer, 0)
    ])

    # Convert to GeoDataFrame
    lines_gdf = gpd.GeoDataFrame(
        {'type': ['UTM boundary', 'Equator'],
         'geometry': [utm_boundary, equator]},
        crs="EPSG:4326"
    )

    # Transform to Web Mercator
    lines_wm = lines_gdf.to_crs(epsg=web_mercator_projection_epsg)

    return lines_wm


def create_galapagos_map(
        gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
        output_path="galapagos_utm_zones_map.png",
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
    important_islands = ["Santiago", "Wolf", "Darwin", "Santa Fé", "Española", "Isabela", "Baltra",
                         "Fernandina", "Floreana", "Santa Cruz", "San Cristóbal",
                         "Genovesa", "San Cristobal", "Marchena", "Pinta"]

    islands_wm_f = islands_wm[islands_wm['nombre'].isin(important_islands)]
    # Determine name column
    name_col = next((col for col in ['nombre', 'NAME', 'Island'] if col in islands.columns), None)
    if not name_col:
        raise ValueError("No valid name column found for labeling.")
    islands_wm_f = islands_wm_f.sort_values("porc_area", ascending=False).drop_duplicates(subset=[name_col])

    # Add UTM zone information to islands
    islands_wm['utm_zone'] = islands_wm['nombre'].map(lambda x: island_utm_zones.get(x, "Unknown"))

    # Create UTM boundary and equator lines
    grid_lines = create_utm_and_equator_lines(islands)

    # Plot islands with different colors based on UTM zone
    islands_wm_15n = islands_wm[islands_wm['utm_zone'] == "15N"]
    islands_wm_15s = islands_wm[islands_wm['utm_zone'] == "15S"]
    islands_wm_16n = islands_wm[islands_wm['utm_zone'] == "16N"]
    islands_wm_16s = islands_wm[islands_wm['utm_zone'] == "16S"]
    islands_wm_unknown = islands_wm[islands_wm['utm_zone'] == "Unknown"]

    # Plot islands by UTM zone with different colors
    islands_wm_15n.plot(ax=ax, alpha=0.7, edgecolor='black', color='#E63946', label='UTM Zone 15N')  # Bright red
    islands_wm_15s.plot(ax=ax, alpha=0.7, edgecolor='black', color='#F4A261', label='UTM Zone 15S')  # Orange/amber
    islands_wm_16n.plot(ax=ax, alpha=0.7, edgecolor='black', color='#457B9D', label='UTM Zone 16N')  # Deep blue
    islands_wm_16s.plot(ax=ax, alpha=0.7, edgecolor='black', color='#2A9D8F', label='UTM Zone 16S')  # Teal green
    islands_wm_unknown.plot(ax=ax, alpha=0.5, edgecolor='black', color='lightgrey')

    # Plot the UTM zone boundary and equator
    utm_boundary = grid_lines[grid_lines['type'] == 'UTM boundary']
    equator = grid_lines[grid_lines['type'] == 'Equator']

    utm_boundary.plot(ax=ax, color='red', linestyle='--', linewidth=2)
    equator.plot(ax=ax, color='blue', linestyle='-', linewidth=2)

    # Add UTM zone labels
    # Calculate positions for the labels (centered in each zone)
    map_bounds = islands_wm.total_bounds
    utm_bound_x = utm_boundary.geometry.iloc[0].coords[0][0]
    equator_y = equator.geometry.iloc[0].coords[0][1]

    label_15n_x = (map_bounds[0] + utm_bound_x) / 2
    label_15n_y = (equator_y + map_bounds[3]) /2

    label_15s_x = (map_bounds[0] + utm_bound_x) / 2
    label_15s_y = (map_bounds[1] + equator_y) / 2

    label_16n_x = (utm_bound_x + map_bounds[2]) / 2
    label_16n_y = (equator_y + map_bounds[3]) / 2

    label_16s_x = (utm_bound_x + map_bounds[2]) / 2
    label_16s_y = (map_bounds[1] + equator_y) / 2

    # Place UTM zone labels in the corners with fixed offsets
    offset_x = 1000  # Horizontal offset from corner
    offset_y = 1000  # Vertical offset from corner

    # 15N - Northwest corner
    label_15n_x = map_bounds[0] + offset_x
    label_15n_y = map_bounds[3] - offset_y

    # 15S - Southwest corner
    label_15s_x = map_bounds[0] + offset_x
    label_15s_y = map_bounds[1] + offset_y

    # 16N - Northeast corner
    label_16n_x = map_bounds[2] - offset_x
    label_16n_y = map_bounds[3] - offset_y

    # 16S - Southeast corner
    label_16s_x = map_bounds[2] - offset_x
    label_16s_y = map_bounds[1] + offset_y

    # Add text labels for UTM zones
    ax.text(label_15n_x, label_15n_y, "UTM Zone 15N",
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    ax.text(label_15s_x, label_15s_y, "UTM Zone 15S",
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    ax.text(label_16n_x, label_16n_y, "UTM Zone 16N",
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    ax.text(label_16s_x, label_16s_y, "UTM Zone 16S",
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    # Add equator label
    equator_mid_x = (map_bounds[0] + map_bounds[2]) / 2
    ax.text(equator_mid_x, equator_y + 9000, "Equator",
            fontsize=12, ha='center', va='center', color='blue',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue', boxstyle='round,pad=0.5'))


    # Title
    ax.set_title('Galápagos Islands - UTM Zones Coverage', fontsize=14)


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
    galapagos_union = islands_wm.geometry.unary_union

    # Add a light grid
    ax.grid(linestyle='--', alpha=0.5, zorder=0)

    # Style the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8)


    # Add north arrow
    na = NorthArrow(location="center left", rotation={"degrees": 0}, scale=0.5)
    ax.add_artist(na.copy())

    # # Draw scale bar
    # draw_accurate_scalebar(ax, islands_wm,
    #                        location=(islands_wm.total_bounds[0] + 100,
    #                                  islands_wm.total_bounds[3]),
    #                        length_km=100,  # Total length in km
    #                        segments=4,
    #                        height=6500)


    # Cleanup main map
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    create_galapagos_map(
        gpkg_path="/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/sampling_issues.gpkg")