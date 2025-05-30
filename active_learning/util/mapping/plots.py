from pathlib import Path

from active_learning.util.mapping.helper import get_geographic_ticks, format_lat_lon, island_plot_config
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent, indicate_detail
from matplotlib_map_utils.core.north_arrow import NorthArrow
from shapely.geometry import Polygon, MultiPolygon, box
import pyproj
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib_map_utils.core.inset_map import inset_map, indicate_extent
from matplotlib_map_utils.core.north_arrow import NorthArrow

from active_learning.util.mapping.helper import get_largest_polygon, add_text_box, format_lat_lon, \
    draw_accurate_scalebar, get_geographic_ticks, find_closest_island

web_mercator_projection_epsg = 3857

def plot_orthomomsaic_training_data(group_extent, group_gdf, gdf_group_annotations, group_id, vis_config,
                                    gpkg_path="/Volumes/2TB/SamplingIssues/sampling_issues.gpkg",
                                    output_dir=None):
    """
    Plot orthomosaic training data for a specific group showing:
    - Group polygons (raster masks)
    - Annotations (points)
    - Location on Galápagos inset map
    """

    epsg = vis_config["epsg"]
    # Load islands data
    islands = gpd.read_file(gpkg_path, layer='islands_galapagos')
    islands_all = islands.to_crs(epsg=epsg)
    islands_wm = islands_all[islands_all['tipo'] == 'Isla']
    islands_wm = islands_wm[islands_wm['nombre'] == vis_config["island"]]



    # Convert group data to Web Mercator for plotting
    group_gdf_wm = group_gdf.to_crs(epsg=epsg)
    if not gdf_group_annotations.empty:
        annotations_wm = gdf_group_annotations.to_crs(epsg=epsg)
    else:
        annotations_wm = gpd.GeoDataFrame()

    # Create figure with main plot
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    group_bounds = group_gdf_wm.total_bounds
    # Plot group polygons (raster masks)
    # Highlight the group location on inset map
    islands_clip = islands_all.cx[group_bounds[0]:group_bounds[2], group_bounds[1]:group_bounds[3]]
    islands_clip.plot(ax=ax, facecolor='grey', alpha=0.5, edgecolor='black', linewidth=1.5)

    # Apply offset to centroid
    island_label_x = islands_clip.centroid.x.iloc[0]
    island_label_y = islands_clip.centroid.y.iloc[0]



    group_gdf_wm.plot(ax=ax, alpha=0.8, edgecolor='red', facecolor='lightblue', linewidth=2)

    # Plot annotations if available
    if not annotations_wm.empty:
        annotations_wm.plot(ax=ax, color='red', markersize=50, alpha=0.8, marker='o')

    # Set extent to group bounds with some padding

    padding = max(group_bounds[2] - group_bounds[0], group_bounds[3] - group_bounds[1]) * 0.1
    ax.set_xlim(group_bounds[0] - padding, group_bounds[2] + padding)
    ax.set_ylim(group_bounds[1] - padding, group_bounds[3] + padding)

    # Add title with group info
    num_polygons = len(group_gdf)
    num_annotations = len(annotations_wm) if not annotations_wm.empty else 0
    total_area = group_gdf_wm.area.sum() / 1e6  # Convert to km²

    ax.set_title(f'Training Data Group {group_id}\n'
                 f'{num_polygons} orthomosaics, {num_annotations} annotations, '
                 f'{total_area:.2f} km² total area',
                 fontsize=14, pad=20)

    # Add coordinate grid
    x_ticks_proj, x_ticks_geo, y_ticks_proj, y_ticks_geo = get_geographic_ticks(
        ax, epsg_from=epsg, epsg_to=4326)

    ax.set_xticks(x_ticks_proj)
    ax.set_yticks(y_ticks_proj)
    #ax.set_xticklabels([format_lat_lon(x, 0, is_latitude=False) for x in x_ticks_geo])
    #ax.set_yticklabels([format_lat_lon(y, 0, is_latitude=True) for y in y_ticks_geo])

    # Add grid
    ax.grid(linestyle='--', alpha=0.5, zorder=0)

    # Style tick labels
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(9)

    # Add inset map showing location in Galápagos
    iax = inset_map(ax, location=vis_config["inset_map_location"], size=3, pad=0.1, xticks=[], yticks=[])

    # Plot all Galápagos islands in inset
    islands_wm.plot(ax=iax, alpha=0.7, edgecolor='black', color='lightgrey')
    # Add the label (no box, no arrow)
    iax.text(island_label_x, island_label_y, vis_config["island"],
            fontsize=12, ha='center', va='center')

    group_gdf_wm.plot(ax=iax, color='red', markersize=30, alpha=0.8)

    # Add extent indicator
    indicate_extent(iax, ax, epsg, epsg)

    # Add north arrow
    na = NorthArrow(location="upper right", rotation={"degrees": 0}, scale=0.4)
    ax.add_artist(na.copy())

    # # Add scale bar
    # draw_accurate_scalebar(ax, group_gdf_wm,
    #                        location=(group_bounds[0] + padding / 2, group_bounds[1] + padding / 2),
    #                        length_km=min(10, (group_bounds[2] - group_bounds[0]) / 1000 / 3),  # Adaptive scale
    #                        segments=2,
    #                        height=max(1000, padding / 10))

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue',
               markeredgecolor='red', markersize=10, label='Orthomosaic coverage'),
    ]
    if not annotations_wm.empty:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=4, label=f'Iguana annotations ({num_annotations})')
        )

    ax.legend(handles=legend_elements, loc=vis_config["legend_location"], frameon=True,
              fancybox=True, shadow=True, framealpha=0.9)

    # Add info text box
    info_text = f"""Group {group_id} Summary:
Orthomosaics: {num_polygons}
Annotations: {num_annotations}
Total area: {total_area:.2f} km² 
Projection EPSG: {epsg}"""
# Missions: {', '.join(group_gdf['Orthophoto/Panorama name'].unique())}"""

    add_text_box(ax, info_text, xy=vis_config["textbox_location"], fontsize=9, alpha=0.8, pad=0.5)

    plt.tight_layout()

    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir) / f"training_data_group_{group_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved group {group_id} plot to {output_path}")

    plt.show()

    return fig, ax