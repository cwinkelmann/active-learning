import random

import geopandas
import geopandas as gpd
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

from active_learning.config.mapping import get_island_code, drone_mapping
from active_learning.util.rename import get_site_code

## copy the template and the nearby images to a new folder
import shutil
from pathlib import Path
import matplotlib.patches as mpatches

import random

import geopandas
import geopandas as gpd
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

from active_learning.config.mapping import get_island_code, drone_mapping
from active_learning.util.rename import get_site_code

## copy the template and the nearby images to a new folder
import shutil
from pathlib import Path
import matplotlib.patches as mpatches

def visualise_flight_path(date: str,
                          site: str,
                          site_code: str,
                          group_data: gpd.GeoDataFrame):
    """
    Create a beautifully styled visualization of drone flight paths with distance annotations.

    Parameters:
    -----------
    date : str
        The date of the flight
    site : str
        The name of the site
    site_code : str
        The site code
    group_data : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with geometry points
    """
    assert 'distance_to_prev' in group_data.columns

    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=120, facecolor='#fafafa')

    # Extract coordinates for line plotting
    x = [point.x for point in group_data.geometry]
    y = [point.y for point in group_data.geometry]

    # Create a continuous colormap for the line based on height if available
    if 'height' in group_data.columns:
        color_values = group_data['height']
        color_label = 'Height (m)'
    elif 'RelativeAltitude' in group_data.columns:
        color_values = group_data['RelativeAltitude']
        color_label = 'Relative Altitude (m)'
    else:
        color_values = group_data['distance_to_prev']
        color_label = 'Distance to Previous Point (m)'

    # Create segments for colored line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Use a nicer colormap
    cmap = plt.cm.viridis

    # Create a LineCollection with colormap
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(color_values.min(), color_values.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.7, zorder=1)
    lc.set_array(color_values)
    line = ax.add_collection(lc)

    # Add scatter points at regular intervals
    n_points = min(30, len(group_data))
    indices = np.linspace(0, len(group_data) - 1, n_points, dtype=int)

    scatter = ax.scatter(
        [x[i] for i in indices],
        [y[i] for i in indices],
        c=[color_values.iloc[i] for i in indices],
        cmap=cmap,
        s=80,
        alpha=0.8,
        edgecolor='white',
        linewidth=1.5,
        zorder=2
    )

    # Add colorbar with improved styling
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046, aspect=30)
    cbar.set_label(color_label, fontsize=12, color='#555555')
    cbar.ax.tick_params(colors='#666666')

    # Add start and end markers
    ax.scatter(x[0], y[0], s=150, marker='o', color='green', edgecolor='white', linewidth=2, label='Start', zorder=3)
    ax.scatter(x[-1], y[-1], s=150, marker='x', color='red', linewidth=3, label='End', zorder=3)

    # Add direction arrows along the path
    arrow_indices = np.linspace(0, len(x) - 2, 5, dtype=int)
    for i in arrow_indices:
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ax.arrow(x[i], y[i], dx * 0.8, dy * 0.8, head_width=max(abs(dx), abs(dy)) * 0.05,
                 head_length=max(abs(dx), abs(dy)) * 0.08, fc='#4a4a4a', ec='#4a4a4a', zorder=4)

    # Add distance labels every nth point to avoid clutter
    n = max(1, len(group_data) // 8)
    for i in range(0, len(group_data), n):
        if i > 0:  # Skip the first point (distance is 0)
            dist = group_data.iloc[i]['distance_to_prev']
            cumulative_dist = group_data.iloc[:i + 1]['distance_to_prev'].sum()
            if dist > 0:  # Only label points with positive distance
                ax.annotate(f"{cumulative_dist:.1f}m",
                            (x[i], y[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white",
                                      ec="#666666",
                                      alpha=0.9))

    # Improve aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Add title with flight info
    ax.set_title(f"{site} Flight Path",
                 fontsize=18, fontweight='bold', color='#333333', pad=20)

    # Add subtitle with flight details
    subtitle = f"Site Code: {site_code} • Date: {date}"

    fig.suptitle(f"{site} Altitude Profiles",
                 fontsize=18, fontweight='bold', color='#333333', y=0.98)

    # Add subtitle with flight details
    fig.text(0.5, 0.945, subtitle, ha='center', fontsize=12, color='#666666')


    # Calculate flight statistics
    total_distance = group_data['distance_to_prev'].sum()
    max_distance = group_data['distance_to_prev'].max()
    avg_distance = group_data['distance_to_prev'].mean()
    max_speed = (group_data['distance_to_prev'] /
                 (group_data['timestamp'].diff().dt.total_seconds()
                  if 'timestamp' in group_data.columns else 1)).max()

    # Add flight statistics in a styled box
    stats_text = (f"Total distance: {total_distance:.1f}m\n"
                  f"Avg. distance between points: {avg_distance:.1f}m\n"
                  f"Max. distance between points: {max_distance:.1f}m\n"
                  f"Data points: {len(group_data)}")

    if 'timestamp' in group_data.columns:
        stats_text += f"\nMax. speed: {max_speed:.2f}m/s"

    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="white",
                      ec="#dddddd",
                      alpha=0.9))

    # Add legend for start and end points
    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              facecolor='white', edgecolor='#dddddd')

    # Add north arrow if coordinates are geographic
    if group_data.crs and group_data.crs.is_geographic:
        # North arrow with improved styling
        arrow_x, arrow_y = 0.97, 0.10
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.07),
                    xycoords='axes fraction', textcoords='axes fraction',
                    ha='center', va='center', fontsize=14, fontweight='bold', color='#444444',
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='#444444'))

        # Add a circular background for the north arrow
        circle = plt.Circle((arrow_x, arrow_y - 0.035), 0.025,
                            transform=ax.transAxes, fc='white', ec='#dddddd', zorder=0)
        ax.add_patch(circle)

    # Add scale bar if axes have a consistent scale
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    if 0.9 < x_range / y_range < 1.1:  # Check if scales are similar
        scale_length = x_range * 0.1  # 10% of the x-range
        scale_x = min(x) + x_range * 0.05
        scale_y = min(y) + y_range * 0.05
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
                color='black', linewidth=2)
        ax.text(scale_x + scale_length / 2, scale_y + y_range * 0.01,
                f"{scale_length:.1f}m", ha='center', fontsize=8)


    plt.tight_layout()

    return fig, ax


def visualise_height_profile(date: str,
                             site: str,
                             site_code: str,
                             group_data: gpd.GeoDataFrame):
    """
    Create a beautifully styled visualization of drone height profiles:
    "height", "gps_altitude", "RelativeAltitude" against cumulative flight distance.

    Parameters:
    -----------
    date : str
        The date of the flight
    site : str
        The name of the site
    site_code : str
        The site code
    group_data : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with geometry points
    """
    assert 'distance_to_prev' in group_data.columns

    # Calculate cumulative distance (distance from start point)
    group_data['cumulative_distance'] = group_data['distance_to_prev'].cumsum()

    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a figure with three subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=120, facecolor='#fafafa', sharex=True)

    # List of altitude metrics to plot with custom colors and labels
    altitude_metrics = [
        {'name': 'height', 'color': '#4287f5', 'label': 'Height'},
        {'name': 'RelativeAltitude', 'color': '#42c8f5', 'label': 'Relative Altitude'}
    ]

    # Plot each altitude metric
    for i, metric_info in enumerate(altitude_metrics):
        ax = axes[i]
        metric = metric_info['name']
        color = metric_info['color']
        label = metric_info['label']

        if metric in group_data.columns:
            # Create gradient color for fill
            gradient_color = f"{color}40"  # Adding transparency

            # Get x and y data
            x = group_data['cumulative_distance']
            y = group_data[metric]

            # Plot the height profile with shadow effect
            ax.fill_between(x, 0, y, color=gradient_color, alpha=0.5)

            # Plot line
            line = ax.plot(x, y, color=color, linewidth=2.5, label=label)

            # Add scatter points at regular intervals to avoid overcrowding
            n_points = min(30, len(group_data))
            indices = np.linspace(0, len(group_data) - 1, n_points, dtype=int)
            ax.scatter(x.iloc[indices], y.iloc[indices],
                       color=color, s=40, alpha=0.7, edgecolor='white', linewidth=1)

            # Add statistic annotations in a nicer box
            min_val = y.min()
            max_val = y.max()
            mean_val = y.mean()
            median_val = y.median()

            stats_text = (f"Min: {min_val:.1f}m\n"
                          f"Max: {max_val:.1f}m\n"
                          f"Mean: {mean_val:.1f}m\n"
                          f"Median: {median_val:.1f}m")

            ax.text(0.02, 0.95, stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc="white",
                              ec=color,
                              alpha=0.9))

            # Set title and labels for the subplot
            ax.set_title(f"{label} Profile", fontsize=14, fontweight='bold', color='#333333')
            ax.set_ylabel(f"{label} (m)", fontsize=12, color='#555555')

            # Add styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dddddd')
            ax.spines['bottom'].set_color('#dddddd')
            ax.tick_params(colors='#666666')

            # Add horizontal line for mean with improved styling
            ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

            # Add annotation for mean line
            ax.text(x.max() * 0.98, mean_val * 1.02,
                    f"Mean: {mean_val:.1f}m",
                    va='bottom', ha='right', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white",
                              ec=color,
                              alpha=0.9))

            # Add min and max markers
            min_idx = y.idxmin()
            max_idx = y.idxmax()

            ax.annotate(f"Min: {min_val:.1f}m",
                        xy=(x.loc[min_idx], min_val),
                        xytext=(0, -20),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color=color),
                        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
                        ha='center', va='top', fontsize=9)

            ax.annotate(f"Max: {max_val:.1f}m",
                        xy=(x.loc[max_idx], max_val),
                        xytext=(0, 20),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color=color),
                        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
                        ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, f"{label} data not available",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#999999',
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#dddddd"))

    # Set common x-axis label
    axes[-1].set_xlabel("Distance from Start Point (m)", fontsize=14, color='#555555')

    # Add overall title with flight info in a badge-style element
    fig.suptitle(f"{site} Altitude Profiles",
                 fontsize=18, fontweight='bold', color='#333333', y=0.98)

    # Add subtitle with flight details
    subtitle = f"Site Code: {site_code} • Date: {date}"
    fig.text(0.5, 0.945, subtitle, ha='center', fontsize=12, color='#666666')

    # Add a semi-transparent divider line
    fig.text(0.5, 0.935, "—" * 50, ha='center', fontsize=8, color='#dddddd')

    # Add flight statistics to the figure in a modern badge
    total_distance = group_data['cumulative_distance'].max()
    max_speed = (group_data['distance_to_prev'] /
                 (group_data['timestamp'].diff().dt.total_seconds()
                  if 'timestamp' in group_data.columns else 1)).max()

    flight_stats = (f"Total Distance: {total_distance:.1f}m  •  "
                    f"Data Points: {len(group_data)}  •  "
                    f"{'Max Speed: ' + f'{max_speed:.2f}m/s' if 'timestamp' in group_data.columns else ''}")

    fig.text(0.5, 0.02, flight_stats, ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#dddddd", alpha=0.9))


    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25)

    return fig, axes



def visualise_flights(date: str, site: str, site_code: str, group_data: gpd.GeoDataFrame, figure_path = None):
    """
    Visualize flight paths and altitude profiles for a given date and site.
    :param date:
    :param site:
    :param site_code:
    :param group_data:
    :param figure_path:
    :return:
    """
    logger.info(f"Visualising flight paths for {site} at {date}")
    logger.warning(f"Deprecated, use visualise_flights_speed_multiple")


    fig, ax = visualise_height_profile(date, site, site_code, group_data)
    if figure_path:
        fig.savefig(figure_path / f"{site_code}_{date}_altitude_profile.png")
    plt.show()

    fig, ax = visualise_flight_path(date, site, site_code, group_data)
    if figure_path:
        fig.savefig(figure_path / f"{site_code}_{date}_flight_path.png")
    plt.show()

    pass

def visualise_flights_speed_multiple(fcd_entries, date=None, site=None,
                                     smoothing_window=5,
                                     column='speed_m_per_s',
                                     unit='m/s',
                                     title='Flight Height Comparison',
                                     y_text = 'Flight Height (m)'):
    """
    Create a beautifully styled visualization of flight speeds for multiple site codes in one diagram.

    Parameters:
    -----------
    fcd_entries : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with geometry points for multiple flights
    date : str, optional
        Filter by specific date
    site : str, optional
        Filter by specific site name
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=120, facecolor='#fafafa')

    # Get unique site codes for color mapping
    mission_folder = fcd_entries['mission_folder'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission_folder)))
    color_map = dict(zip(mission_folder, colors))

    # Define line styles for additional visual distinction
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    line_style_map = dict(zip(mission_folder, [line_styles[i % len(line_styles)] for i in range(len(mission_folder))]))

    # Define markers for scatter points
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', '<', '>']
    marker_map = dict(zip(mission_folder, [markers[i % len(markers)] for i in range(len(mission_folder))]))

    # Calculate cumulative distance for each flight and plot
    legend_handles = []

    # Loop through each flight group
    for (current_date, site_code, mission_folder), group_data in fcd_entries.groupby(
            ['YYYYMMDD', 'site_code', 'mission_folder']):

        # Calculate cumulative distance
        group_data = group_data.sort_index()
        group_data['cumulative_distance'] = group_data['distance_to_prev'].cumsum()

        group_data[f'{column}_smooth'] = group_data[column].rolling(window=smoothing_window, min_periods=1).mean()

        # Plot the speed profile
        line = ax.plot(
            group_data['cumulative_distance'],
            group_data[f'{column}_smooth'],
            color=color_map[mission_folder],
            linewidth=2.5,
            alpha=0.8,
            label=f"{site_code} ({current_date})"
        )

        # Add scatter points at regular intervals
        n_points = min(15, len(group_data))
        indices = np.linspace(0, len(group_data) - 1, n_points, dtype=int)
        ax.scatter(
            group_data['cumulative_distance'].iloc[indices],
            group_data[f'{column}_smooth'].iloc[indices],
            color=color_map[mission_folder],
            s=60,
            alpha=0.7,
            edgecolor='white',
            linewidth=1
        )

        # Find and mark max speed point
        maximum_value_idx = group_data[f'{column}_smooth'].idxmax()
        max_value = group_data.loc[maximum_value_idx, f'{column}_smooth']
        max_dist = group_data.loc[maximum_value_idx, 'cumulative_distance']

        ax.scatter(
            max_dist, max_value,
            color=color_map[mission_folder],
            s=100,
            marker='*',
            edgecolor='white',
            linewidth=1.5,
            zorder=10
        )

        # Add annotation for max speed
        ax.annotate(
            f"Max: {max_value:.2f} {unit}",
            xy=(max_dist, max_value),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color_map[mission_folder], alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map[mission_folder], alpha=0.9),
            fontsize=8
        )

        # Add flight stats to the legend
        avg_value = group_data[f'{column}_smooth'].mean()
        total_dist = group_data['cumulative_distance'].max()
        legend_handles.append(mpatches.Patch(
            color=color_map[mission_folder],
            label=f"{mission_folder}: Avg {avg_value:.2f} {unit}, Dist {total_dist:.1f}m"
        ))

    # Set labels and title
    ax.set_xlabel("Distance Along Flight Path (m)", fontsize=12, color='#555555')
    ax.set_ylabel(y_text, fontsize=12, color='#555555')

    # Main title
    if date:
        title += f" - Date: {date}"
    if site:
        title += f" - Site: {site}"

    ax.set_title(title, fontsize=18, fontweight='bold', color='#333333', pad=20)

    # Improve aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Add custom legend with flight statistics
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#dddddd',
        fontsize=10
    )

    # Add overall statistics
    all_flights_stats = (
        f"Total Flights: {len(fcd_entries.groupby(['YYYYMMDD', 'flight_code', 'site_code']))}\n"
        f"Total Missions: {len(mission_folder)}"
    )

    ax.text(
        0.02, 0.02,
        all_flights_stats,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.9)
    )

    # Add watermark
    fig.text(
        0.99, 0.01,
        "Iguanas From Above",
        ha='right',
        va='bottom',
        fontsize=8,
        color='#cccccc',
        style='italic'
    )

    plt.tight_layout()

    return fig, ax


def visualise_flights_speed_multiple_2(fcd_entries, date=None, site=None,
                                     smoothing_window=5,
                                     column='speed_m_per_s',
                                     unit='m/s',
                                     title='Flight Height Comparison',
                                     y_text='Flight Height (m)'):
    """
    Create a beautifully styled visualization of flight speeds for multiple site codes in one diagram.

    Parameters:
    -----------
    fcd_entries : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with geometry points for multiple flights
    date : str, optional
        Filter by specific date
    site : str, optional
        Filter by specific site name
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=120, facecolor='#fafafa')

    # Get unique mission folders for visual encoding
    mission_folder = fcd_entries['mission_folder'].unique()

    # Define colors from a perceptually distinct colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission_folder)))
    color_map = dict(zip(mission_folder, colors))

    # Define line styles for additional visual distinction
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    line_style_map = dict(zip(mission_folder, [line_styles[i % len(line_styles)] for i in range(len(mission_folder))]))

    # Define markers for scatter points
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', '<', '>']
    marker_map = dict(zip(mission_folder, [markers[i % len(markers)] for i in range(len(mission_folder))]))

    # Calculate cumulative distance for each flight and plot
    legend_handles = []

    # Loop through each flight group
    for (current_date, site_code, current_mission_folder), group_data in fcd_entries.groupby(
            ['YYYYMMDD', 'site_code', 'mission_folder']):
        # Calculate cumulative distance
        group_data = group_data.sort_index()
        group_data['cumulative_distance'] = group_data['distance_to_prev'].cumsum()

        group_data[f'{column}_smooth'] = group_data[column].rolling(window=smoothing_window, min_periods=1).mean()

        # Plot the speed profile with distinct line style
        line = ax.plot(
            group_data['cumulative_distance'],
            group_data[f'{column}_smooth'],
            color=color_map[current_mission_folder],
            linewidth=2.5,
            alpha=0.8,
            linestyle=line_style_map[current_mission_folder],
            label=f"{site_code} ({current_date})"
        )

        # Add scatter points with distinct markers at regular intervals
        n_points = min(15, len(group_data))
        indices = np.linspace(0, len(group_data) - 1, n_points, dtype=int)
        ax.scatter(
            group_data['cumulative_distance'].iloc[indices],
            group_data[f'{column}_smooth'].iloc[indices],
            color=color_map[current_mission_folder],
            s=60,
            alpha=0.7,
            marker=marker_map[current_mission_folder],
            edgecolor='white',
            linewidth=1
        )

        # Find and mark max speed point
        maximum_value_idx = group_data[f'{column}_smooth'].idxmax()
        max_value = group_data.loc[maximum_value_idx, f'{column}_smooth']
        max_dist = group_data.loc[maximum_value_idx, 'cumulative_distance']

        ax.scatter(
            max_dist, max_value,
            color=color_map[current_mission_folder],
            s=120,
            marker='*',
            edgecolor='white',
            linewidth=1.5,
            zorder=10
        )

        # Add annotation for max speed
        ax.annotate(
            f"Max: {max_value:.2f} {unit}",
            xy=(max_dist, max_value),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color_map[current_mission_folder], alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_map[current_mission_folder], alpha=0.9),
            fontsize=8
        )

        # Add flight stats to the legend with line style indicators
        avg_value = group_data[f'{column}_smooth'].mean()
        total_dist = group_data['cumulative_distance'].max()

        # Create custom patch for legend that shows both color and line style
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=color_map[current_mission_folder],
                             linestyle=line_style_map[current_mission_folder],
                             marker=marker_map[current_mission_folder],
                             markersize=8, linewidth=2.5,
                             markerfacecolor=color_map[current_mission_folder],
                             markeredgecolor='white')

        legend_handles.append((legend_line,
                               f"{current_mission_folder}: Avg {avg_value:.2f} {unit}, Dist {total_dist:.1f}m"))

    # Set labels and title
    ax.set_xlabel("Distance Along Flight Path (m)", fontsize=12, color='#555555')
    ax.set_ylabel(y_text, fontsize=12, color='#555555')

    # Main title
    if date:
        title += f" - Date: {date}"
    if site:
        title += f" - Site: {site}"

    ax.set_title(title, fontsize=18, fontweight='bold', color='#333333', pad=20)

    # Improve aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Add custom legend with flight statistics
    from matplotlib.legend import Legend
    handles, labels = zip(*legend_handles)
    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#dddddd',
        fontsize=10
    )

    # Add overall statistics
    all_flights_stats = (
        f"Total Flights: {len(fcd_entries.groupby(['YYYYMMDD', 'flight_code', 'site_code']))}\n"
        f"Total Missions: {len(mission_folder)}"
    )

    ax.text(
        0.02, 0.02,
        all_flights_stats,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.9)
    )

    # Add watermark
    fig.text(
        0.99, 0.01,
        "Iguanas From Above",
        ha='right',
        va='bottom',
        fontsize=8,
        color='#cccccc',
        style='italic'
    )

    plt.tight_layout()

    return fig, ax


def visualise_drone_model(gdf: gpd.GeoDataFrame,
                          title = "Drone Models Used by",
                          legend_title = "Drone Model",
                          aggregation="day",
                          group_col_metric = 'drone_name', figure_path=None):
    """
    Create a stacked barchart to compare the number of images taken by each drone model.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with drone model information
    aggregation : str, default="day"
        The time unit to aggregate by: "day", "month", "year", or "flight"
    figure_path : str, optional
        Path to save the figure. If None, the figure will only be displayed

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create date column if using time-based aggregation


    # Define grouping column based on aggregation level
    if aggregation == "day":
        group_col = 'YYYYMMDD'
    elif aggregation == "week":
        gdf['week'] = pd.to_datetime(gdf['datetime_digitized']).dt.to_period('W')
        group_col = 'week'
    elif aggregation == "month":
        gdf['month'] = pd.to_datetime(gdf['datetime_digitized']).dt.to_period('M')
        group_col = 'month'
    elif aggregation == "year":
        gdf['year'] = pd.to_datetime(gdf['datetime_digitized']).dt.year
        group_col = 'year'
    elif aggregation == "flight":
        if all(col in gdf.columns for col in ['YYYYMMDD', 'flight_code', 'site_code']):
            gdf['flight_id'] = gdf['YYYYMMDD'] + '_' + gdf['flight_code'] + '_' + gdf['site_code']
            group_col = 'flight_id'
    elif aggregation == "expedition_phase":
        group_col = 'expedition_phase'

    else:
        pass
    # else:
    #     raise ValueError("Aggregation must be one of: 'day', 'month', 'year', or 'flight'")

    # Group by aggregation level and model, then count
    model_counts = gdf.groupby([group_col, group_col_metric]).size().unstack(fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120, facecolor='#fafafa')

    # Plot stacked bar chart
    model_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis',
                      alpha=0.8, edgecolor='white', linewidth=0.5)

    # Improve aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.3, color='#dddddd', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Add title and labels
    ax.set_title(f"{title}",
                 fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel(f"{aggregation.capitalize()}", fontsize=12, color='#555555')
    ax.set_ylabel("Number of Images", fontsize=12, color='#555555')

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add legend with improved styling
    leg = ax.legend(title=legend_title, frameon=True, framealpha=0.9,
                    facecolor='white', edgecolor='#dddddd', fontsize=10)
    leg.get_title().set_fontsize(12)

    # Add totals on top of each bar
    for i, total in enumerate(model_counts.sum(axis=1)):
        ax.text(i, total + (total * 0.02), f"{total}",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

        # Add model percentages as vertical text annotations in the bars
        for i, (_, row) in enumerate(model_counts.iterrows()):
            cumulative = 0
            for model, count in row.items():
                if count > 0:  # Only annotate non-zero values
                    percentage = count / row.sum() * 100
                    if percentage >= 5:  # Only show percentage if it's significant
                        # Place text vertically in the middle of each segment
                        ax.text(i, cumulative + count / 2, f"{percentage:.1f}%",
                                ha='center', va='center', fontsize=8, color='white',
                                fontweight='bold', rotation=90)
                cumulative += count

    # Add statistics text box
    total_images = model_counts.sum().sum()
    unique_models = len(model_counts.columns)
    top_model = model_counts.sum().idxmax()
    top_model_count = model_counts.sum()[top_model]
    top_model_pct = top_model_count / total_images * 100

    stats_text = (f"Total Images: {total_images}\n"
                  f"Unique Values: {unique_models}\n"
                  f"Most Used: {top_model} ({top_model_pct:.1f}%)")

    # Place the stats box in the upper left corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.9))

    plt.tight_layout()

    # Save figure if path is provided
    if figure_path:
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {figure_path}")

    return fig, ax


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


def visualize_height_distribution(gdf: gpd.GeoDataFrame,
                                  title="Drone Height Distribution by Year",
                                  height_col='height',
                                  bins=10,
                                  unit='m',
                                  density=True,
                                  figure_path=None):
    """
    Create a histogram showing the distribution of drone heights by year.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with height information
    title : str
        Title for the plot
    height_col : str
        Column name containing the height data
    bins : int or list
        Number of bins or list of bin edges for the histogram
    density : bool
        If True, the result is normalized to form a probability density
    figure_path : str, optional
        Path to save the figure. If None, the figure will only be displayed

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Ensure datetime column exists
    if 'datetime_digitized' not in gdf.columns and 'datetime_original' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_original'])
    elif 'datetime_digitized' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_digitized'])
    else:
        raise ValueError("No datetime column found in the dataset")

    # Extract year from datetime column
    gdf['year'] = gdf['datetime_digitized'].dt.year

    # Drop rows with missing height data
    gdf_filtered = gdf.dropna(subset=[height_col, 'year'])

    # Get unique years for plotting
    years = sorted(gdf_filtered['year'].unique())

    # Set up a modern aesthetic
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure with FacetGrid for multiple histograms
    if len(years) <= 4:
        # For 4 or fewer years, use a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120, facecolor='#fafafa')
        axes = axes.flatten()
    else:
        # For more years, use a single row with multiple columns
        fig, axes = plt.subplots(len(years) // 3 + (1 if len(years) % 3 > 0 else 0), 3,
                                 figsize=(18, 4 * (len(years) // 3 + (1 if len(years) % 3 > 0 else 0))),
                                 dpi=120, facecolor='#fafafa')
        axes = axes.flatten()

    # Define color palette - a gradient of blues
    blues = sns.color_palette("Blues", len(years))

    # First create one histogram for all years to determine common y-axis limits
    all_counts = []
    all_bins = None

    # Find common bin edges or use specified bins
    if isinstance(bins, int):
        # Automatically determine bin edges based on global min and max
        min_height = gdf_filtered[height_col].min()
        max_height = gdf_filtered[height_col].max()
        all_bins = np.linspace(min_height, max_height, bins + 1)
    else:
        all_bins = bins

    # Create histograms for each year
    for i, year in enumerate(years):
        year_data = gdf_filtered[gdf_filtered['year'] == year][height_col]

        if len(year_data) > 0:
            # Create histogram
            counts, edges = np.histogram(year_data, bins=all_bins, density=density)
            all_counts.append(counts)

    # Determine common y-axis limit
    if len(all_counts) > 0:
        max_count = np.max([np.max(c) for c in all_counts if len(c) > 0]) * 1.1
    else:
        max_count = 1.0

    # Plot individual histograms
    for i, year in enumerate(years):
        if i < len(axes):
            ax = axes[i]
            year_data = gdf_filtered[gdf_filtered['year'] == year][height_col]

            if len(year_data) > 0:
                # Plot histogram with KDE
                sns.histplot(year_data, bins=all_bins, kde=True, color=blues[i],
                             alpha=0.7, edgecolor='white', linewidth=0.5,
                             stat='density' if density else 'count',
                             ax=ax)

                # Add year to title
                ax.set_title(f"{year} (n={len(year_data)})", fontsize=14, color='#333333')

                # Calculate mean and median
                mean_val = year_data.mean()
                median_val = year_data.median()

                # Add vertical lines for mean and median
                ax.axvline(mean_val, color='#ff7f0e', linestyle='--', linewidth=1.5,
                           label=f'Mean: {mean_val:.1f}{unit}')
                ax.axvline(median_val, color='#d62728', linestyle='-', linewidth=1.5,
                           label=f'Median: {median_val:.1f}{unit}')

                # Add legend
                ax.legend(frameon=True, framealpha=0.9, fontsize=10)

                # Improve aesthetics
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, linestyle='--', alpha=0.3, color='#dddddd')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#dddddd')
                ax.spines['bottom'].set_color('#dddddd')
                ax.tick_params(colors='#666666')

                # Set consistent y-axis limits
                ax.set_ylim(0, max_count)

                # Label axes
                ax.set_xlabel(f"{height_col} (meters)", fontsize=12, color='#555555')
                if density:
                    ax.set_ylabel("Density", fontsize=12, color='#555555')
                else:
                    ax.set_ylabel("Count", fontsize=12, color='#555555')

                # Add statistics
                stats_text = (f"Min: {year_data.min():.1f}{unit}\n"
                              f"Max: {year_data.max():.1f}{unit}\n"
                              f"Mean: {mean_val:.1f}{unit}\n"
                              f"Median: {median_val:.1f}{unit}\n"
                              f"Std. Dev: {year_data.std():.1f}{unit}")

                # Place statistics text box
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.9))
            else:
                # No data for this year
                ax.text(0.5, 0.5, f"No data for {year}", ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='#999999',
                        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#dddddd"))
                ax.set_title(f"{year}", fontsize=14, color='#333333')

    # Hide empty subplots
    for i in range(len(years), len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', color='#333333', y=0.98)

    # Add a violin plot at the bottom to compare all years
    if len(years) <= 4:  # Only for 2x2 grid
        if len(axes) > len(years):  # If we have an empty slot
            ax_violin = axes[len(years)]
            ax_violin.set_visible(True)

            # Prepare data for violin plot
            yearly_data = []
            yearly_labels = []

            for year in years:
                year_data = gdf_filtered[gdf_filtered['year'] == year][height_col]
                if len(year_data) > 0:
                    yearly_data.append(year_data.values)
                    yearly_labels.append(str(year))

            if len(yearly_data) > 0:
                # Create violin plot
                violin_parts = ax_violin.violinplot(yearly_data, showmeans=True, showmedians=True, showextrema=True)

                # Customize violin plot
                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(blues[i])
                    pc.set_edgecolor('white')
                    pc.set_alpha(0.7)

                # Customize other elements
                for partname in ['cmeans', 'cmedians', 'cbars']:
                    if partname in violin_parts:
                        violin_parts[partname].set_edgecolor('#555555')
                        violin_parts[partname].set_linewidth(1.5)

                # Set x-ticks to years
                ax_violin.set_xticks(np.arange(1, len(yearly_labels) + 1))
                ax_violin.set_xticklabels(yearly_labels)

                # Set title and labels
                ax_violin.set_title("Year-by-Year Height Comparison", fontsize=14, color='#333333')
                ax_violin.set_xlabel("Year", fontsize=12, color='#555555')
                ax_violin.set_ylabel(f"{height_col} (meters)", fontsize=12, color='#555555')

                # Improve aesthetics
                ax_violin.set_facecolor('#f8f9fa')
                ax_violin.grid(True, linestyle='--', alpha=0.3, color='#dddddd', axis='y')
                ax_violin.spines['top'].set_visible(False)
                ax_violin.spines['right'].set_visible(False)
                ax_violin.spines['left'].set_color('#dddddd')
                ax_violin.spines['bottom'].set_color('#dddddd')
                ax_violin.tick_params(colors='#666666')
            else:
                ax_violin.set_visible(False)


    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save figure if path is provided
    if figure_path:
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {figure_path}")

    return fig, axes


def visualize_height_boxplot(gdf: gpd.GeoDataFrame,
                             title="Drone Height by Year",
                             height_col='height',
                             unit = 'm',
                             figure_path=None):
    """
    Create a boxplot showing the distribution of drone heights by year.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with height information
    title : str
        Title for the plot
    height_col : str
        Column name containing the height data
    figure_path : str, optional
        Path to save the figure. If None, the figure will only be displayed

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Ensure datetime column exists
    if 'datetime_digitized' not in gdf.columns and 'datetime_original' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_original'])
    elif 'datetime_digitized' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_digitized'])
    else:
        raise ValueError("No datetime column found in the dataset")

    # Extract year from datetime column
    gdf['year'] = gdf['datetime_digitized'].dt.year

    # Drop rows with missing height data
    gdf_filtered = gdf.dropna(subset=[height_col, 'year'])

    # Set up a modern aesthetic
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120, facecolor='#fafafa')

    # Create boxplot
    boxplot = sns.boxplot(x='year', y=height_col, data=gdf_filtered,
                          palette='Blues', width=0.5,
                          showfliers=False, ax=ax)

    # Add swarmplot for individual points (with transparency for dense plots)
    swarmplot = sns.swarmplot(x='year', y=height_col, data=gdf_filtered,
                              size=4, color='#333333', alpha=0.5, ax=ax)

    # Improve aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.3, color='#dddddd', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors='#666666')

    # Add title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel("Year", fontsize=12, color='#555555')
    ax.set_ylabel(f"{height_col} (meters)", fontsize=12, color='#555555')

    # Add count annotations
    for i, year in enumerate(sorted(gdf_filtered['year'].unique())):
        count = len(gdf_filtered[gdf_filtered['year'] == year])
        median = gdf_filtered[gdf_filtered['year'] == year][height_col].median()
        ax.annotate(f"n={count}\nmedian={median:.1f}{unit}",
                    xy=(i, gdf_filtered[gdf_filtered['year'] == year][height_col].max()),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.9))

    # Add overall statistics
    stats_text = (f"Total Images: {len(gdf_filtered)}\n"
                  f"Overall Median: {gdf_filtered[height_col].median():.1f}{unit}\n"
                  f"Overall Mean: {gdf_filtered[height_col].mean():.1f}{unit}\n"
                  f"Overall Std Dev: {gdf_filtered[height_col].std():.1f}{unit}")

    # Place statistics text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd", alpha=0.9))


    plt.tight_layout()

    # Save figure if path is provided
    if figure_path:
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {figure_path}")

    return fig, ax


def visualize_height_ridgeplot(gdf: gpd.GeoDataFrame,
                               title="Drone Height Distribution by Year",
                               height_col='height',
                               unit='m',
                               figure_path=None):
    """
    Create a ridge plot showing the distribution of drone heights by year.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing flight data with height information
    title : str
        Title for the plot
    height_col : str
        Column name containing the height data
    figure_path : str, optional
        Path to save the figure. If None, the figure will only be displayed

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    try:
        import joypy
    except ImportError:
        print("Installing joypy package...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joypy"])
        import joypy

    # Ensure datetime column exists
    if 'datetime_digitized' not in gdf.columns and 'datetime_original' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_original'])
    elif 'datetime_digitized' in gdf.columns:
        gdf['datetime_digitized'] = pd.to_datetime(gdf['datetime_digitized'])
    else:
        raise ValueError("No datetime column found in the dataset")

    # Extract year from datetime column
    gdf['year'] = gdf['datetime_digitized'].dt.year

    # Drop rows with missing height data
    gdf_filtered = gdf.dropna(subset=[height_col, 'year'])

    # Convert years to strings for better visualization
    gdf_filtered['year'] = gdf_filtered['year'].astype(str)

    # Sort years in chronological order
    years_sorted = sorted(gdf_filtered['year'].unique())

    # Set up a modern aesthetic
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, axes = joypy.joyplot(
        data=gdf_filtered,
        by='year',
        column=height_col,
        labels=years_sorted,
        range_style='own',
        grid=True,
        linewidth=1,
        legend=True,
        figsize=(12, 8),
        fade=True,
        colormap=LinearSegmentedColormap.from_list("blues",
                                                   sns.color_palette("Blues", len(years_sorted)))
    )

    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#333333', y=0.98)

    # Add annotation with counts for each year
    for i, year in enumerate(years_sorted):
        count = len(gdf_filtered[gdf_filtered['year'] == year])
        mean_val = gdf_filtered[gdf_filtered['year'] == year][height_col].mean()
        median_val = gdf_filtered[gdf_filtered['year'] == year][height_col].median()

        # Add text with count and statistics
        axes[i].text(0.02, 0.8, f"n={count}\nMean={mean_val:.1f}{unit}\nMedian={median_val:.2f}{unit}",
                          transform=axes[i].transAxes, fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.9))

    # Add x-label to the bottom axis
    axes[-1].set_xlabel(f"{height_col} ({unit})", fontsize=12, color='#555555')


    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save figure if path is provided
    if figure_path:
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {figure_path}")

    return fig, axes


# Example usage
if __name__ == "__main__":
    # Example data (replace with your actual data)
    # Create sample data
    np.random.seed(42)
    data = {
        'height': np.concatenate([
            np.random.normal(30, 5, 100),  # 2020 heights
            np.random.normal(35, 4, 150),  # 2021 heights
            np.random.normal(32, 7, 200),  # 2022 heights
            np.random.normal(28, 3, 120)  # 2023 heights
        ]),
        'datetime_original': np.concatenate([
            pd.date_range('2020-01-01', '2020-12-31', periods=100),
            pd.date_range('2021-01-01', '2021-12-31', periods=150),
            pd.date_range('2022-01-01', '2022-12-31', periods=200),
            pd.date_range('2023-01-01', '2023-12-31', periods=120)
        ])
    }

    # Create GeoDataFrame (without actual geometries for this example)
    gdf = gpd.GeoDataFrame(data)

    # Visualize height distribution using histograms
    fig1, ax1 = visualize_height_distribution(gdf, title="Sample Drone Height Distribution by Year")

    # Visualize height distribution using boxplot
    fig2, ax2 = visualize_height_boxplot(gdf, title="Sample Drone Height by Year")

    # Visualize height distribution using ridge plot
    fig3, ax3 = visualize_height_ridgeplot(gdf, title="Sample Drone Height Distribution by Year")

    plt.show()