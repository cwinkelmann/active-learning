import geopandas as gpd
import pandas as pd
import numpy as np

from active_learning.config.mapping import get_island_code, drone_mapping
from active_learning.util.rename import get_site_code


def get_analysis_ready_image_metadata(gdf_all: gpd.GeoDataFrame):
    """

    """
    try:
        # drop
        gdf_all.drop(columns=['Unnamed: 0'], inplace=True)
    except KeyError:
        pass

    # gdf_all.to_parquet("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.parquet", compression="snappy")
    # use the mapping
    island_code_mapping = get_island_code()
    gdf_all["island_code"] = gdf_all["island"].apply(
        lambda x: island_code_mapping.get(x, x) if isinstance(x, str) else x)

    gdf_all["site_code"] = gdf_all["folder_name"].apply(lambda x: get_site_code(x))
    # take the image name and split it by the underscore
    # Extract everything before the first underscore from folder_name to create site_code
    gdf_all["flight_code"] = gdf_all["folder_name"].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else x)

    gdf_all['YYYYMMDD'] = gdf_all['datetime_digitized'].dt.strftime('%Y-%m-%d')
    gdf_all["drone_name"] = gdf_all["body_serial_number"].apply(lambda x: drone_mapping.get(x, "falcon"))

    # Use absolute height value
    gdf_all['gsd_abs_width_cm'] = gdf_all["height"].apply(
        lambda h: calculate_gsd(h)[0] if pd.notna(h) else None
    )
    gdf_all['gsd_abs_height_cm'] = gdf_all["height"].apply(
        lambda h: calculate_gsd(h)[1] if pd.notna(h) else None
    )
    gdf_all['gsd_abs_avg_cm'] = gdf_all["height"].apply(
        lambda h: calculate_gsd(h)[2] if pd.notna(h) else None
    )

    # Use absolute height value
    gdf_all['gsd_rel_width_cm'] = gdf_all["RelativeAltitude"].apply(
        lambda h: calculate_gsd(h)[0] if pd.notna(h) else None
    )
    gdf_all['gsd_rel_height_cm'] = gdf_all["RelativeAltitude"].apply(
        lambda h: calculate_gsd(h)[1] if pd.notna(h) else None
    )
    gdf_all['gsd_rel_avg_cm'] = gdf_all["RelativeAltitude"].apply(
        lambda h: calculate_gsd(h)[2] if pd.notna(h) else None
    )


    return gdf_all


def calculate_bearing(point1, point2):
    """Calculate the bearing between two points in degrees"""
    if point1 is None or point2 is None:
        return None

    # Extract coordinates
    lon1, lat1 = point1.x, point1.y
    lon2, lat2 = point2.x, point2.y

    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Calculate bearing
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))

    # Normalize to 0-360
    bearing = (bearing + 360) % 360

    return bearing


def classify_bearing(bearing):
    """Classify bearing into cardinal directions"""
    if bearing is None:
        return None

    # Define direction ranges
    if (bearing >= 337.5 or bearing < 22.5):
        return "N"
    elif (bearing >= 22.5 and bearing < 67.5):
        return "NE"
    elif (bearing >= 67.5 and bearing < 112.5):
        return "E"
    elif (bearing >= 112.5 and bearing < 157.5):
        return "SE"
    elif (bearing >= 157.5 and bearing < 202.5):
        return "S"
    elif (bearing >= 202.5 and bearing < 247.5):
        return "SW"
    elif (bearing >= 247.5 and bearing < 292.5):
        return "W"
    elif (bearing >= 292.5 and bearing < 337.5):
        return "NW"
    else:
        return None


def calculate_forward_overlap(distance, ground_width, ground_height, direction):
    """
    Calculate forward overlap percentage based on distance and image footprint
    """
    if distance is None or ground_width is None or ground_height is None or direction is None:
        return None

    # For predominantly North-South flights
    if direction in ["N", "S", "NE", "SE", "NW", "SW"]:
        # Assuming flight direction is along the height of the image
        overlap_distance = ground_height - distance
        overlap_percentage = (overlap_distance / ground_height) * 100
    # For predominantly East-West flights
    else:  # direction in ["E", "W"]
        # Assuming flight direction is along the width of the image
        overlap_distance = ground_width - distance
        overlap_percentage = (overlap_distance / ground_width) * 100

    # Ensure we don't return negative overlap
    return max(0, overlap_percentage)


def get_flight_metrics(gdf_all: gpd.GeoDataFrame, gsd_col="gsd_rel_avg_cm"):
    """
    Calculate flight metrics such as distance, time difference, speed, overlap, and risk score.
    :param gdf_all:
    :param gsd_col:
    :return:
    """
    gdf_all = gdf_all.sort_values(by=['datetime_digitized'])
    # Reset index to ensure operations work on sorted data
    gdf_all = gdf_all.reset_index(drop=True)

    # Assuming your dataframe is named gdf_all and is already sorted by datetime_digitized
    # Let's create columns for distance, time difference, and speed

    # First, create a shifted version of geometry and datetime columns to get "previous" values
    gdf_all['prev_geometry'] = gdf_all['geometry'].shift(1)
    gdf_all['prev_datetime'] = gdf_all['datetime_digitized'].shift(1)

    # Calculate distance in meters between consecutive points
    gdf_all['distance_to_prev'] = gdf_all.apply(
        lambda row: row['geometry'].distance(row['prev_geometry'])
        if row['prev_geometry'] is not None else np.nan,
        axis=1
    )

    # Calculate time difference in seconds between consecutive shots
    gdf_all['time_diff_seconds'] = gdf_all.apply(
        lambda row: (row['datetime_digitized'] - row['prev_datetime']).total_seconds()
        if pd.notnull(row['prev_datetime']) else np.nan,
        axis=1
    )

    # Calculate speed in meters per second
    gdf_all['speed_m_per_s'] = gdf_all['distance_to_prev'] / gdf_all['time_diff_seconds']
    gdf_all['speed_m_per_s'] = pd.to_numeric(gdf_all['speed_m_per_s'], errors='coerce')

    # Handle first row (no previous point)
    gdf_all.loc[gdf_all.index[0], ['distance_to_prev', 'time_diff_seconds', 'speed_m_per_s']] = np.nan

    gdf_all['risk_score'] = gdf_all['speed_m_per_s'] * gdf_all['exposure_time'] * 1500

    # Calculate shift in millimeters of how much the drone moved during the shot
    gdf_all['shift_mm'] = gdf_all.apply(
        lambda row: row['speed_m_per_s'] * row['exposure_time'] * 1000 # * 1000 because of meter to millimeter
        if pd.notna(row['speed_m_per_s']) and pd.notna(row['exposure_time'])
        else np.nan,
        axis=1
    )

    gdf_all['shift_pixels'] = gdf_all.apply(
        lambda row: row['shift_mm'] / (row[gsd_col] * 10)
        if pd.notna(row['shift_mm']) and pd.notna(row[gsd_col]) and row[gsd_col] > 0
        else np.nan,
        axis=1
    )

    # Calculate image overlap based on GSD and distances
    # Calculate the ground footprint dimensions in meters for each image
    gdf_all['ground_width_m'] = gdf_all.apply(
        lambda row: (row["image_width"] * row[gsd_col]) / 100  # Convert cm to m
        if pd.notna(row[gsd_col]) and row[gsd_col] > 0
        else np.nan,
        axis=1
    )

    gdf_all['ground_height_m'] = gdf_all.apply(
        lambda row: (row["image_height"] * row[gsd_col]) / 100  # Convert cm to m
        if pd.notna(row[gsd_col]) and row[gsd_col] > 0
        else np.nan,
        axis=1
    )

    # To determine overlap, we need to know the flight direction
    # Let's calculate the bearing between consecutive points
    gdf_all['bearing_to_prev'] = gdf_all.apply(
        lambda row: calculate_bearing(row['prev_geometry'], row['geometry'])
        if row['prev_geometry'] is not None else np.nan,
        axis=1
    )

    # Classify the bearing into approximate flight directions (N-S, E-W, etc.)
    gdf_all['flight_direction'] = gdf_all['bearing_to_prev'].apply(
        lambda bearing: classify_bearing(bearing) if pd.notna(bearing) else None
    )

    # Calculate overlap of each image in percentages
    # For simplicity, we'll use a direct approach based on distance and image footprint
    gdf_all['forward_overlap_pct'] = gdf_all.apply(
        lambda row: calculate_forward_overlap(
            row['distance_to_prev'],
            row['ground_width_m'],
            row['ground_height_m'],
            row['flight_direction']
        ) if pd.notna(row['distance_to_prev']) and pd.notna(row['ground_width_m'])
        else np.nan,
        axis=1
    )

    # Is the flight a oblique flight or a nadir flight
    gdf_all['is_oblique'] = gdf_all['GimbalPitchDegree'].apply(
        lambda pitch: pitch >= -75
    )
    gdf_all['is_nadir'] = gdf_all['GimbalPitchDegree'].apply(
        lambda pitch: pitch < -75
    )

    # Clean up by dropping intermediate columns
    gdf_all = gdf_all.drop(columns=['prev_geometry', 'prev_datetime'])

    return gdf_all


def calculate_gsd(height,
                  sensor_width=13.2,  # Mavic 2 Pro sensor width in mm
                  sensor_height=8.8,  # Mavic 2 Pro sensor height in mm
                  focal_length=10.26,    # Actual focal length in mm (not 35mm equivalent)
                  image_width=5472,  # Image width in pixels
                  image_height=3648):  # Image height in pixels
    """
    Calculate Ground Sampling Distance (GSD) for Mavic 2 Pro drone.

    Parameters:
    -----------
    height : float
        Flying height above ground level in meters
    sensor_width : float
        Camera sensor width in millimeters (13.2mm for Mavic 2 Pro)
    sensor_height : float
        Camera sensor height in millimeters (8.8mm for Mavic 2 Pro)
    focal_length : float
        Camera focal length in millimeters (28mm for Mavic 2 Pro Hasselblad)
    image_width : int
        Image width in pixels (5472 for Mavic 2 Pro)
    image_height : int
        Image height in pixels (3648 for Mavic 2 Pro)

    Returns:
    --------
    tuple
        (GSD_width, GSD_height, GSD_average) in cm/pixel
    """
    # Calculate GSD in cm/pixel
    gsd_width = (sensor_width * height * 100) / (focal_length * image_width)
    gsd_height = (sensor_height * height * 100) / (focal_length * image_height)
    gsd_average = (gsd_width + gsd_height) / 2

    return gsd_width, gsd_height, gsd_average


def find_anomalous_images(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Find images taken at a speed of 5 m/s or more with a shutter speed of 1/1500 or slower
    """

    # replace NaN with 0
    df['exposure_time'] = df['exposure_time'].fillna(0)
    df['speed_m_per_s'] = df['speed_m_per_s'].fillna(0)
    df['time_diff_seconds'] = df['time_diff_seconds'].fillna(0)

    # Convert exposure_time to numeric, handling any non-numeric values
    df['exposure_time'] = pd.to_numeric(df['exposure_time'], errors='coerce')

    # Convert speed_m_per_s to numeric, handling any non-numeric values
    df['speed_m_per_s'] = pd.to_numeric(df['speed_m_per_s'], errors='coerce')

    # Filter images where:
    # 1. Speed is 5 m/s or more
    # 2. Exposure time is 1/1500 seconds (0.000667) or longer
    # Note: Longer exposure time = slower shutter speed
    anomalous_images = df[(df['speed_m_per_s'] >= 6.0) & (df['exposure_time'] >= 1 / 1500) ]

    # Add a 'risk_score' column - higher values mean higher risk of motion blur
    anomalous_images = anomalous_images.sort_values('risk_score', ascending=False)

    # Print summary
    print(f"Found {len(anomalous_images)} images with potentially problematic settings")
    print(f"These images were taken at high speed (≥5 m/s) with slow shutter (≤1/1500s)")
    print("They may exhibit motion blur due to these conditions")

    # Select relevant columns for display
    display_columns = [
        'image_name', 'mission_folder', 'exposure_time', 'speed_m_per_s',
        'height', 'RelativeAltitude',
        'photographic_sensitivity', 'datetime_original',
        'gsd_rel_avg_cm', 'gsd_abs_avg_cm',
        'risk_score'
    ]

    # Filter to only include columns that exist in the dataframe
    existing_columns = [col for col in display_columns if col in anomalous_images.columns]

    return anomalous_images[existing_columns]

def get_near_images(gdf, grouping_column, template_image, difference="photographic_sensitivity", n=4, max_distance=20):
    """
    Find up to n images within max_distance meters from a template image
    that have the most different values in the specified column and come from different groups.

    Parameters:
    -----------
    gdf : GeoDataFrame
        The GeoDataFrame containing the image data
    grouping_column : str
        Column name to ensure selected images come from different groups
    template_image : GeoSeries or GeoDataFrame row
        The template image to compare against
    difference : str
        The column name to check for different values
    n : int
        Maximum number of other images to return
    max_distance : float
        Maximum distance in meters between images

    Returns:
    --------
    tuple
        (template_image, GeoDataFrame of nearby images with most different values)
    """
    template_value = template_image[difference]
    nearby_different = []

    # Get all unique groups except the template's group
    template_group = template_image[grouping_column]
    other_groups = gdf[gdf[grouping_column] != template_group][grouping_column].unique()

    # For each group, find nearby images first, then select the one with the most different value
    for group in other_groups[:n]:  # Limit to n groups
        group_df = gdf[gdf[grouping_column] == group]

        # First, filter by geographic distance
        nearby_in_group = []
        for idx, row in group_df.iterrows():
            dist = template_image.geometry.distance(row.geometry)
            if dist <= max_distance:
                nearby_in_group.append((idx, row))

        if not nearby_in_group:
            continue  # No nearby images in this group

        # From these nearby candidates, find the one with the most different value
        max_diff = -1
        best_idx = None

        for idx, row in nearby_in_group:
            # Skip if same value as template
            if row[difference] == template_value:
                continue

            value_diff = abs(float(row[difference]) - float(template_value))
            if value_diff > max_diff:
                max_diff = value_diff
                best_idx = idx

        # If we found a suitable image with a different value
        if best_idx is not None:
            nearby_different.append(gdf.loc[best_idx])

        # Break if we've reached n images
        if len(nearby_different) >= n:
            break

    return template_image, gpd.GeoDataFrame(nearby_different, crs=gdf.crs)