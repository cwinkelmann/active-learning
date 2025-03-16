import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from shapely.geometry import Point

from active_learning.util.drone_flight_check import get_flight_metrics

# Create a synthetic GeoDataFrame
# Let's create a sequence of points that form a simple grid-like flight pattern

# Starting point (arbitrary coordinates)
start_lat, start_lon = -0.3019, -91.6501  # Cabo Douglas, Fernandina, Galapagos, Ecuador

# Create timestamps at regular intervals
start_time = datetime(2023, 5, 15, 10, 0, 0)
time_interval = 5  # seconds between shots

@pytest.fixture
def gdf() -> gpd.GeoDataFrame:
    """
    Creates a synthetic GeoDataFrame with drone flight data and tests the get_flight_metrics function.
    The test validates that calculations are performed correctly and output values are within expected ranges.
    """


    # Create flight pattern
    points = []
    times = []

    # First leg: East direction
    degree_step = 0.0005
    for i in range(5):
        points.append(Point(start_lon + i * degree_step, start_lat))
        times.append(start_time + timedelta(seconds=i * time_interval))

    # Turn south
    current_lon = start_lon + 4 * degree_step
    current_lat = start_lat

    # Second leg: South direction
    for i in range(1, 5):
        points.append(Point(current_lon, current_lat - i * degree_step))
        times.append(start_time + timedelta(seconds=(4 + i) * time_interval))

    # Turn west
    current_lon = start_lon + 4 * degree_step
    current_lat = start_lat - 4 * degree_step

    # Third leg: West direction
    for i in range(1, 5):
        points.append(Point(current_lon - i * degree_step, current_lat))
        times.append(start_time + timedelta(seconds=(8 + i) * time_interval))

    # Create dataframe
    data = {
        'geometry': points,
        'datetime_digitized': times,
        'exposure_time': [1 / 500] * len(points),  # 1/500 second exposure time
        'gsd_rel_avg_cm': [2.5] * len(points),  # 2.5 cm/pixel GSD
        'image_width': [5472] * len(points),  # pixels
        'image_height': [3648] * len(points)  # pixels
    }

    gdf = gpd.GeoDataFrame(data, crs={'init': 'epsg:4326'}, geometry='geometry')
    gdf.to_crs(epsg="32715", inplace=True)
    return gdf





def test_get_flight_metrics(gdf):
    """
    Tests the get_flight_metrics function using pytest assertions.

    Args:
        gdf: A GeoDataFrame with drone flight data
    """
    # Run the get_flight_metrics function on the data
    result_gdf = get_flight_metrics(gdf)

    # Get the time interval from the data (assuming 5 seconds between shots)
    time_interval = (gdf['datetime_digitized'].iloc[1] - gdf['datetime_digitized'].iloc[0]).total_seconds()

    # Expected approximate distance between points (0.0005 degrees at latitude 34)
    expected_distance = 55  # meters

    # 1. Check that the first row has NaN for distance, time diff, and speed
    assert pd.isna(result_gdf.iloc[0]['distance_to_prev']), "First row distance should be NaN"
    assert pd.isna(result_gdf.iloc[0]['time_diff_seconds']), "First row time_diff should be NaN"
    assert pd.isna(result_gdf.iloc[0]['speed_m_per_s']), "First row speed should be NaN"

    # 2. Check that time differences match our input interval
    avg_time_diff = result_gdf['time_diff_seconds'].dropna().mean()
    assert np.isclose(avg_time_diff, time_interval, rtol=0.01), \
        f"Time difference average {avg_time_diff:.2f}s doesn't match expected {time_interval}s"

    # 3. Check distances
    avg_distance = result_gdf['distance_to_prev'].dropna().mean()
    assert np.isclose(avg_distance, expected_distance, rtol=0.2), \
        f"Average distance {avg_distance:.2f}m doesn't match expected {expected_distance}m"

    # 4. Check speed calculations
    expected_speed = expected_distance / time_interval
    avg_speed = result_gdf['speed_m_per_s'].dropna().mean()
    assert np.isclose(avg_speed, expected_speed, rtol=0.2), \
        f"Average speed {avg_speed:.2f}m/s doesn't match expected {expected_speed:.2f}m/s"

    # 5. Check shift calculations
    expected_shift_mm = expected_speed * (1 / 500) * 1000  # speed * exposure_time * 1000
    avg_shift = result_gdf['shift_mm'].dropna().mean()
    assert np.isclose(avg_shift, expected_shift_mm, rtol=0.2), \
        f"Average shift {avg_shift:.2f}mm doesn't match expected {expected_shift_mm:.2f}mm"

    # 6. Check ground footprint calculations
    expected_ground_width = 5472 * 2.5 / 100  # image_width * gsd in cm / 100 -> meters
    expected_ground_height = 3648 * 2.5 / 100  # image_height * gsd in cm / 100 -> meters

    avg_ground_width = result_gdf['ground_width_m'].mean()
    avg_ground_height = result_gdf['ground_height_m'].mean()

    assert np.isclose(avg_ground_width, expected_ground_width, rtol=0.01), \
        f"Ground width {avg_ground_width:.2f}m doesn't match expected {expected_ground_width:.2f}m"
    assert np.isclose(avg_ground_height, expected_ground_height, rtol=0.01), \
        f"Ground height {avg_ground_height:.2f}m doesn't match expected {expected_ground_height:.2f}m"

    # 7. Check bearing and direction assignments
    direction_counts = result_gdf['flight_direction'].value_counts()
    expected_directions = {'E', 'S'}
    detected_directions = set(direction_counts.index.dropna())

    assert expected_directions.issubset(detected_directions), \
        f"Expected directions {expected_directions} not found in {detected_directions}"

    # 8. Check overlap calculations
    # Expected overlap based on our synthetic data
    if 'E' in direction_counts or 'W' in direction_counts:
        ew_overlap = result_gdf[result_gdf['flight_direction'].isin(['E', 'W'])]['forward_overlap_pct'].dropna().mean()
        ew_expected = 67.0  # For E-W: (136.8-45)/136.8 * 100 = ~67%

        assert np.isclose(ew_overlap, ew_expected, rtol=0.2), \
            f"E-W overlap {ew_overlap:.2f}% doesn't match expected {ew_expected:.2f}%"

    if 'N' in direction_counts or 'S' in direction_counts:
        ns_overlap = result_gdf[result_gdf['flight_direction'].isin(['N', 'S'])]['forward_overlap_pct'].dropna().mean()
        ns_expected = 40.0  # For N-S: (91.2-55)/91.2 * 100 = ~40%

        assert np.isclose(ns_overlap, ns_expected, rtol=0.2), \
            f"N-S overlap {ns_overlap:.2f}% doesn't match expected {ns_expected:.2f}%"
