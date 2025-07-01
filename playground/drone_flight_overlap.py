#!/usr/bin/env python3
"""
Drone Forward Overlap Calculator

Calculates forward overlap percentages for drone photogrammetry missions
based on flight speed, image capture intervals, Ground Sampling Distance and sensor specifications.
"""

import numpy as np
import pandas as pd


def calculate_forward_overlap(sensor_forward_pixels, gsd_meters_per_pixel,
                              flight_speeds, capture_intervals):
    """
    Calculate forward overlap percentages for drone photogrammetry.

    Parameters:
    -----------
    sensor_forward_pixels : int
        Number of pixels in the forward direction of flight
    gsd_meters_per_pixel : float
        Ground Sample Distance in meters per pixel
    flight_speeds : list
        List of flight speeds in m/s
    capture_intervals : list
        List of capture intervals in seconds

    Returns:
    --------
    pandas.DataFrame
        Table with forward overlap percentages
    """

    # Calculate ground coverage in forward direction
    ground_coverage_forward = sensor_forward_pixels * gsd_meters_per_pixel

    print(f"Sensor specifications:")
    print(f"Forward direction: {sensor_forward_pixels} pixels")
    print(f"GSD: {gsd_meters_per_pixel} m/px ({gsd_meters_per_pixel * 100:.1f} cm/px)")
    print(f"Forward ground coverage: {ground_coverage_forward:.3f} meters")
    print()

    # Initialize results dictionary
    results = {}

    # Calculate overlap for each speed
    for speed in flight_speeds:
        overlaps = []

        for interval in capture_intervals:
            # Distance traveled between shots
            distance_traveled = speed * interval

            # Convert distance to pixels
            pixels_moved = distance_traveled / gsd_meters_per_pixel

            # Calculate overlap
            overlap_pixels = max(0, sensor_forward_pixels - pixels_moved)
            overlap_percentage = (overlap_pixels / sensor_forward_pixels) * 100

            overlaps.append(overlap_percentage)

        results[f"{speed} m/s"] = overlaps

    # Create DataFrame
    df = pd.DataFrame(results, index=[f"{interval}s" for interval in capture_intervals])

    return df


def print_overlap_table(df):
    """Print the overlap table in a formatted way."""
    print("Forward Overlap Percentages:")
    print("=" * 60)

    # Print header
    header = "Interval\\Speed\t" + "\t".join(df.columns)
    print(header)
    print("-" * 60)

    # Print data rows
    for interval in df.index:
        row = f"{interval}\t\t"
        row += "\t".join([f"{val:.1f}%" for val in df.loc[interval]])
        print(row)
    print()


def example_calculation(sensor_forward_pixels, gsd_meters_per_pixel,
                        speed, interval):
    """Show detailed calculation for one example."""
    print(f"Example Calculation:")
    print(f"Speed: {speed} m/s, Interval: {interval}s")
    print("-" * 40)

    distance_traveled = speed * interval
    pixels_moved = distance_traveled / gsd_meters_per_pixel
    overlap_pixels = sensor_forward_pixels - pixels_moved
    overlap_percentage = (overlap_pixels / sensor_forward_pixels) * 100

    print(f"Distance traveled: {distance_traveled} meters")
    print(f"Pixels moved: {pixels_moved:.1f} pixels")
    print(f"Sensor forward dimension: {sensor_forward_pixels} pixels")
    print(f"Overlap pixels: {overlap_pixels:.1f} pixels")
    print(f"Forward overlap: {overlap_percentage:.1f}%")
    print()


def main():
    """Main function to run the calculations."""

    # Define parameters
    # Assuming 5472×3648 sensor with 3648 pixels in forward direction
    SENSOR_FORWARD_PIXELS = 3648  # Forward direction (short edge)
    GSD_METERS_PER_PIXEL = 0.007  # 0.7 cm/px

    # Flight parameters to test
    FLIGHT_SPEEDS = [2, 3, 4, 5, 6, 7]  # m/s
    CAPTURE_INTERVALS = [1, 2, 3, 4, 5]  # seconds

    print("Drone Forward Overlap Calculator")
    print("================================")
    print()

    # Show example calculation
    example_calculation(SENSOR_FORWARD_PIXELS, GSD_METERS_PER_PIXEL, 2, 2)

    # Calculate overlap table
    overlap_df = calculate_forward_overlap(
        SENSOR_FORWARD_PIXELS,
        GSD_METERS_PER_PIXEL,
        FLIGHT_SPEEDS,
        CAPTURE_INTERVALS
    )

    # Print formatted table
    print_overlap_table(overlap_df)

    # Also print as pandas DataFrame for easy copying
    print("Pandas DataFrame format:")
    print(overlap_df.round(1))
    print()

    # Save to CSV if needed
    # overlap_df.round(1).to_csv('drone_overlap_table.csv')
    # print("Table saved to 'drone_overlap_table.csv'")


if __name__ == "__main__":
    main()