"""
Drone Flight Visualisation
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import plotly.express as px
import os
import numpy as np
from shapely.geometry import Point
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide", page_title="Drone Flight Visualization")


# Function to load data
@st.cache_data
def load_data():
    # This function would normally read from a file
    # For this example, we'll assume the data is already available and formatted as shown in the example

    # Replace this with your actual data loading code
    # For example:
    # df = pd.read_csv('your_data.csv')
    # gdf = gpd.GeoDataFrame(
    #     df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    # )

    # For now, we'll create a placeholder to demonstrate the app's functionality
    try:
        gdf = gpd.read_file('drone_flight_data.geojson')
        st.success("Data loaded from GeoJSON file.")
    except:
        st.warning("Could not load data from file. Using example data.")
        # Create dummy data based on the structure from the provided example
        data = {
            'image_name': [f"Fer_FCD02_DJI_{i:04d}_20122021_Nazca.JPG" for i in range(100)],
            'latitude': [-0.3033968888888889 + i * 0.00001 for i in range(100)],
            'longitude': [-91.65105191666667 + i * 0.00002 for i in range(100)],
            'height': [24.0 + (i % 4) for i in range(100)],
            'flight_code': ['FCD02' for i in range(100)],
            'datetime_digitized': pd.date_range(start='2021-12-20 12:34:47', periods=100, freq='3S'),
            'distance_to_prev': [np.nan] + [np.random.uniform(1.5, 6.0) for _ in range(99)],
            'time_diff_seconds': [np.nan] + [3.0 for _ in range(99)],
            'speed_m_per_s': [np.nan] + [d / 3.0 for d in [np.random.uniform(1.5, 6.0) for _ in range(99)]],
            'speed_km_per_h': [np.nan] + [(d / 3.0) * 3.6 for d in [np.random.uniform(1.5, 6.0) for _ in range(99)]],
            'island': ['Fernandina' for i in range(100)],
            'site_code': ['FCD' for i in range(100)],
            'YYYYMMDD': ['2021-12-20' for i in range(100)],
            'filepath': [
                f"/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina/FCD02_20122021/Fer_FCD02_DJI_{i:04d}_20122021_Nazca.JPG"
                for i in range(100)]
        }
        df = pd.DataFrame(data)

        # Create geometry column
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Convert to UTM for accurate distance calculations if needed
    # Assuming you're in zone 15S based on your data
    if not gdf.crs.is_projected:
        st.info("Converting to UTM for accurate measurements")
        gdf = gdf.to_crs(epsg=32715)  # UTM zone 15S

    return gdf


# Function to create a folium map
def create_flight_map(gdf, selected_images=None):
    # Create a copy of the dataframe for map operations
    gdf_map = gdf.copy()

    # Convert to WGS84 for mapping
    if gdf_map.crs.is_projected:
        gdf_map = gdf_map.to_crs(epsg=4326)

    # Get center coordinates
    center_lat = gdf_map.geometry.centroid.y.mean()
    center_lon = gdf_map.geometry.centroid.x.mean()

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')

    # Add flight path line
    coordinates = [(y, x) for x, y in zip(gdf_map.geometry.x, gdf_map.geometry.y)]
    folium.PolyLine(
        coordinates,
        weight=3,
        color='blue',
        opacity=0.7,
        tooltip='Flight Path'
    ).add_to(m)

    # Create marker cluster for all points
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each image
    for idx, row in gdf_map.iterrows():
        # Create popup content
        popup_content = f"""
        <b>Image:</b> {row['image_name']}<br>
        <b>Datetime:</b> {row['datetime_digitized']}<br>
        <b>Height:</b> {row['height']}m<br>
        <b>Speed:</b> {round(row['speed_km_per_h'], 2) if not pd.isna(row['speed_km_per_h']) else 'N/A'} km/h<br>
        <b>Distance to prev:</b> {round(row['distance_to_prev'], 2) if not pd.isna(row['distance_to_prev']) else 'N/A'}m<br>
        """

        # Determine icon color based on selection
        icon_color = 'blue'
        if selected_images is not None and row['image_name'] in selected_images:
            icon_color = 'red'

        # Create marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"Image {idx}: {row['image_name']}",
            icon=folium.Icon(color=icon_color, icon='camera', prefix='fa')
        ).add_to(marker_cluster)

    return m


# Function to display image details
def display_image_details(gdf, image_name):
    row = gdf[gdf['image_name'] == image_name].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Details")
        st.write(f"**Filename:** {row['image_name']}")
        st.write(f"**Date/Time:** {row['datetime_digitized']}")
        st.write(f"**Flight Code:** {row['flight_code']}")
        st.write(f"**Location:** {row['island']} - {row['site_code']}")
        st.write(f"**Coordinates:** {row.geometry.y:.6f}, {row.geometry.x:.6f}")
        st.write(f"**Height:** {row['height']} meters")

        if not pd.isna(row['speed_km_per_h']):
            st.write(f"**Speed:** {row['speed_km_per_h']:.2f} km/h ({row['speed_m_per_s']:.2f} m/s)")
        else:
            st.write("**Speed:** N/A (First image in sequence)")

        if not pd.isna(row['distance_to_prev']):
            st.write(f"**Distance from previous:** {row['distance_to_prev']:.2f} meters")
        else:
            st.write("**Distance from previous:** N/A (First image in sequence)")

    with col2:
        # In a real app, you would load and display the actual image
        # Since we don't have access to the actual files, we'll show a placeholder
        st.subheader("Image Preview")
        file_path = row['filepath']
        try:
            # Try to open the image file (this would work in a real deployment)
            img = Image.open(file_path)
            st.image(img, caption=row['image_name'])
        except:
            # If we can't open the file, show a placeholder message
            st.info(f"Image preview not available. Would show: {file_path}")
            # Create a placeholder image
            placeholder = np.ones((300, 400, 3), dtype=np.uint8) * 200
            # Add text to indicate this is a placeholder
            st.image(placeholder, caption=f"Placeholder for: {row['image_name']}")


# Main app function
def main():
    st.title("Drone Flight Data Visualization")

    # Load the data
    gdf = load_data()

    # Sidebar for filters
    st.sidebar.header("Filters")

    # Filter by flight
    flights = gdf['flight_code'].unique()
    selected_flight = st.sidebar.selectbox("Select Flight", flights)

    # Filter the data
    filtered_gdf = gdf[gdf['flight_code'] == selected_flight]

    # Filter by date
    dates = filtered_gdf['YYYYMMDD'].unique()
    selected_date = st.sidebar.selectbox("Select Date", dates)

    # Further filter the data
    filtered_gdf = filtered_gdf[filtered_gdf['YYYYMMDD'] == selected_date]

    # Selection mode
    selection_mode = st.sidebar.radio(
        "Select images by:",
        ["Height", "Speed", "None"]
    )

    selected_images = None

    if selection_mode == "Height":
        min_height = float(filtered_gdf['height'].min())
        max_height = float(filtered_gdf['height'].max())

        height_range = st.sidebar.slider(
            "Height Range (meters)",
            min_value=min_height,
            max_value=max_height,
            value=(min_height, max_height)
        )

        # Filter images by height range
        height_filtered = filtered_gdf[
            (filtered_gdf['height'] >= height_range[0]) &
            (filtered_gdf['height'] <= height_range[1])
            ]

        selected_images = height_filtered['image_name'].tolist()

    elif selection_mode == "Speed":
        # Filter out NaN speeds (first image in sequence)
        speed_gdf = filtered_gdf[~filtered_gdf['speed_km_per_h'].isna()]

        min_speed = float(speed_gdf['speed_km_per_h'].min())
        max_speed = float(speed_gdf['speed_km_per_h'].max())

        speed_range = st.sidebar.slider(
            "Speed Range (km/h)",
            min_value=min_speed,
            max_value=max_speed,
            value=(min_speed, max_speed)
        )

        # Filter images by speed range
        speed_filtered = filtered_gdf[
            (filtered_gdf['speed_km_per_h'] >= speed_range[0]) &
            (filtered_gdf['speed_km_per_h'] <= speed_range[1])
            ]

        selected_images = speed_filtered['image_name'].tolist()

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Flight Map", "Metrics", "Image Browser"])

    with tab1:
        st.header(f"Flight Path - {selected_flight} ({selected_date})")

        # Create and display map
        flight_map = create_flight_map(filtered_gdf, selected_images)
        folium_static(flight_map, width=1000, height=600)

        # Display statistics about the selected images
        if selected_images:
            st.subheader(f"Selected Images: {len(selected_images)}")
            if selection_mode == "Height":
                st.write(f"Height Range: {height_range[0]} to {height_range[1]} meters")
            elif selection_mode == "Speed":
                st.write(f"Speed Range: {speed_range[0]:.2f} to {speed_range[1]:.2f} km/h")

    with tab2:
        st.header("Flight Metrics")

        # Create three columns for different metrics
        col1, col2 = st.columns(2)

        with col1:
            # Plot height over time
            fig_height = px.line(
                filtered_gdf,
                x='datetime_digitized',
                y='height',
                title='Flight Height Over Time',
                labels={'datetime_digitized': 'Time', 'height': 'Height (meters)'}
            )
            st.plotly_chart(fig_height, use_container_width=True)

            # Distance between shots
            fig_dist = px.line(
                filtered_gdf[~filtered_gdf['distance_to_prev'].isna()],
                x='datetime_digitized',
                y='distance_to_prev',
                title='Distance Between Consecutive Shots',
                labels={'datetime_digitized': 'Time', 'distance_to_prev': 'Distance (meters)'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Plot speed over time
            fig_speed = px.line(
                filtered_gdf[~filtered_gdf['speed_km_per_h'].isna()],
                x='datetime_digitized',
                y='speed_km_per_h',
                title='Flight Speed Over Time',
                labels={'datetime_digitized': 'Time', 'speed_km_per_h': 'Speed (km/h)'}
            )
            st.plotly_chart(fig_speed, use_container_width=True)

            # Histogram of speeds
            fig_speed_hist = px.histogram(
                filtered_gdf[~filtered_gdf['speed_km_per_h'].isna()],
                x='speed_km_per_h',
                nbins=20,
                title='Distribution of Flight Speeds',
                labels={'speed_km_per_h': 'Speed (km/h)'}
            )
            st.plotly_chart(fig_speed_hist, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        stats_cols = st.columns(4)

        with stats_cols[0]:
            st.metric("Total Images", len(filtered_gdf))
            st.metric("Avg. Height (m)", f"{filtered_gdf['height'].mean():.2f}")

        with stats_cols[1]:
            total_distance = filtered_gdf['distance_to_prev'].sum()
            st.metric("Total Distance (m)", f"{total_distance:.2f}")
            st.metric("Avg. Distance (m)", f"{filtered_gdf['distance_to_prev'].mean():.2f}")

        with stats_cols[2]:
            flight_duration = (filtered_gdf['datetime_digitized'].max() - filtered_gdf[
                'datetime_digitized'].min()).total_seconds()
            st.metric("Flight Duration", f"{flight_duration // 60}m {flight_duration % 60}s")
            st.metric("Avg. Speed (km/h)", f"{filtered_gdf['speed_km_per_h'].mean():.2f}")

        with stats_cols[3]:
            st.metric("Max Speed (km/h)", f"{filtered_gdf['speed_km_per_h'].max():.2f}")
            st.metric("Min Speed (km/h)",
                      f"{filtered_gdf['speed_km_per_h'][~filtered_gdf['speed_km_per_h'].isna()].min():.2f}")

    with tab3:
        st.header("Image Browser")

        # Display images based on selection
        images_to_display = filtered_gdf
        if selected_images:
            images_to_display = filtered_gdf[filtered_gdf['image_name'].isin(selected_images)]

        # Show a selectbox with image names
        image_names = images_to_display['image_name'].tolist()

        if image_names:
            selected_image = st.selectbox("Select an image to view details", image_names)

            # Display details of the selected image
            display_image_details(filtered_gdf, selected_image)
        else:
            st.warning("No images match your filter criteria")


if __name__ == "__main__":
    main()