"""
"""
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
import os
import shutil
import pydeck as pdk
import json

from active_learning.util.rename import get_island_from_folder, rename_images

st.set_page_config(page_title="Mission GUI", layout="wide")

# Create tabs using st.tabs
tabs = st.tabs(["post mission", "mission stats", "iguana_detection"])

# ---------------------------
# POST MISSION TAB
# ---------------------------
with tabs[0]:
    st.header("Post Mission")

    # Input fields for the folder, island, and mission.
    folder_input = st.text_input("Insert folder path", value="/Users/christian/data/2TB/ai-core/data/fake_test_data/Floreana/FLPC02_22012023")
    folder_input = Path(folder_input)
    if folder_input is not None:
        island, mission_code_with_date = get_island_from_folder(folder_input)
        island_text = st.text(f"Island: {island}")
        mission_text = st.text(f"Mission: {mission_code_with_date}")
    folder_output = st.text_input("Insert output path, use same as input to rename", value=folder_input)

    # Placeholder for table
    table_placeholder = st.empty()
    # When "Run" is clicked, generate a DataFrame for demonstration.
    if st.button("Check"):
        if folder_input:
            # For demonstration purposes, we'll simulate a DataFrame.
            # In your actual use case, you might scan folder_input and generate the old/new names.
            # Example: list all files in the folder and then build new names based on island/mission.
            path = Path(folder_input)
            if not path.exists():
                st.error("Folder does not exist.")
            else:
                files = [f for f in path.glob("*") if f.is_file() and not f.name.startswith(".")]
                df_rename_images = rename_images(island, mission_code_with_date, files)
                table_placeholder.dataframe(df_rename_images, use_container_width=True)

                # Save the DataFrame in the session state so it can be used when renaming.
                st.session_state.df_rename_images = df_rename_images
                st.session_state.folder_input = folder_input
                st.session_state.folder_output = folder_output
                st.session_state.island = island
                st.session_state.mission_code_with_date = mission_code_with_date
        else:
            st.error("Please provide a folder path.")

    if folder_input:
        if str(folder_output) == str(folder_input):
            st.write(f"Click 'Rename' for RENAMING images in {folder_input} to {folder_output}.")
        else:
            st.write(f"Click 'Rename' for MOVING images in {folder_input} to {folder_output}.")

    # The "Rename" button, below the table, triggers the actual renaming.
    if st.button("Rename"):
        if "df_rename_images" in st.session_state and st.session_state.df_rename_images is not None:
            df_rename_images = st.session_state.df_rename_images
            input_folder = Path(st.session_state.folder_input)
            folder_output = Path(st.session_state.folder_output)
            # For each row, build the new folder path and rename (move) the file.
            errors = []

            for _, row in df_rename_images.iterrows():
                old_name = row["old_name"]
                new_name = row["new_name"]
                src = input_folder / old_name
                dst_dir = folder_output
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / new_name
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    errors.append(f"Error renaming {src} to {dst}: {e}")
            if errors:
                st.error("Some files could not be renamed:")
                for err in errors:
                    st.write(err)
            else:
                st.success("All files were renamed successfully.")
        else:
            st.warning("Please run the process first to generate the renaming table.")

# ---------------------------
# MISSION STATS TAB
# ---------------------------
with tabs[1]:
    st.header("Mission Stats")
    st.write("Mission stats content goes here...")

    st.header("GeoJSON Map Example with PyDeck")

    # Load the GeoJSON for points. This could be from a file or a string.
    # For demonstration, we'll use an inline string.
    geojson_points_str = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {"name": "Point A"},
          "geometry": {"type": "Point", "coordinates": [102.0, 0.5]}
        },
        {
          "type": "Feature",
          "properties": {"name": "Point B"},
          "geometry": {"type": "Point", "coordinates": [103.0, 1.5]}
        }
      ]
    }
    """

    # Parse the GeoJSON data
    points_data = json.loads(geojson_points_str)

    # Extract coordinates from the point features.
    # Note: GeoJSON coordinates are in the order [longitude, latitude]
    coordinates = [feature["geometry"]["coordinates"] for feature in points_data["features"]]

    # Convert the list to a numpy array for easy manipulation.
    coords_arr = np.array(coordinates)

    # Calculate the mean longitude and latitude.
    center_long, center_lat = coords_arr.mean(axis=0)
    st.write(f"Calculated center: Latitude {center_lat}, Longitude {center_long}")

    # Define a GeoJsonLayer with the points data.
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        points_data,
        pickable=True,
        stroked=True,
        filled=True,
        point_radius_min_pixels=5,
        get_fill_color="[200, 30, 0, 160]",
        get_line_color="[0, 0, 0, 255]",
    )

    # Set initial view state based on the calculated center.
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_long,
        zoom=8,
        pitch=0
    )

    # Create the deck (map) with the layer and initial view state.
    r = pdk.Deck(
        layers=[geojson_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"}
    )

    # Display the map in Streamlit.
    st.pydeck_chart(r)

# ---------------------------
# IGUANA DETECTION TAB
# ---------------------------
with tabs[2]:
    st.header("Iguana Detection")
    st.write("Iguana detection content goes here...")
