"""
flights are often named by their island, site, then flight number
FLights like SCBB01_13012023, SCBB02_13012023, SCBB02a_13012023, SCBB03_13012023, SCBB04_13012023
like belong togeher, while SCBTC01_14012023 is not part of it, yet it might be just 300 meters away.
This was reclustered which often messes up the original membership of flights when original missions are just too close together



"""
import os

import shutil

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN
import numpy as np
import re

gdf_all = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/clustered_points_nadir.shp')
base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above_Organized_Clusters_NADIR")

def get_site_code(s):
    match = re.match(r'^[A-Za-z]+', s)  # Match only leading letters
    return match.group(0) if match else ''

gdf_all["site_code"] = gdf_all["folder_nam"].apply(get_site_code)

# Ensure output column exists
# gdf_all["new_image_path"] = None

# Iterate over clusters
for cluster_id, gdf_cluster in gdf_all.groupby("cluster"):
    if cluster_id == -1:
        logger.info(f"Skipping noise points in cluster -1")
        continue  # Skip unclustered points
    if len(gdf_cluster) < 2000:
        logger.info(f"For now we care only for huge sets")
        continue
    # Use the first row to extract island and date info
    sample_row = gdf_cluster.iloc[0]
    island = sample_row["island"]
    date = sample_row["date"]
    # log the folder names and warn when multiple sites are in the same cluster
    site_codes = gdf_cluster["site_code"].unique()
    # Define the folder structure
    folder_name = f"{island}_{date}_sc_{'_'.join(site_codes)}_Cluster_{cluster_id}"
    folder_path = base_path / folder_name



    # Create directory if it doesn't exist
    folder_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"linking {len(gdf_cluster)} images to {folder_path}")



    gdf_cluster.to_file(folder_path / f"{folder_name}.shp")

    # Move images
    for _, row in gdf_cluster.iterrows():
        old_path = Path(row["filepath"])  # Ensure this column exists
        if pd.isna(old_path) or not old_path.exists():
            logger.warning(f"Skipping missing image: {old_path}")
            continue


        # Define new image path
        new_path = folder_path / old_path.name



        # Create a hard link if it doesn't already exist
        try:
            if not new_path.exists():
                # new_path.symlink_to(old_path)  # Create hard link
                # Copy the file
                shutil.copy(old_path, new_path)
            else:
                logger.info(f"Hard link already exists: {new_path}")

            # Update the database with the new path
            gdf_all.at[row.name, "filepath"] = str(new_path)

        except OSError as e:
            logger.error(f"Failed to create hard link for {old_path}: {e}")

# Save the updated database
output_file = "/Volumes/G-DRIVE/Iguanas_From_Above/updated_database.shp"
gdf_all.to_file(output_file)

logger.info(f"Migration complete. Updated database saved to {output_file}")